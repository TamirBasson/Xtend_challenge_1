"""Phase 3 feature extraction.

Scope is intentionally limited to keypoint + descriptor extraction on
cleaned frames. Matching is handled by a later module.

Default detector is SIFT (robust to scale/rotation, no extra dependencies).
AKAZE is exposed as an alternative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .frame_loader import Frame
from .preprocessing import (
    DEFAULT_OVERLAY_REGIONS,
    CALIBRATION_SIZE,
    OverlayRegion,
    build_overlay_mask,
)


SUPPORTED_METHODS = ("sift", "akaze", "superpoint")


@dataclass
class FeatureSet:
    """Keypoints + descriptors extracted from a single frame."""
    frame_name: str
    method: str
    image_shape: Tuple[int, int]
    keypoints: List[cv2.KeyPoint] = field(default_factory=list)
    descriptors: Optional[np.ndarray] = None

    @property
    def num_keypoints(self) -> int:
        return len(self.keypoints)


def _make_detector(method: str):
    """Instantiate an OpenCV feature detector for the given method."""
    m = method.lower()
    if m == "sift":
        return cv2.SIFT_create()
    if m == "akaze":
        return cv2.AKAZE_create()
    raise ValueError(f"Unsupported method {method!r}; expected one of {SUPPORTED_METHODS}")


def build_detection_mask(
    image: np.ndarray,
    regions: Sequence[OverlayRegion] = DEFAULT_OVERLAY_REGIONS,
    calibration_size: Tuple[int, int] = CALIBRATION_SIZE,
) -> np.ndarray:
    """Return a uint8 mask where 255 = 'allowed for keypoint detection'.

    Pixels inside overlay regions (HUD / telemetry) are set to 0 so the
    detector ignores them, even after inpainting.
    """
    overlay = build_overlay_mask(image, regions=regions, calibration_size=calibration_size)
    return cv2.bitwise_not(overlay)


def extract_features(
    image: np.ndarray,
    method: str = "sift",
    mask: Optional[np.ndarray] = None,
    frame_name: str = "",
) -> FeatureSet:
    """Detect keypoints and compute descriptors on a single image.

    Parameters
    ----------
    image : BGR or grayscale uint8 image.
    method : "sift", "akaze", or "superpoint".
    mask : optional uint8 mask (255 = allowed, 0 = ignored).
    frame_name : used only for bookkeeping / debug prints.

    Note on "superpoint"
    --------------------
    Dispatches to `src.deep_features.extract_superpoint` (lazy import so
    torch/lightglue are not required for the SIFT/AKAZE paths). Returns
    a `FeatureSet` with 256-D float32 descriptors — downstream code that
    treats descriptor-dim as opaque (matching / geometry / transfer) is
    unaffected.
    """
    if method.lower() == "superpoint":
        from .deep_features import extract_superpoint
        return extract_superpoint(image, mask=mask, frame_name=frame_name)

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    detector = _make_detector(method)
    keypoints, descriptors = detector.detectAndCompute(gray, mask)

    return FeatureSet(
        frame_name=frame_name,
        method=method.lower(),
        image_shape=gray.shape[:2],
        keypoints=list(keypoints) if keypoints is not None else [],
        descriptors=descriptors,
    )


def extract_features_for_frames(
    frames: Iterable[Frame],
    method: str = "sift",
    use_mask: bool = True,
    regions: Sequence[OverlayRegion] = DEFAULT_OVERLAY_REGIONS,
    calibration_size: Tuple[int, int] = CALIBRATION_SIZE,
    source_dir: Optional[str | Path] = None,
) -> List[FeatureSet]:
    """Extract features for each frame and return a list of `FeatureSet`.

    Parameters
    ----------
    frames : iterable of `Frame` objects (filename is preserved).
    method : detector name.
    use_mask : if True, ignores overlay regions during detection.
    regions, calibration_size : overlay definitions (passed to the mask builder).
    source_dir : if provided, load the image from `<source_dir>/<frame.name>`
                 instead of the original input path. Use this to run on
                 the cleaned frames produced by Phase 2.
    """
    source_dir = Path(source_dir) if source_dir is not None else None
    results: List[FeatureSet] = []

    for frame in frames:
        if source_dir is not None:
            path = source_dir / frame.name
            image = cv2.imread(str(path))
            if image is None:
                raise IOError(f"Failed to read cleaned frame: {path}")
        else:
            image = frame.load_image()

        mask = None
        if use_mask:
            mask = build_detection_mask(
                image, regions=regions, calibration_size=calibration_size,
            )

        fs = extract_features(image, method=method, mask=mask, frame_name=frame.name)
        results.append(fs)

    return results


def apply_grid_filter(
    fs: FeatureSet,
    grid_rows: int = 4,
    grid_cols: int = 5,
    max_per_cell: int = 15,
) -> FeatureSet:
    """Return a new FeatureSet keeping at most max_per_cell keypoints per
    spatial grid cell, selected by highest SIFT response.

    This enforces spatial diversity before matching: dense regions (e.g. the
    horizon/treeline band) are sub-sampled to the same quota as sparse regions
    (sky, bare ground), redistributing the match budget across the full image.

    The input FeatureSet is not mutated. Descriptor rows are re-indexed to
    match the surviving keypoints so the returned FeatureSet is self-consistent
    and can be passed directly to a matcher.

    Parameters
    ----------
    fs            : source FeatureSet (keypoints + descriptors must be set).
    grid_rows     : number of rows in the spatial grid.
    grid_cols     : number of columns in the spatial grid.
    max_per_cell  : maximum keypoints to retain per cell (by response strength).
    """
    if not fs.keypoints or fs.descriptors is None:
        return fs  # nothing to filter; return as-is

    h, w = fs.image_shape
    cell_h = h / grid_rows
    cell_w = w / grid_cols

    kept_indices: List[int] = []
    for row in range(grid_rows):
        y0, y1 = row * cell_h, (row + 1) * cell_h
        for col in range(grid_cols):
            x0, x1 = col * cell_w, (col + 1) * cell_w
            cell_kp_idx = [
                i for i, kp in enumerate(fs.keypoints)
                if x0 <= kp.pt[0] < x1 and y0 <= kp.pt[1] < y1
            ]
            # sort by descending response; keep the strongest max_per_cell
            cell_kp_idx.sort(key=lambda i: fs.keypoints[i].response, reverse=True)
            kept_indices.extend(cell_kp_idx[:max_per_cell])

    kept_indices = sorted(set(kept_indices))
    filtered_kps = [fs.keypoints[i] for i in kept_indices]
    filtered_desc = fs.descriptors[kept_indices]

    return FeatureSet(
        frame_name=fs.frame_name,
        method=fs.method,
        image_shape=fs.image_shape,
        keypoints=filtered_kps,
        descriptors=filtered_desc,
    )


def draw_keypoints(
    image: np.ndarray,
    feature_set: FeatureSet,
    color: Tuple[int, int, int] = (0, 255, 0),
    rich: bool = True,
) -> np.ndarray:
    """Return a copy of `image` with keypoints drawn.

    If `rich=True` (default) each keypoint is drawn with size and orientation
    (useful to visually assess scale/rotation distribution).
    """
    flags = (
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        if rich else cv2.DRAW_MATCHES_FLAGS_DEFAULT
    )
    return cv2.drawKeypoints(image, feature_set.keypoints, None, color=color, flags=flags)
