"""Deep-learning feature extraction (SuperPoint) — new pipeline.

This module is an **additive** alternative to `src.features` (SIFT/AKAZE).
The SIFT pipeline is not modified; callers opt in explicitly by choosing
`method="superpoint"`.

The return type is the existing `FeatureSet` so that downstream code
(`matching`, `geometry`, `transfer`) keeps working without changes:

  * `keypoints`  : List[cv2.KeyPoint]   (pt = (x, y); response = SP score)
  * `descriptors`: np.ndarray (N, 256)  float32, L2-normalized by SuperPoint
  * `image_shape`: (H, W)
  * `method`     : "superpoint"

Heavy deps (torch, lightglue) are imported lazily so that SIFT-only users
are not forced to install them.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .features import FeatureSet


# Cached singletons so repeated calls don't reload the model.
_DEVICE = None
_EXTRACTOR = None


def get_device():
    """Return the torch device (CUDA if available, else CPU). Cached."""
    global _DEVICE
    if _DEVICE is None:
        import torch
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def get_superpoint_extractor(max_keypoints: int = 2048):
    """Return a cached SuperPoint extractor on the resolved device."""
    global _EXTRACTOR
    if _EXTRACTOR is None:
        try:
            from lightglue import SuperPoint
        except ImportError as e:
            raise ImportError(
                "SuperPoint requires the 'lightglue' package.\n"
                "Install with: pip install git+https://github.com/cvg/LightGlue.git"
            ) from e
        device = get_device()
        _EXTRACTOR = (
            SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
        )
    return _EXTRACTOR


def _image_to_tensor(image: np.ndarray, device):
    """Convert a BGR/gray uint8 image to a (1, C, H, W) float tensor in [0, 1]."""
    import torch

    if image.ndim == 3:
        # OpenCV loads BGR; LightGlue's SuperPoint converts RGB -> gray internally.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0  # (C, H, W)
    return tensor.unsqueeze(0).to(device)                           # (1, C, H, W)


def _apply_mask(
    keypoints_xy: np.ndarray,
    descriptors: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop keypoints that fall on mask==0 pixels (overlay/HUD areas)."""
    if keypoints_xy.size == 0:
        return keypoints_xy, descriptors, scores
    h, w = mask.shape[:2]
    xs = np.clip(keypoints_xy[:, 0].astype(int), 0, w - 1)
    ys = np.clip(keypoints_xy[:, 1].astype(int), 0, h - 1)
    keep = mask[ys, xs] > 0
    return keypoints_xy[keep], descriptors[keep], scores[keep]


def extract_superpoint(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    frame_name: str = "",
    max_keypoints: int = 2048,
) -> FeatureSet:
    """Extract SuperPoint keypoints + descriptors from a single image.

    Parameters
    ----------
    image        : BGR or grayscale uint8 image.
    mask         : optional uint8 mask (255 = allowed, 0 = ignored),
                   applied *after* detection by dropping keypoints that
                   fall on disallowed pixels.
    frame_name   : bookkeeping label copied into the `FeatureSet`.
    max_keypoints: cap on keypoints returned by SuperPoint (top-K by score).
    """
    import torch

    device = get_device()
    extractor = get_superpoint_extractor(max_keypoints=max_keypoints)

    img_tensor = _image_to_tensor(image, device=device)
    with torch.no_grad():
        feats = extractor.extract(img_tensor)

    # feats: dict with batched tensors; peel the batch dim.
    kps_t = feats["keypoints"][0]                  # (N, 2) float32
    desc_t = feats["descriptors"][0]               # (N, 256) float32
    scores_t = feats["keypoint_scores"][0]         # (N,) float32

    kps = kps_t.detach().cpu().numpy().astype(np.float32)
    desc = desc_t.detach().cpu().numpy().astype(np.float32)
    scores = scores_t.detach().cpu().numpy().astype(np.float32)

    if mask is not None:
        kps, desc, scores = _apply_mask(kps, desc, scores, mask)

    # Build cv2.KeyPoint objects so downstream code that uses `.pt` / `.response`
    # keeps working (apply_grid_filter, draw_*, points_a/points_b all rely on this).
    cv_kps = [
        cv2.KeyPoint(float(x), float(y), 1.0, -1.0, float(s))
        for (x, y), s in zip(kps, scores)
    ]

    h, w = image.shape[:2]
    return FeatureSet(
        frame_name=frame_name,
        method="superpoint",
        image_shape=(h, w),
        keypoints=cv_kps,
        descriptors=desc,
    )
