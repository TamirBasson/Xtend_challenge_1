"""Phase 2 preprocessing: overlay (HUD / telemetry) removal.

Scope is intentionally limited to overlay removal. Contrast normalization
and other preprocessing steps are deferred to later phases.

The HUD overlay is static across all frames (same resolution, same layout),
so a fixed set of axis-aligned rectangles is sufficient to describe it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from .frame_loader import Frame


@dataclass(frozen=True)
class OverlayRegion:
    """Axis-aligned rectangle describing an overlay area.

    Coordinates are (x, y, w, h) in pixels, calibrated against
    `CALIBRATION_SIZE`. They are scaled automatically if the input
    frame has a different resolution.
    """
    name: str
    x: int
    y: int
    w: int
    h: int


CALIBRATION_SIZE: Tuple[int, int] = (1280, 720)


DEFAULT_OVERLAY_REGIONS: Tuple[OverlayRegion, ...] = (
    OverlayRegion("top_status_bar",        0,   0, 1280,  50),
    OverlayRegion("latlon_text",          35,  58,  180,  50),
    OverlayRegion("center_crosshair",    395, 320,  470,  55),
    OverlayRegion("bottom_left_panel",     0, 635,  360,  85),
    OverlayRegion("bottom_right_telem",  748, 675,  532,  45),
)


CLEAN_METHODS = ("inpaint", "fill")
PerImageRegions = Dict[str, List[OverlayRegion]]


def _scale_regions(
    regions: Sequence[OverlayRegion],
    image_size: Tuple[int, int],
    calibration_size: Tuple[int, int],
) -> List[OverlayRegion]:
    """Scale regions from calibration resolution to the given image size."""
    cal_w, cal_h = calibration_size
    img_w, img_h = image_size
    if (cal_w, cal_h) == (img_w, img_h):
        return list(regions)

    sx = img_w / cal_w
    sy = img_h / cal_h
    scaled: List[OverlayRegion] = []
    for r in regions:
        scaled.append(OverlayRegion(
            name=r.name,
            x=int(round(r.x * sx)),
            y=int(round(r.y * sy)),
            w=int(round(r.w * sx)),
            h=int(round(r.h * sy)),
        ))
    return scaled


def build_overlay_mask(
    image: np.ndarray,
    regions: Sequence[OverlayRegion] = DEFAULT_OVERLAY_REGIONS,
    calibration_size: Tuple[int, int] = CALIBRATION_SIZE,
) -> np.ndarray:
    """Return a uint8 single-channel mask where 255 marks overlay pixels."""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for r in _scale_regions(regions, (w, h), calibration_size):
        x0 = max(0, r.x)
        y0 = max(0, r.y)
        x1 = min(w, r.x + r.w)
        y1 = min(h, r.y + r.h)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255

    return mask


def clean_frame(
    image: np.ndarray,
    method: str = "inpaint",
    regions: Sequence[OverlayRegion] = DEFAULT_OVERLAY_REGIONS,
    calibration_size: Tuple[int, int] = CALIBRATION_SIZE,
    inpaint_radius: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (clean_image, mask) for a single frame.

    Parameters
    ----------
    image : BGR uint8 image.
    method : "inpaint" (TELEA inpainting) or "fill" (black fill).
    """
    if method not in CLEAN_METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {CLEAN_METHODS}")

    mask = build_overlay_mask(image, regions=regions, calibration_size=calibration_size)

    if method == "inpaint":
        clean = cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)
    else:
        clean = image.copy()
        clean[mask == 255] = 0

    return clean, mask


def resolve_regions_for_frame(
    frame_name: str,
    global_regions: Sequence[OverlayRegion],
    per_image_regions: PerImageRegions | None = None,
) -> List[OverlayRegion]:
    """Return effective regions for a frame as global + frame-specific additions."""
    effective = list(global_regions)
    if per_image_regions:
        specific = per_image_regions.get(frame_name)
        if specific:
            effective.extend(specific)
    return effective


def save_regions_to_json(
    regions: Sequence[OverlayRegion],
    path: str | Path,
    calibration_size: Tuple[int, int] = CALIBRATION_SIZE,
    per_image_regions: PerImageRegions | None = None,
) -> Path:
    """Serialize global regions (+ optional per-image regions) to JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "calibration_size": {"width": int(calibration_size[0]),
                             "height": int(calibration_size[1])},
        "regions": [
            {"name": r.name, "x": r.x, "y": r.y, "w": r.w, "h": r.h}
            for r in regions
        ],
    }
    if per_image_regions:
        data["per_image_regions"] = {
            str(frame_name): [
                {"name": r.name, "x": r.x, "y": r.y, "w": r.w, "h": r.h}
                for r in frame_regions
            ]
            for frame_name, frame_regions in per_image_regions.items()
        }
    out.write_text(json.dumps(data, indent=2))
    return out


def load_regions_from_json(
    path: str | Path,
    include_per_image: bool = False,
) -> Tuple[List[OverlayRegion], Tuple[int, int]] | Tuple[List[OverlayRegion], Tuple[int, int], PerImageRegions]:
    """Load regions and their calibration size from a JSON file.

    Returns
    -------
    (regions, (calibration_width, calibration_height))
    or, when `include_per_image=True`:
    (regions, (calibration_width, calibration_height), per_image_regions)
    """
    p = Path(path)
    data = json.loads(p.read_text())
    cal = data.get("calibration_size", {})
    cal_w = int(cal.get("width", CALIBRATION_SIZE[0]))
    cal_h = int(cal.get("height", CALIBRATION_SIZE[1]))
    regions = [
        OverlayRegion(
            name=str(r["name"]),
            x=int(r["x"]), y=int(r["y"]),
            w=int(r["w"]), h=int(r["h"]),
        )
        for r in data.get("regions", [])
    ]
    if not include_per_image:
        return regions, (cal_w, cal_h)

    per_image_raw = data.get("per_image_regions", {})
    per_image_regions: PerImageRegions = {}
    if isinstance(per_image_raw, dict):
        for frame_name, items in per_image_raw.items():
            if not isinstance(items, list):
                continue
            parsed: List[OverlayRegion] = []
            for r in items:
                if not isinstance(r, dict):
                    continue
                parsed.append(
                    OverlayRegion(
                        name=str(r["name"]),
                        x=int(r["x"]), y=int(r["y"]),
                        w=int(r["w"]), h=int(r["h"]),
                    )
                )
            per_image_regions[str(frame_name)] = parsed

    return regions, (cal_w, cal_h), per_image_regions


def save_clean_frames(
    frames: Iterable[Frame],
    output_dir: str | Path,
    method: str = "inpaint",
    regions: Sequence[OverlayRegion] = DEFAULT_OVERLAY_REGIONS,
    calibration_size: Tuple[int, int] = CALIBRATION_SIZE,
    per_image_regions: PerImageRegions | None = None,
) -> List[Path]:
    """Clean each frame and write it to `output_dir`, preserving filenames.

    Returns the list of output file paths, in frame order.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for frame in frames:
        image = frame.load_image()
        frame_regions = resolve_regions_for_frame(
            frame_name=frame.name,
            global_regions=regions,
            per_image_regions=per_image_regions,
        )
        clean, _mask = clean_frame(
            image,
            method=method,
            regions=frame_regions,
            calibration_size=calibration_size,
        )
        out_path = out_dir / frame.name
        ok = cv2.imwrite(str(out_path), clean)
        if not ok:
            raise IOError(f"Failed to write cleaned frame: {out_path}")
        written.append(out_path)

    return written
