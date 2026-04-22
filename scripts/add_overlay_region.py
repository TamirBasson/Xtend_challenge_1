"""Append a single image-specific overlay region to overlay_regions.json.

Unlike `scripts/define_overlay_regions.py` (which writes the full list from
scratch), this helper incrementally extends the per-image region map by drawing
ONE additional rectangle for a specific image filename.

Usage (from repository root):
    python scripts/add_overlay_region.py
    python scripts/add_overlay_region.py --name center_crosshair
    python scripts/add_overlay_region.py --frame drones_images_input/2026-02-15_16-25-03_04569.png
    python scripts/add_overlay_region.py --regions config/overlay_regions.json

Controls in the selection window:
  - Left-click and drag to define the rectangle.
  - Press ENTER or SPACE to confirm.
  - Press C or ESC to abort without saving.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2

from src.preprocessing import (  # noqa: E402
    OverlayRegion,
    load_regions_from_json,
    save_regions_to_json,
)


DEFAULT_FRAME = REPO_ROOT / "drones_images_input" / "2026-02-15_16-25-03_04569.png"
DEFAULT_REGIONS = REPO_ROOT / "config" / "overlay_regions.json"


def _prompt_roi(window: str, image, title: str) -> Tuple[int, int, int, int]:
    print(f"\n>>> Draw rectangle for: {title}")
    print("    ENTER/SPACE = confirm | C/ESC = abort")
    bbox = cv2.selectROI(window, image, fromCenter=False, showCrosshair=True)
    return tuple(int(v) for v in bbox)  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Append a single overlay region to an existing JSON file."
    )
    parser.add_argument("--name", type=str, default=None,
                        help="Name for the new region (prompted if omitted).")
    parser.add_argument("--frame", type=Path, default=DEFAULT_FRAME,
                        help=f"Reference frame to draw on (default: {DEFAULT_FRAME.name}).")
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGIONS,
                        help=f"Existing regions JSON to extend (default: {DEFAULT_REGIONS}).")
    args = parser.parse_args()

    if not args.frame.is_file():
        print(f"ERROR: reference frame not found: {args.frame}")
        return 1
    if not args.regions.is_file():
        print(f"ERROR: regions file not found: {args.regions}")
        print("       Run scripts/define_overlay_regions.py first to create it.")
        return 1

    loaded = load_regions_from_json(args.regions, include_per_image=True)
    existing, calibration_size, per_image_regions = loaded
    print(f"Loaded {len(existing)} global region(s) from {args.regions}")
    for r in existing:
        print(f"  - {r.name:20s} x={r.x:4d} y={r.y:4d} w={r.w:4d} h={r.h:4d}")
    print(f"Per-image keys    : {len(per_image_regions)}")
    print(f"Calibration size : {calibration_size[0]} x {calibration_size[1]}")

    image = cv2.imread(str(args.frame))
    if image is None:
        print(f"ERROR: could not decode image: {args.frame}")
        return 1

    h, w = image.shape[:2]
    print(f"Reference frame  : {args.frame}")
    print(f"Frame key        : {args.frame.name}")
    print(f"Frame resolution : {w} x {h}")

    if (w, h) != calibration_size:
        print("WARNING: reference frame resolution differs from calibration size.")
        print("         The drawn rectangle will be saved in frame pixels and the")
        print("         calibration size in the JSON will be kept as-is. Make sure")
        print("         you pick a frame at the calibration resolution, otherwise")
        print("         update --frame or re-run define_overlay_regions.py.")

    name = args.name
    if not name:
        name = input("Name for the new region: ").strip()
    if not name:
        print("ERROR: empty region name.")
        return 2
    frame_key = args.frame.name
    frame_regions: List[OverlayRegion] = list(per_image_regions.get(frame_key, []))
    if any(r.name == name for r in frame_regions):
        print(f"ERROR: region name {name!r} already exists for frame {frame_key!r}.")
        return 2

    window = "Add overlay region"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(w, 1600), min(h, 900))

    bbox = _prompt_roi(window, image, name)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    x, y, rw, rh = bbox
    if rw <= 0 or rh <= 0:
        print("No rectangle drawn; nothing saved.")
        return 2

    new_region = OverlayRegion(name=name, x=x, y=y, w=rw, h=rh)
    frame_regions.append(new_region)
    per_image_regions[frame_key] = frame_regions

    save_regions_to_json(
        existing,
        args.regions,
        calibration_size=calibration_size,
        per_image_regions=per_image_regions,
    )
    print(f"\nAppended image-specific region to {args.regions}")
    print(f"  frame={frame_key}")
    print(f"  + {new_region.name:20s} x={new_region.x:4d} y={new_region.y:4d} "
          f"w={new_region.w:4d} h={new_region.h:4d}")
    print(f"Total regions for frame now: {len(frame_regions)}")
    print(f"Global regions unchanged    : {len(existing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
