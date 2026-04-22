"""Interactive overlay-region picker.

Usage (from repository root):
    python scripts/define_overlay_regions.py
    python scripts/define_overlay_regions.py --frame drones_images_input/2026-02-15_16-35-56_05934.png
    python scripts/define_overlay_regions.py --output config/overlay_regions.json

For each predefined region name you will be asked to draw a rectangle:
  - Left-click and drag on the image window to define the rectangle.
  - Press ENTER or SPACE to confirm.
  - Press C to skip this region.
  - Press ESC inside the selection to abort the whole session.

After the predefined list, you can add custom named regions.

The resulting JSON file is picked up automatically by:
  - scripts/run_phase2_clean.py    (batch-clean all frames)
  - scripts/validate_phase2.py     (visual validation)
or via an explicit `--regions PATH` flag.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2

from src.preprocessing import OverlayRegion, save_regions_to_json  # noqa: E402


DEFAULT_FRAME = REPO_ROOT / "drones_images_input" / "2026-02-15_16-25-03_04569.png"
DEFAULT_OUTPUT = REPO_ROOT / "config" / "overlay_regions.json"

PREDEFINED_NAMES = (
    "top_status_bar",
    "latlon_text",
    "center_crosshair",
    "bottom_left_panel",
    "bottom_right_telem",
)


def _prompt_roi(window: str, image, title: str) -> Tuple[int, int, int, int]:
    """Show the image and let the user draw one ROI."""
    print(f"\n>>> Draw rectangle for: {title}")
    print("    ENTER/SPACE = confirm | C = skip this region | ESC = abort")
    bbox = cv2.selectROI(window, image, fromCenter=False, showCrosshair=True)
    return tuple(int(v) for v in bbox)  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactively define overlay regions.")
    parser.add_argument("--frame", type=Path, default=DEFAULT_FRAME,
                        help=f"Reference frame for calibration (default: {DEFAULT_FRAME.name}).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output JSON path (default: {DEFAULT_OUTPUT}).")
    parser.add_argument("--no-extra", action="store_true",
                        help="Do not prompt for custom regions after the predefined list.")
    args = parser.parse_args()

    if not args.frame.is_file():
        print(f"ERROR: reference frame not found: {args.frame}")
        return 1

    image = cv2.imread(str(args.frame))
    if image is None:
        print(f"ERROR: could not decode image: {args.frame}")
        return 1

    h, w = image.shape[:2]
    print(f"Reference frame : {args.frame}")
    print(f"Resolution      : {w} x {h}  (this is saved as the calibration size)")
    print(f"Output          : {args.output}")

    window = "Define overlay region"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(w, 1600), min(h, 900))

    regions: List[OverlayRegion] = []

    for name in PREDEFINED_NAMES:
        bbox = _prompt_roi(window, image, name)
        x, y, rw, rh = bbox
        if rw <= 0 or rh <= 0:
            print(f"    skipped {name}")
            continue
        region = OverlayRegion(name=name, x=x, y=y, w=rw, h=rh)
        regions.append(region)
        print(f"    added {region}")

    if not args.no_extra:
        while True:
            cv2.destroyWindow(window)
            cv2.waitKey(1)
            resp = input("\nAdd another (custom) region? [y/N]: ").strip().lower()
            if resp not in ("y", "yes"):
                break
            name = input("    name: ").strip()
            if not name:
                print("    empty name, skipping")
                continue
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, min(w, 1600), min(h, 900))
            bbox = _prompt_roi(window, image, name)
            x, y, rw, rh = bbox
            if rw <= 0 or rh <= 0:
                print(f"    skipped {name}")
                continue
            region = OverlayRegion(name=name, x=x, y=y, w=rw, h=rh)
            regions.append(region)
            print(f"    added {region}")

    cv2.destroyAllWindows()
    cv2.waitKey(1)

    if not regions:
        print("\nNo regions were defined; nothing to save.")
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_regions_to_json(regions, args.output, calibration_size=(w, h))
    print(f"\nSaved {len(regions)} region(s) to {args.output}")
    for r in regions:
        print(f"  - {r.name:20s} x={r.x:4d} y={r.y:4d} w={r.w:4d} h={r.h:4d}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
