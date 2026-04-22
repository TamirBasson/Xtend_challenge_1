"""Phase 2 entry script: remove HUD / telemetry overlay from every frame.

Usage (from repository root):
    python scripts/run_phase2_clean.py
    python scripts/run_phase2_clean.py --method fill

Writes cleaned frames to `outputs/clean_frames/` with identical filenames.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import (  # noqa: E402
    load_frames,
    save_clean_frames,
    DEFAULT_OVERLAY_REGIONS,
    CALIBRATION_SIZE,
    load_regions_from_json,
)


INPUT_FOLDER = REPO_ROOT / "drones_images_input"
OUTPUT_FOLDER = REPO_ROOT / "outputs" / "clean_frames"
DEFAULT_REGIONS_JSON = REPO_ROOT / "config" / "overlay_regions.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 overlay removal")
    parser.add_argument(
        "--method",
        choices=("inpaint", "fill"),
        default="inpaint",
        help="Overlay cleaning strategy (default: inpaint).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FOLDER,
        help=f"Output folder (default: {OUTPUT_FOLDER}).",
    )
    parser.add_argument(
        "--regions",
        type=Path,
        default=None,
        help=(
            "Path to JSON file with overlay regions. "
            f"Defaults to {DEFAULT_REGIONS_JSON} if present, "
            "otherwise the built-in DEFAULT_OVERLAY_REGIONS are used."
        ),
    )
    args = parser.parse_args()

    regions_path = args.regions
    if regions_path is None and DEFAULT_REGIONS_JSON.is_file():
        regions_path = DEFAULT_REGIONS_JSON

    if regions_path is not None:
        loaded = load_regions_from_json(regions_path, include_per_image=True)
        regions, calibration_size, per_image_regions = loaded
        print(f"Overlay regions : loaded {len(regions)} global from {regions_path}")
        print(f"Per-image sets  : {len(per_image_regions)}")
    else:
        regions = list(DEFAULT_OVERLAY_REGIONS)
        calibration_size = CALIBRATION_SIZE
        per_image_regions = {}
        print(f"Overlay regions : using built-in defaults ({len(regions)} regions)")

    frames = load_frames(INPUT_FOLDER)
    print(f"Loaded {len(frames)} frames from {INPUT_FOLDER}")
    print(f"Cleaning method : {args.method}")
    print(f"Output folder   : {args.output}")

    written = save_clean_frames(
        frames, args.output,
        method=args.method,
        regions=regions,
        calibration_size=calibration_size,
        per_image_regions=per_image_regions,
    )

    print(f"\nWrote {len(written)} cleaned frames:")
    for p in written:
        print(f"  - {p.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
