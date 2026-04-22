"""Phase 1 entry script: load the dataset and preview sample frames.

Usage (from repository root):
    python scripts/run_phase1_preview.py

This script:
1. Loads all frames from `drones_images_input/`.
2. Prints the total number of frames and their filenames.
3. Displays up to 6 sample frames in a grid for visual inspection.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import load_frames, iterate_frames, show_grid  # noqa: E402


INPUT_FOLDER = REPO_ROOT / "drones_images_input"
MAX_PREVIEW = 6


def main() -> int:
    frames = load_frames(INPUT_FOLDER)

    print(f"Input folder : {INPUT_FOLDER}")
    print(f"Frames found : {len(frames)}")
    for frame in frames:
        print(f"  [{frame.index:02d}] {frame.name}")

    if not frames:
        print("No frames found. Nothing to preview.")
        return 1

    print("\nValidating that all frames can be decoded ...")
    for frame, image in iterate_frames(frames):
        h, w = image.shape[:2]
        print(f"  [{frame.index:02d}] {frame.name}  ->  {w} x {h}")

    print(f"\nDisplaying first {min(MAX_PREVIEW, len(frames))} frames ...")
    show_grid(frames, max_images=MAX_PREVIEW, cols=3)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
