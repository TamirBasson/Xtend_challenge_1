"""Phase 1 validation script.

Performs all required sanity checks on the data loading step:
  1. Loads images from `drones_images_input/`.
  2. Prints the total count and the first 3 file paths.
  3. Saves a grid preview of the first images using `show_grid`.
  4. Asserts every image is non-empty, has a valid (H, W, C) shape,
     and decodes without errors.
  5. Asserts that at least one image was loaded.

Results (preview image) are written to:
    outputs/phase1_preview.png
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

_original_show = plt.show
def _save_and_close(*_args, **_kwargs):
    """Redirect `plt.show()` to saving the active figures to disk."""
    for num in plt.get_fignums():
        fig = plt.figure(num)
        out_path = OUTPUT_DIR / f"phase1_preview_fig{num}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"    saved preview -> {out_path}")
    plt.close("all")
plt.show = _save_and_close

from src import load_frames, iterate_frames, show_grid  # noqa: E402


INPUT_FOLDER = REPO_ROOT / "drones_images_input"


def main() -> int:
    print("=" * 60)
    print(" PHASE 1 VALIDATION")
    print("=" * 60)
    print(f"Input folder: {INPUT_FOLDER}")

    try:
        frames = load_frames(INPUT_FOLDER)
    except Exception as exc:
        print(f"[FAIL] load_frames raised: {exc}")
        traceback.print_exc()
        return 1

    print(f"\n[1] Total number of images found: {len(frames)}")

    assert len(frames) > 0, "No images were loaded from the input folder."

    print("\n[2] First 3 image file paths:")
    for frame in frames[:3]:
        print(f"    - {frame.path}")

    print("\n[3] Decoding and shape validation:")
    errors: list[str] = []
    for frame, image in iterate_frames(frames):
        if image is None:
            errors.append(f"{frame.name}: decoder returned None")
            continue
        if image.size == 0:
            errors.append(f"{frame.name}: image is empty")
            continue
        if image.ndim != 3:
            errors.append(f"{frame.name}: expected 3 dims, got {image.ndim}")
            continue
        h, w, c = image.shape
        if h <= 0 or w <= 0 or c not in (1, 3, 4):
            errors.append(f"{frame.name}: invalid shape {image.shape}")
            continue
        print(f"    [{frame.index:02d}] {frame.name}  ->  shape={image.shape}, dtype={image.dtype}")

    if errors:
        print("\n[FAIL] Validation errors encountered:")
        for e in errors:
            print(f"    - {e}")
        return 2

    print("\n[4] All images are non-empty and have valid (H, W, C) shapes.")

    print("\n[5] Rendering grid preview of first 3 frames ...")
    show_grid(frames, max_images=3, cols=3)

    print("\n[6] Rendering grid preview of all frames ...")
    show_grid(frames, max_images=len(frames), cols=4)

    print("\n" + "=" * 60)
    print(" PHASE 1 VALIDATION: PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
