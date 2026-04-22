"""Phase 2 validation script.

Performs a visual sanity check on overlay removal:
  1. Loads the first frame.
  2. Builds the overlay mask.
  3. Produces a cleaned frame (inpaint method).
  4. Saves a side-by-side figure [raw | mask overlay | cleaned].
  5. Also saves a grid of cleaned versions of the first 6 frames.

Outputs are written to:
    outputs/phase2_mask_overlay.png   (raw + rectangles highlighted)
    outputs/phase2_triplet.png        (raw | mask | clean)
    outputs/phase2_clean_grid.png     (6 cleaned frames)
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np

from src import load_frames, build_overlay_mask, clean_frame, load_regions_from_json  # noqa: E402
from src.preprocessing import DEFAULT_OVERLAY_REGIONS, _scale_regions, CALIBRATION_SIZE  # noqa: E402


INPUT_FOLDER = REPO_ROOT / "drones_images_input"
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
DEFAULT_REGIONS_JSON = REPO_ROOT / "config" / "overlay_regions.json"


def _resolve_regions(explicit_path: Path | None):
    """Pick regions + calibration size, preferring (1) explicit flag, (2) default JSON, (3) built-ins."""
    import argparse  # noqa: F401 - only used for CLI above

    path = explicit_path
    if path is None and DEFAULT_REGIONS_JSON.is_file():
        path = DEFAULT_REGIONS_JSON

    if path is not None:
        regions, cal = load_regions_from_json(path)
        print(f"Overlay regions : loaded {len(regions)} from {path}")
        return regions, cal

    print(f"Overlay regions : using built-in defaults ({len(DEFAULT_OVERLAY_REGIONS)} regions)")
    return list(DEFAULT_OVERLAY_REGIONS), CALIBRATION_SIZE


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _draw_regions(image: np.ndarray, regions, calibration_size) -> np.ndarray:
    """Return a copy of `image` with overlay rectangles drawn in red."""
    h, w = image.shape[:2]
    annotated = image.copy()
    for r in _scale_regions(regions, (w, h), calibration_size):
        cv2.rectangle(
            annotated, (r.x, r.y), (r.x + r.w, r.y + r.h),
            color=(0, 0, 255), thickness=2,
        )
        cv2.putText(
            annotated, r.name, (r.x + 4, max(r.y + 14, 14)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA,
        )
    return annotated


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2 overlay-removal validation")
    parser.add_argument("--regions", type=Path, default=None,
                        help="Path to overlay-regions JSON (defaults to config/overlay_regions.json if present).")
    args = parser.parse_args()

    print("=" * 60)
    print(" PHASE 2 VALIDATION - Overlay Removal")
    print("=" * 60)

    regions, calibration_size = _resolve_regions(args.regions)

    frames = load_frames(INPUT_FOLDER)
    assert len(frames) > 0, "No input frames found."
    print(f"Loaded {len(frames)} frames.")

    first = frames[0]
    image = first.load_image()
    print(f"Sample frame : {first.name}  shape={image.shape}")

    mask = build_overlay_mask(image, regions=regions, calibration_size=calibration_size)
    overlay_fraction = float(mask.mean()) / 255.0
    print(f"Overlay mask covers {overlay_fraction * 100:.2f}% of image area.")

    clean, _ = clean_frame(image, method="inpaint",
                           regions=regions, calibration_size=calibration_size)
    assert clean.shape == image.shape, "Cleaned image shape mismatch."
    assert clean.dtype == image.dtype, "Cleaned image dtype mismatch."

    diff_in_mask = np.abs(clean.astype(int) - image.astype(int)).sum(axis=2)
    changed_inside = int((diff_in_mask[mask == 255] > 0).sum())
    total_mask_pixels = int((mask == 255).sum())
    unchanged_outside = int((diff_in_mask[mask == 0] == 0).sum())
    total_outside = int((mask == 0).sum())
    print(f"Pixels changed inside mask  : {changed_inside}/{total_mask_pixels}")
    print(f"Pixels unchanged outside    : {unchanged_outside}/{total_outside}")
    assert total_outside == 0 or unchanged_outside == total_outside, (
        "Cleaning modified pixels outside the overlay mask."
    )

    annotated = _draw_regions(image, regions, calibration_size)
    plt.figure(figsize=(10, 6))
    plt.imshow(_bgr_to_rgb(annotated))
    plt.title("Overlay regions (red rectangles)")
    plt.axis("off")
    plt.tight_layout()
    out1 = OUTPUT_DIR / "phase2_mask_overlay.png"
    plt.savefig(out1, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {out1}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(_bgr_to_rgb(image));          axes[0].set_title("Raw");          axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray");           axes[1].set_title("Overlay mask"); axes[1].axis("off")
    axes[2].imshow(_bgr_to_rgb(clean));          axes[2].set_title("Cleaned (inpaint)"); axes[2].axis("off")
    plt.tight_layout()
    out2 = OUTPUT_DIR / "phase2_triplet.png"
    plt.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {out2}")

    sample_count = min(6, len(frames))
    cols = 3
    rows = (sample_count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, frame in zip(axes, frames[:sample_count]):
        img = frame.load_image()
        cleaned, _ = clean_frame(img, method="inpaint",
                                 regions=regions, calibration_size=calibration_size)
        ax.imshow(_bgr_to_rgb(cleaned))
        ax.set_title(f"[{frame.index}] {frame.name}", fontsize=9)
        ax.axis("off")
    for ax in axes[sample_count:]:
        ax.axis("off")
    plt.tight_layout()
    out3 = OUTPUT_DIR / "phase2_clean_grid.png"
    plt.savefig(out3, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {out3}")

    print("\n" + "=" * 60)
    print(" PHASE 2 VALIDATION: PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
