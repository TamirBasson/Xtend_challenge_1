"""Phase 3 entry script: extract features from cleaned frames.

Usage (from repository root):
    python scripts/run_phase3_features.py
    python scripts/run_phase3_features.py --method superpoint
    python scripts/run_phase3_features.py --source drones_images_input

Writes a keypoint-overlay grid to:
    outputs/phase3_keypoints_grid.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import (  # noqa: E402
    load_frames,
    extract_features_for_frames,
    draw_keypoints,
    DEFAULT_OVERLAY_REGIONS,
    CALIBRATION_SIZE,
    load_regions_from_json,
)


INPUT_FOLDER = REPO_ROOT / "drones_images_input"
CLEAN_FOLDER = REPO_ROOT / "outputs" / "clean_frames"
OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_REGIONS_JSON = REPO_ROOT / "config" / "overlay_regions.json"


def _resolve_regions(explicit_path):
    path = explicit_path
    if path is None and DEFAULT_REGIONS_JSON.is_file():
        path = DEFAULT_REGIONS_JSON
    if path is not None:
        regions, cal = load_regions_from_json(path)
        print(f"Overlay regions : loaded {len(regions)} from {path}")
        return regions, cal
    print(f"Overlay regions : using built-in defaults ({len(DEFAULT_OVERLAY_REGIONS)} regions)")
    return list(DEFAULT_OVERLAY_REGIONS), CALIBRATION_SIZE


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 feature extraction")
    parser.add_argument(
        "--method",
        choices=("superpoint",),
        default="superpoint",
        help="Feature extractor for SuperGlue/LightGlue matching (default: superpoint).",
    )
    parser.add_argument("--source", type=Path, default=CLEAN_FOLDER,
                        help=f"Folder with images to process (default: {CLEAN_FOLDER}).")
    parser.add_argument("--regions", type=Path, default=None,
                        help="Optional overlay-regions JSON (default: auto-detect).")
    parser.add_argument("--no-mask", action="store_true",
                        help="Disable overlay-exclusion mask during detection.")
    args = parser.parse_args()

    regions, calibration_size = _resolve_regions(args.regions)

    frames = load_frames(INPUT_FOLDER)
    print(f"Loaded {len(frames)} frame references from {INPUT_FOLDER}")
    print(f"Reading images   : {args.source}")
    print(f"Detector method  : {args.method}")
    print(f"Use overlay mask : {not args.no_mask}")

    feature_sets = extract_features_for_frames(
        frames,
        method=args.method,
        use_mask=not args.no_mask,
        regions=regions,
        calibration_size=calibration_size,
        source_dir=args.source,
    )

    counts = [fs.num_keypoints for fs in feature_sets]
    print("\nKeypoint counts per frame:")
    for frame, fs in zip(frames, feature_sets):
        print(f"  [{frame.index:02d}] {frame.name}  ->  {fs.num_keypoints} kp")
    print(f"\nTotal={sum(counts)}  mean={np.mean(counts):.0f}  "
          f"min={min(counts)}  max={max(counts)}")

    n = len(feature_sets)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, frame, fs in zip(axes, frames, feature_sets):
        img = cv2.imread(str(args.source / frame.name))
        vis = draw_keypoints(img, fs, color=(0, 255, 0), rich=False)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"[{frame.index}] {frame.name}  ({fs.num_keypoints} kp)", fontsize=8)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    out_path = OUTPUT_DIR / f"phase3_keypoints_grid_{args.method}.png"
    plt.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close()
    print(f"\nSaved keypoint grid -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
