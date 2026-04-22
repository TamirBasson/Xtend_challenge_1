"""Phase 3 validation: verify feature extraction on cleaned frames.

Checks performed:
  1. Cleaned frames exist and are readable.
  2. Each frame yields > MIN_KEYPOINTS keypoints.
  3. Descriptor shape matches detector expectations.
  4. No keypoints fall inside overlay regions (mask is respected).
  5. Saves a per-frame zoom visualization of the first frame's keypoints.
"""

from __future__ import annotations

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
    build_overlay_mask,
    DEFAULT_OVERLAY_REGIONS,
    CALIBRATION_SIZE,
    load_regions_from_json,
)


INPUT_FOLDER = REPO_ROOT / "drones_images_input"
CLEAN_FOLDER = REPO_ROOT / "outputs" / "clean_frames"
OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_REGIONS_JSON = REPO_ROOT / "config" / "overlay_regions.json"

MIN_KEYPOINTS = 200
EXPECTED_SIFT_DESC_DIM = 128


def _resolve_regions(path):
    if path is None and DEFAULT_REGIONS_JSON.is_file():
        path = DEFAULT_REGIONS_JSON
    if path is not None:
        regions, cal = load_regions_from_json(path)
        print(f"Overlay regions : loaded {len(regions)} from {path}")
        return regions, cal
    print(f"Overlay regions : built-in defaults ({len(DEFAULT_OVERLAY_REGIONS)} regions)")
    return list(DEFAULT_OVERLAY_REGIONS), CALIBRATION_SIZE


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Phase 3 validation")
    parser.add_argument("--regions", type=Path, default=None)
    parser.add_argument("--method", default="sift")
    args = parser.parse_args()

    print("=" * 60)
    print(" PHASE 3 VALIDATION - Feature Extraction")
    print("=" * 60)

    regions, calibration_size = _resolve_regions(args.regions)

    if not CLEAN_FOLDER.is_dir():
        print(f"[FAIL] Cleaned frames folder not found: {CLEAN_FOLDER}")
        print("       Run `python scripts/run_phase2_clean.py` first.")
        return 1

    frames = load_frames(INPUT_FOLDER)
    assert len(frames) > 0, "No input frames found."

    for frame in frames:
        clean_path = CLEAN_FOLDER / frame.name
        assert clean_path.is_file(), f"Missing cleaned frame: {clean_path}"
    print(f"[1] All {len(frames)} cleaned frames present.")

    feature_sets = extract_features_for_frames(
        frames, method=args.method, use_mask=True,
        regions=regions, calibration_size=calibration_size,
        source_dir=CLEAN_FOLDER,
    )

    counts = [fs.num_keypoints for fs in feature_sets]
    print("\n[2] Keypoint counts:")
    for frame, fs in zip(frames, feature_sets):
        flag = "ok" if fs.num_keypoints >= MIN_KEYPOINTS else "LOW"
        print(f"    [{frame.index:02d}] {frame.name}  {fs.num_keypoints:5d} kp  [{flag}]")
    print(f"    total={sum(counts)}  mean={np.mean(counts):.0f}  "
          f"min={min(counts)}  max={max(counts)}")

    low_frames = [fs for fs in feature_sets if fs.num_keypoints < MIN_KEYPOINTS]
    assert not low_frames, (
        f"{len(low_frames)} frame(s) produced fewer than {MIN_KEYPOINTS} keypoints: "
        f"{[fs.frame_name for fs in low_frames]}"
    )

    first = feature_sets[0]
    if args.method == "sift":
        assert first.descriptors is not None, "SIFT returned no descriptors"
        assert first.descriptors.shape[1] == EXPECTED_SIFT_DESC_DIM, (
            f"SIFT descriptor dim {first.descriptors.shape[1]} != {EXPECTED_SIFT_DESC_DIM}"
        )
        assert first.descriptors.shape[0] == first.num_keypoints, (
            f"descriptors rows {first.descriptors.shape[0]} != keypoints {first.num_keypoints}"
        )
        print(f"\n[3] SIFT descriptor shape OK: {first.descriptors.shape} (float32 expected).")
    else:
        print(f"\n[3] Descriptor shape: {first.descriptors.shape}")

    print("\n[4] Checking that no keypoints fall inside overlay regions...")
    total_violations = 0
    first_clean = cv2.imread(str(CLEAN_FOLDER / frames[0].name))
    overlay = build_overlay_mask(first_clean, regions=regions,
                                 calibration_size=calibration_size)
    for fs in feature_sets:
        violations = 0
        for kp in fs.keypoints:
            x = int(round(kp.pt[0])); y = int(round(kp.pt[1]))
            if 0 <= y < overlay.shape[0] and 0 <= x < overlay.shape[1]:
                if overlay[y, x] == 255:
                    violations += 1
        if violations:
            print(f"    {fs.frame_name}: {violations} keypoint(s) inside overlay mask")
        total_violations += violations
    assert total_violations == 0, (
        f"{total_violations} keypoints fell inside overlay regions — mask not respected."
    )
    print("    No keypoints in overlay regions across all frames.")

    first_image = cv2.imread(str(CLEAN_FOLDER / frames[0].name))
    vis = draw_keypoints(first_image, feature_sets[0], rich=True)
    out_path = OUTPUT_DIR / f"phase3_keypoints_sample_{args.method}.png"
    cv2.imwrite(str(out_path), vis)
    print(f"\n[5] Saved sample keypoint visualization -> {out_path}")

    n = len(feature_sets)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, frame, fs in zip(axes, frames, feature_sets):
        img = cv2.imread(str(CLEAN_FOLDER / frame.name))
        v = draw_keypoints(img, fs, rich=False)
        ax.imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
        ax.set_title(f"[{frame.index}] {fs.num_keypoints} kp", fontsize=8)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    grid_path = OUTPUT_DIR / f"phase3_keypoints_grid_{args.method}.png"
    plt.savefig(grid_path, dpi=90, bbox_inches="tight")
    plt.close()
    print(f"    Saved grid visualization     -> {grid_path}")

    print("\n" + "=" * 60)
    print(" PHASE 3 VALIDATION: PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
