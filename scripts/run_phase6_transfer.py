"""Phase 6 entry script: epipolar-guided point transfer (Milestone 3 baseline).

For a small set of pairs (default: the 3 representative pairs used in Phase 5)
this runs:
    1. Phase 3 feature extraction on the cleaned frames (SuperPoint).
    2. Phase 4 matching (LightGlue).
    3. Phase 5 RANSAC (fundamental matrix).
    4. Phase 6 transfer: for each pair, pick K source pixels from the RANSAC
       inlier set and transfer them to the target using the F matrix.

For every query point we save a side-by-side visualization that shows:
    - source pixel (red cross) on the source frame
    - epipolar line (yellow) on the target frame
    - predicted pixel (green cross)
    - ground-truth inlier correspondence (magenta cross, for reference only)

Usage (from repository root):
    python scripts/run_phase6_transfer.py
    python scripts/run_phase6_transfer.py --pairs "0,1 3,7"
    python scripts/run_phase6_transfer.py --num-points 3 --patch-size 31
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from src import (  # noqa: E402
    load_frames,
    extract_features_for_frames,
    DEFAULT_OVERLAY_REGIONS,
    CALIBRATION_SIZE,
    load_regions_from_json,
    match_frame_pairs,
    parse_pairs_arg,
    DEFAULT_F_METHOD,
    DEFAULT_F_THRESHOLD,
    DEFAULT_F_CONFIDENCE,
    DEFAULT_MIN_INLIERS,
    estimate_fundamental_for_matches,
    DEFAULT_PATCH_SIZE,
    DEFAULT_STEP,
    transfer_point,
    draw_transfer,
)


INPUT_FOLDER = REPO_ROOT / "drones_images_input"
CLEAN_FOLDER = REPO_ROOT / "outputs" / "clean_frames"
OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_REGIONS_JSON = REPO_ROOT / "config" / "overlay_regions.json"

DEFAULT_PAIRS: List[Tuple[int, int]] = [(0, 1), (3, 7), (4, 10)]
LABELS = {(0, 1): "easy", (3, 7): "medium", (4, 10): "difficult"}

GRID_ROWS = 4
GRID_COLS = 5
GRID_MAX = 15


def _resolve_regions(path):
    if path is None and DEFAULT_REGIONS_JSON.is_file():
        path = DEFAULT_REGIONS_JSON
    if path is not None:
        regions, cal = load_regions_from_json(path)
        print(f"Overlay regions : loaded {len(regions)} from {path}")
        return regions, cal
    print(f"Overlay regions : built-in defaults ({len(DEFAULT_OVERLAY_REGIONS)} regions)")
    return list(DEFAULT_OVERLAY_REGIONS), CALIBRATION_SIZE


def _select_query_points(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    image_shape: Tuple[int, int],
    num_points: int,
    patch_margin: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Pick up to `num_points` inlier pairs spread across the source image.

    Uses a simple coarse grid (2x3) and picks the first valid inlier in each
    cell; falls back to the first inliers if the grid yields too few.
    """
    h, w = image_shape
    if len(pts_a) == 0:
        return []

    rows, cols = 2, 3
    cell_h = h / rows
    cell_w = w / cols
    picked: List[Tuple[int, Tuple[int, int]]] = []
    used = set()

    for r in range(rows):
        for c in range(cols):
            for i, (x, y) in enumerate(pts_a):
                if i in used:
                    continue
                if (patch_margin <= x <= w - 1 - patch_margin
                        and patch_margin <= y <= h - 1 - patch_margin
                        and int(y / cell_h) == r
                        and int(x / cell_w) == c):
                    picked.append((i, (r, c)))
                    used.add(i)
                    break

    picked = picked[:num_points]

    # Fallback: pad with the remaining best inliers in insertion order.
    if len(picked) < num_points:
        for i in range(len(pts_a)):
            if i in used:
                continue
            x, y = pts_a[i]
            if (patch_margin <= x <= w - 1 - patch_margin
                    and patch_margin <= y <= h - 1 - patch_margin):
                picked.append((i, (-1, -1)))
                used.add(i)
                if len(picked) >= num_points:
                    break

    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for i, _ in picked:
        out.append((pts_a[i], pts_b[i]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6: epipolar-guided point transfer.")
    parser.add_argument("--pairs", default=None,
                        help='Pairs to run, e.g. "0,1 3,7 4,10" (default: 3 representatives).')
    parser.add_argument("--num-points", type=int, default=3,
                        help="Query points per pair (default: 3).")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE,
                        help=f"NCC patch size (odd int, default: {DEFAULT_PATCH_SIZE}).")
    parser.add_argument("--step", type=float, default=DEFAULT_STEP,
                        help=f"Sampling step along the epipolar line in px (default: {DEFAULT_STEP}).")
    parser.add_argument("--regions", type=Path, default=None)
    parser.add_argument("--method", default=DEFAULT_F_METHOD)
    parser.add_argument("--threshold", type=float, default=DEFAULT_F_THRESHOLD)
    parser.add_argument("--confidence", type=float, default=DEFAULT_F_CONFIDENCE)
    parser.add_argument("--min-inliers", type=int, default=DEFAULT_MIN_INLIERS)
    parser.add_argument("--draw-samples", action="store_true",
                        help="Overlay the sampled candidate points on the target image.")
    args = parser.parse_args()

    explicit = parse_pairs_arg(args.pairs)
    pairs = explicit if explicit is not None else list(DEFAULT_PAIRS)

    regions, cal = _resolve_regions(args.regions)

    frames = load_frames(INPUT_FOLDER)
    if not CLEAN_FOLDER.is_dir():
        print(f"ERROR: {CLEAN_FOLDER} not found. Run Phase 2 first.")
        return 1
    print(f"Frames           : {len(frames)}")
    print(f"Pairs            : {pairs}")
    print(f"Points per pair  : {args.num_points}")
    print(f"Patch size       : {args.patch_size}  | step: {args.step} px")
    print()

    print("Extracting SuperPoint features from cleaned frames ...")
    feature_sets = extract_features_for_frames(
        frames, method="superpoint", use_mask=True,
        regions=regions, calibration_size=cal, source_dir=CLEAN_FOLDER,
    )

    print("Running matching + RANSAC on selected pairs ...")
    match_results = match_frame_pairs(
        feature_sets, pairs,
        pipeline="superpoint",
    )
    ransac_results = estimate_fundamental_for_matches(
        match_results,
        method=args.method, threshold=args.threshold,
        confidence=args.confidence, min_inliers=args.min_inliers,
    )

    mr_map = {(m.idx_a, m.idx_b): m for m in match_results}
    rr_map = {(r.idx_a, r.idx_b): r for r in ransac_results}

    csv_path = OUTPUT_DIR / "phase6_transfer_stats.csv"
    csv_rows: List[List] = []

    print("\nTransfer queries:")
    for (i, j) in pairs:
        label = LABELS.get((i, j), f"pair_{i:02d}_{j:02d}")
        mr = mr_map.get((i, j))
        rr = rr_map.get((i, j))
        if mr is None or rr is None:
            print(f"  {label} ({i:2d},{j:2d}): no match/ransac result, skipping.")
            continue
        if not rr.success or rr.F is None:
            print(f"  {label} ({i:2d},{j:2d}): RANSAC failed "
                  f"(inl={rr.num_inliers}/{rr.num_tentative}), skipping.")
            continue

        img_a = cv2.imread(str(CLEAN_FOLDER / rr.name_a))
        img_b = cv2.imread(str(CLEAN_FOLDER / rr.name_b))
        if img_a is None or img_b is None:
            print(f"  {label} ({i:2d},{j:2d}): could not load images, skipping.")
            continue

        pts_a_all = rr.inlier_points_a(mr.fs_a_filtered)
        pts_b_all = rr.inlier_points_b(mr.fs_b_filtered)
        queries = _select_query_points(
            pts_a_all, pts_b_all,
            image_shape=img_a.shape[:2],
            num_points=args.num_points,
            patch_margin=args.patch_size // 2,
        )

        print(f"  {label:10} ({i:2d},{j:2d}): inliers={rr.num_inliers}  "
              f"queries={len(queries)}")

        for q_idx, (src_px, gt_px) in enumerate(queries):
            result = transfer_point(
                source_pixel=tuple(src_px.tolist()),
                image_src=img_a, image_dst=img_b,
                F=rr.F, source_is_a=True,
                patch_size=args.patch_size, step=args.step,
            )
            err = float("nan")
            if result.predicted_pixel is not None:
                err = float(np.hypot(result.predicted_pixel[0] - gt_px[0],
                                     result.predicted_pixel[1] - gt_px[1]))

            vis = draw_transfer(
                img_a, img_b, result,
                ground_truth=tuple(gt_px.tolist()),
                draw_samples=args.draw_samples,
            )
            out = OUTPUT_DIR / f"phase6_{label}_q{q_idx}_{i:02d}_{j:02d}.png"
            cv2.imwrite(str(out), vis)

            print(f"      q{q_idx}: src=({src_px[0]:6.1f},{src_px[1]:6.1f})  "
                  f"pred="
                  f"{'(%6.1f,%6.1f)' % result.predicted_pixel if result.predicted_pixel else '     none    '}"
                  f"  gt=({gt_px[0]:6.1f},{gt_px[1]:6.1f})  "
                  f"score={result.score:.3f}  err={err:.1f}px  "
                  f"-> {out.name}")

            csv_rows.append([
                i, j, label, q_idx,
                float(src_px[0]), float(src_px[1]),
                float(gt_px[0]), float(gt_px[1]),
                (result.predicted_pixel[0] if result.predicted_pixel else ""),
                (result.predicted_pixel[1] if result.predicted_pixel else ""),
                result.score, err,
                result.num_samples, result.num_scored,
                args.patch_size, args.step,
                result.success, result.note,
            ])

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "idx_a", "idx_b", "label", "q",
            "src_u", "src_v", "gt_u", "gt_v",
            "pred_u", "pred_v",
            "ncc_score", "error_px",
            "num_samples", "num_scored",
            "patch_size", "step", "success", "note",
        ])
        w.writerows(csv_rows)
    print(f"\nSaved stats CSV  : {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
