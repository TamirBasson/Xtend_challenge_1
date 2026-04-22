"""Phase 5 entry script: RANSAC-based fundamental-matrix estimation.

Runs Phase 4 (matching) + Phase 5 (RANSAC) over session-internal pairs
and saves per-pair statistics.

Usage (from repository root):
    python scripts/run_phase5_ransac.py
    python scripts/run_phase5_ransac.py --method usac_magsac --threshold 0.5
    python scripts/run_phase5_ransac.py --pairs "0,1 3,7 4,10"

Outputs:
    outputs/phase5_ransac_stats.csv
    outputs/phase5_inliers_<i>_<j>.png          (one per successful pair, top N)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

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
    select_pairs,
    parse_pairs_arg,
    match_frame_pairs,
    SUPPORTED_F_METHODS,
    DEFAULT_F_METHOD,
    DEFAULT_F_THRESHOLD,
    DEFAULT_F_CONFIDENCE,
    DEFAULT_MIN_INLIERS,
    estimate_fundamental_for_matches,
    draw_inlier_matches,
)


INPUT_FOLDER = REPO_ROOT / "drones_images_input"
CLEAN_FOLDER = REPO_ROOT / "outputs" / "clean_frames"
OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_REGIONS_JSON = REPO_ROOT / "config" / "overlay_regions.json"


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
    parser = argparse.ArgumentParser(description="Phase 5 RANSAC / fundamental matrix.")
    # Pair selection (same vocabulary as Phase 4)
    parser.add_argument("--policy", choices=("session", "all", "window"), default="session")
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--pairs", default=None,
                        help='Explicit pairs, e.g. "0,1 3,7 4,10" (overrides --policy).')
    parser.add_argument("--regions", type=Path, default=None)

    # Deprecated SIFT matching flags kept only for CLI compatibility.
    parser.add_argument("--matcher", default="flann", help=argparse.SUPPRESS)
    parser.add_argument("--ratio", type=float, default=0.75, help=argparse.SUPPRESS)
    parser.add_argument("--mutual", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--grid", dest="grid_filter", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--no-grid", dest="grid_filter", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--grid-rows", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--grid-cols", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--grid-max", type=int, default=15, help=argparse.SUPPRESS)

    # RANSAC options
    parser.add_argument("--method", choices=sorted(SUPPORTED_F_METHODS),
                        default=DEFAULT_F_METHOD,
                        help="RANSAC variant for findFundamentalMat.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="RANSAC reprojection error in pixels (default 0.5).")
    parser.add_argument("--confidence", type=float, default=DEFAULT_F_CONFIDENCE)
    parser.add_argument("--min-inliers", type=int, default=DEFAULT_MIN_INLIERS)
    parser.add_argument("--draw-top-n", type=int, default=5)

    args = parser.parse_args()

    regions, cal = _resolve_regions(args.regions)

    frames = load_frames(INPUT_FOLDER)
    if not CLEAN_FOLDER.is_dir():
        print(f"ERROR: {CLEAN_FOLDER} not found. Run Phase 2 first.")
        return 1

    print(f"Frames           : {len(frames)}")
    print("Extracting SuperPoint features from cleaned frames ...")
    feature_sets = extract_features_for_frames(
        frames, method="superpoint", use_mask=True,
        regions=regions, calibration_size=cal, source_dir=CLEAN_FOLDER,
    )

    explicit = parse_pairs_arg(args.pairs)
    if explicit is not None:
        pairs = explicit
        print(f"Pair policy      : explicit ({len(pairs)} pairs)")
    else:
        pairs = select_pairs(frames, policy=args.policy, window_size=args.window)
        print(f"Pair policy      : {args.policy} ({len(pairs)} pairs)")

    print("Matching         : lightglue (SuperPoint pipeline; legacy matcher flags ignored)")
    print(f"RANSAC           : method={args.method}  threshold={args.threshold}px  "
          f"confidence={args.confidence}  min_inliers={args.min_inliers}")
    print()

    print("Running matching ...")
    match_results = match_frame_pairs(
        feature_sets, pairs,
        pipeline="superpoint",
    )

    print("Running RANSAC ...")
    ransac_results = estimate_fundamental_for_matches(
        match_results,
        method=args.method, threshold=args.threshold,
        confidence=args.confidence, min_inliers=args.min_inliers,
        progress=True,
    )

    # Persist stats
    csv_path = OUTPUT_DIR / "phase5_ransac_stats.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "idx_a", "idx_b", "name_a", "name_b",
            "tentative", "inliers", "inlier_ratio_percent",
            "f_estimated", "success",
        ])
        for r in ransac_results:
            w.writerow([
                r.idx_a, r.idx_b, r.name_a, r.name_b,
                r.num_tentative, r.num_inliers,
                f"{100 * r.inlier_ratio:.2f}",
                r.f_estimated, r.success,
            ])
    print(f"\nSaved stats CSV  : {csv_path}")

    # Summary
    n_ok = sum(1 for r in ransac_results if r.success)
    n_f  = sum(1 for r in ransac_results if r.f_estimated)
    n    = len(ransac_results)
    ratios = [r.inlier_ratio for r in ransac_results if r.f_estimated]
    inls   = [r.num_inliers for r in ransac_results if r.f_estimated]
    print(f"\nSummary          :")
    print(f"  success (>= {args.min_inliers} inl): {n_ok}/{n}  ({100 * n_ok / n:.1f}%)")
    print(f"  F estimated           : {n_f}/{n}  ({100 * n_f / n:.1f}%)")
    if inls:
        print(f"  inliers   : min={min(inls)}  mean={np.mean(inls):.0f}  max={max(inls)}  median={int(np.median(inls))}")
        print(f"  ratio     : min={100*min(ratios):.1f}%  mean={100*np.mean(ratios):.1f}%  max={100*max(ratios):.1f}%")

    # Visualize top-N successful pairs by inlier count
    successful = [r for r in ransac_results if r.success]
    top = sorted(successful, key=lambda r: r.num_inliers, reverse=True)[:args.draw_top_n]
    mr_by_key = {(r.idx_a, r.idx_b): r for r in match_results}

    print(f"\nSaving top-{len(top)} inlier visualizations:")
    for rr in top:
        mr = mr_by_key[(rr.idx_a, rr.idx_b)]
        img_a = cv2.imread(str(CLEAN_FOLDER / rr.name_a))
        img_b = cv2.imread(str(CLEAN_FOLDER / rr.name_b))
        vis = draw_inlier_matches(img_a, img_b, mr, rr, max_draw=80)
        out = OUTPUT_DIR / f"phase5_inliers_{rr.idx_a:02d}_{rr.idx_b:02d}.png"
        cv2.imwrite(str(out), vis)
        print(f"  ({rr.idx_a},{rr.idx_b})  "
              f"tent={rr.num_tentative}  inl={rr.num_inliers}  -> {out.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
