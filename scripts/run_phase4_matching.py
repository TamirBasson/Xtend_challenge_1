"""Phase 4 entry script: pairwise tentative matching.

Usage (from repository root):
    # Deep pipeline (default): SuperPoint + LightGlue
    python scripts/run_phase4_matching.py
    python scripts/run_phase4_matching.py --policy all
    python scripts/run_phase4_matching.py --policy window --window 3
    python scripts/run_phase4_matching.py --pairs "0,1 3,7 4,10"
    python scripts/run_phase4_matching.py --method superpoint

Note:
    This script is SuperPoint + LightGlue only (`--method superpoint`).

Outputs:
    outputs/phase4_match_stats.csv
    outputs/phase4_matches_<i>_<j>.png          (one per pair, top N)
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
    draw_tentative_matches,
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
    parser = argparse.ArgumentParser(description="Phase 4 pairwise matching (tentative).")
    parser.add_argument("--policy", choices=("session", "all", "window"), default="session",
                        help="Pair-selection policy (default: session).")
    parser.add_argument("--window", type=int, default=3,
                        help="Temporal window size when --policy window (default: 3).")
    parser.add_argument("--pairs", default=None,
                        help='Explicit pairs, e.g. "0,1 3,7 4,10" (overrides --policy).')
    parser.add_argument("--method", choices=("superpoint",), default="superpoint",
                        help="Pipeline selector (SuperPoint + LightGlue only).")
    parser.add_argument("--regions", type=Path, default=None)
    parser.add_argument("--feature-method", default=None,
                        help="Override detector name. Defaults to --method (superpoint).")
    parser.add_argument("--draw-top-n", type=int, default=5,
                        help="How many of the best-scoring pairs to save visualizations for.")
    # Kept for backward CLI compatibility; ignored by SuperPoint+LightGlue.
    parser.add_argument("--matcher", default="flann", help=argparse.SUPPRESS)
    parser.add_argument("--ratio", type=float, default=0.75, help=argparse.SUPPRESS)
    parser.add_argument("--mutual", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--grid", dest="grid_filter", action="store_true", default=True,
                        help=argparse.SUPPRESS)
    parser.add_argument("--no-grid", dest="grid_filter", action="store_false",
                        help=argparse.SUPPRESS)
    parser.add_argument("--grid-rows", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--grid-cols", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--grid-max", type=int, default=15, help=argparse.SUPPRESS)
    args = parser.parse_args()

    regions, cal = _resolve_regions(args.regions)

    frames = load_frames(INPUT_FOLDER)
    if not CLEAN_FOLDER.is_dir():
        print(f"ERROR: {CLEAN_FOLDER} not found. Run Phase 2 first.")
        return 1

    pipeline = args.method.lower()
    feature_method = args.feature_method or pipeline  # mirror pipeline by default

    print(f"Frames           : {len(frames)}")
    print(f"Pipeline         : {pipeline}")
    print(f"Feature method   : {feature_method}")
    print("Extracting features from cleaned frames ...")
    feature_sets = extract_features_for_frames(
        frames, method=feature_method, use_mask=True,
        regions=regions, calibration_size=cal,
        source_dir=CLEAN_FOLDER,
    )

    explicit = parse_pairs_arg(args.pairs)
    if explicit is not None:
        pairs = explicit
        print(f"Pair policy      : explicit ({len(pairs)} pairs)")
    else:
        pairs = select_pairs(frames, policy=args.policy, window_size=args.window)
        print(f"Pair policy      : {args.policy} ({len(pairs)} pairs)")

    print("Matcher          : lightglue (grid/ratio/mutual flags ignored)")
    print("Running matching:")

    results = match_frame_pairs(
        feature_sets, pairs,
        method=args.matcher, ratio=args.ratio, mutual=args.mutual,
        grid_filter=args.grid_filter,
        grid_rows=args.grid_rows, grid_cols=args.grid_cols,
        grid_max_per_cell=args.grid_max,
        progress=True,
        pipeline=pipeline,
    )

    suffix = f"_{pipeline}"
    csv_path = OUTPUT_DIR / f"phase4_match_stats{suffix}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "idx_a", "idx_b", "name_a", "name_b",
            "desc_a", "desc_b",
            "raw_matches", "tentative_matches", "ratio_kept_percent",
            "grid_filtered",
        ])
        for r in results:
            w.writerow([
                r.idx_a, r.idx_b, r.name_a, r.name_b,
                r.num_desc_a, r.num_desc_b,
                r.num_raw_matches, r.num_tentative,
                f"{100 * r.ratio_kept:.2f}",
                r.grid_filtered,
            ])
    print(f"\nSaved stats CSV  : {csv_path}")

    tents = [r.num_tentative for r in results]
    if tents:
        print(f"Tentative matches: min={min(tents)}  "
              f"mean={np.mean(tents):.0f}  "
              f"max={max(tents)}  "
              f"median={int(np.median(tents))}")

    top = sorted(results, key=lambda r: r.num_tentative, reverse=True)[:args.draw_top_n]
    print(f"\nSaving top-{len(top)} pair visualizations:")
    for r in top:
        img_a = cv2.imread(str(CLEAN_FOLDER / r.name_a))
        img_b = cv2.imread(str(CLEAN_FOLDER / r.name_b))
        vis = draw_tentative_matches(img_a, img_b, r, max_draw=80)
        out = OUTPUT_DIR / f"phase4_matches{suffix}_{r.idx_a:02d}_{r.idx_b:02d}.png"
        cv2.imwrite(str(out), vis)
        print(f"  ({r.idx_a},{r.idx_b})  "
              f"raw={r.num_raw_matches}  tentative={r.num_tentative}  -> {out.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
