"""Phase 4 validation: pairwise tentative matching + SIFT vs SuperPoint comparison.

Two usage modes, selected by `--method`:

  --method sift           (default; backward-compatible)
      [1] Session-aware matching WITHOUT grid filter (baseline) — shows clustering.
      [2] Session-aware matching WITH grid filter — shows spatial improvement.
      [3] Per-pair spatial coverage metric.
      [4] Handoff sanity: points_a / points_b shape and dtype.
      [5] Representative pair visualizations (easy / medium / difficult).

  --method superpoint
      Runs the SuperPoint + LightGlue deep pipeline AND the SIFT (grid-filtered)
      baseline on the same pairs, then prints the head-to-head comparison:

          method     | matches | inliers | coverage | mean_error_px

      Plus representative-pair visualizations for both pipelines.

Assertions (both modes):
  * No pair has 0 tentative matches after the selected pipeline.
  * points_a / points_b return (N, 2) float32.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
    select_pairs,
    draw_tentative_matches,
    DEFAULT_RATIO,
    FrameMatchResult,
    estimate_fundamental_for_matches,
    DEFAULT_F_METHOD,
    DEFAULT_F_THRESHOLD,
    DEFAULT_F_CONFIDENCE,
    DEFAULT_MIN_INLIERS,
)


INPUT_FOLDER = REPO_ROOT / "drones_images_input"
CLEAN_FOLDER = REPO_ROOT / "outputs" / "clean_frames"
OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_REGIONS_JSON = REPO_ROOT / "config" / "overlay_regions.json"

# Informational coverage threshold (warning only — not a hard assertion).
MIN_CELL_COVERAGE_WARN = 0.30

# Grid defaults (must match matching defaults for the comparison to be fair).
GRID_ROWS = 4
GRID_COLS = 5
GRID_MAX_PER_CELL = 15
TOTAL_CELLS = GRID_ROWS * GRID_COLS


def _resolve_regions(path):
    if path is None and DEFAULT_REGIONS_JSON.is_file():
        path = DEFAULT_REGIONS_JSON
    if path is not None:
        regions, cal = load_regions_from_json(path)
        print(f"Overlay regions : loaded {len(regions)} from {path}")
        return regions, cal
    print(f"Overlay regions : built-in defaults ({len(DEFAULT_OVERLAY_REGIONS)} regions)")
    return list(DEFAULT_OVERLAY_REGIONS), CALIBRATION_SIZE


def _cell_coverage(result: FrameMatchResult, image_h: int, image_w: int,
                   grid_rows: int, grid_cols: int) -> float:
    """Fraction of grid cells containing >= 1 match point (from frame A)."""
    if result.num_tentative == 0:
        return 0.0
    pts = result.points_a()
    cell_h = image_h / grid_rows
    cell_w = image_w / grid_cols
    occupied = set()
    for x, y in pts:
        r = min(int(y / cell_h), grid_rows - 1)
        c = min(int(x / cell_w), grid_cols - 1)
        occupied.add((r, c))
    return len(occupied) / (grid_rows * grid_cols)


def _y_std_normalized(result: FrameMatchResult, image_h: int) -> float:
    """Normalized std-dev of match Y coordinates."""
    if result.num_tentative == 0:
        return 0.0
    pts = result.points_a()
    return float(np.std(pts[:, 1]) / image_h)


def _mean_epipolar_error_px(
    match_result: FrameMatchResult,
    F: np.ndarray,
    inlier_mask: np.ndarray,
) -> float:
    """Mean distance (in pixels) from inlier point in B to the epipolar line
    of its corresponding point in A.

    Line in image B for pt_a is computed via `computeCorrespondEpilines` with
    `whichImage=1`. For a line (a, b, c) and point (x, y):
        d = |a*x + b*y + c| / sqrt(a^2 + b^2)
    """
    if F is None or not inlier_mask.any():
        return float("nan")
    pts_a = match_result.points_a()
    pts_b = match_result.points_b()
    pa_in = pts_a[inlier_mask].reshape(-1, 1, 2).astype(np.float32)
    pb_in = pts_b[inlier_mask].reshape(-1, 2)
    lines_b = cv2.computeCorrespondEpilines(pa_in, 1, F).reshape(-1, 3)
    a, b, c = lines_b[:, 0], lines_b[:, 1], lines_b[:, 2]
    denom = np.sqrt(a * a + b * b) + 1e-12
    d = np.abs(a * pb_in[:, 0] + b * pb_in[:, 1] + c) / denom
    return float(np.mean(d))


def _extract_once(frames, regions, cal, pipeline: str):
    """Extract features once per pipeline (expensive for SuperPoint)."""
    label = "SIFT+grid" if pipeline == "sift" else "SuperPoint+LightGlue"
    print(f"\n>>> Extracting features for pipeline: {label}")
    feature_method = "sift" if pipeline == "sift" else "superpoint"
    feature_sets = extract_features_for_frames(
        frames, method=feature_method, use_mask=True,
        regions=regions, calibration_size=cal, source_dir=CLEAN_FOLDER,
    )
    return feature_sets, label


def _match_and_ransac(
    feature_sets,
    pairs: Sequence[Tuple[int, int]],
    pipeline: str,
    matcher: str,
    ratio: float,
) -> Tuple[List[FrameMatchResult], List]:
    """Match + RANSAC a given pair list, returning both result lists."""
    match_results = match_frame_pairs(
        feature_sets, pairs,
        method=matcher, ratio=ratio, mutual=False,
        grid_filter=(pipeline == "sift"),
        grid_rows=GRID_ROWS, grid_cols=GRID_COLS, grid_max_per_cell=GRID_MAX_PER_CELL,
        progress=False,
        pipeline=pipeline,
    )
    ransac_results = estimate_fundamental_for_matches(
        match_results,
        method=DEFAULT_F_METHOD,
        threshold=DEFAULT_F_THRESHOLD,
        confidence=DEFAULT_F_CONFIDENCE,
        min_inliers=DEFAULT_MIN_INLIERS,
        progress=False,
    )
    return match_results, ransac_results


def _aggregate_metrics(
    match_results: Sequence[FrameMatchResult],
    ransac_results: Sequence,
    img_h: int, img_w: int,
) -> Dict[str, float]:
    matches = [r.num_tentative for r in match_results]
    inliers = [rr.num_inliers for rr in ransac_results]
    coverage = [_cell_coverage(r, img_h, img_w, GRID_ROWS, GRID_COLS)
                for r in match_results]
    errors = [
        _mean_epipolar_error_px(mr, rr.F, rr.inlier_mask)
        for mr, rr in zip(match_results, ransac_results)
        if rr.f_estimated and rr.inlier_mask is not None and rr.num_inliers > 0
    ]
    return {
        "matches_mean": float(np.mean(matches)) if matches else 0.0,
        "matches_min": int(np.min(matches)) if matches else 0,
        "matches_max": int(np.max(matches)) if matches else 0,
        "inliers_mean": float(np.mean(inliers)) if inliers else 0.0,
        "inliers_min": int(np.min(inliers)) if inliers else 0,
        "inliers_max": int(np.max(inliers)) if inliers else 0,
        "coverage_mean": float(np.mean(coverage)) if coverage else 0.0,
        "coverage_min": float(np.min(coverage)) if coverage else 0.0,
        "coverage_max": float(np.max(coverage)) if coverage else 0.0,
        "mean_err_mean": float(np.mean(errors)) if errors else float("nan"),
        "mean_err_max": float(np.max(errors)) if errors else float("nan"),
        "pairs_with_F": int(sum(1 for rr in ransac_results if rr.f_estimated)),
        "pairs_success": int(sum(1 for rr in ransac_results if rr.success)),
        "n_pairs": len(match_results),
    }


def _print_comparison_table(
    sift_mr: Sequence[FrameMatchResult], sift_rr: Sequence,
    deep_mr: Sequence[FrameMatchResult], deep_rr: Sequence,
    img_h: int, img_w: int,
) -> None:
    """Print the SIFT vs SuperPoint head-to-head comparison."""
    print("\n" + "=" * 78)
    print(" SIFT vs SuperPoint+LightGlue — Per-pair comparison")
    print("=" * 78)
    print(f"    {'pair':>7}  "
          f"|| {'matches':>8} {'inliers':>8} {'cover':>6} {'err_px':>7}  "
          f"|| {'matches':>8} {'inliers':>8} {'cover':>6} {'err_px':>7}")
    print(f"    {'':>7}  "
          f"||      SIFT + grid filter baseline         "
          f"||    SuperPoint + LightGlue")
    print("    " + "-" * 74)

    for smr, srr, dmr, drr in zip(sift_mr, sift_rr, deep_mr, deep_rr):
        sc = _cell_coverage(smr, img_h, img_w, GRID_ROWS, GRID_COLS)
        dc = _cell_coverage(dmr, img_h, img_w, GRID_ROWS, GRID_COLS)
        se = (_mean_epipolar_error_px(smr, srr.F, srr.inlier_mask)
              if srr.f_estimated and srr.inlier_mask is not None
              else float("nan"))
        de = (_mean_epipolar_error_px(dmr, drr.F, drr.inlier_mask)
              if drr.f_estimated and drr.inlier_mask is not None
              else float("nan"))
        print(f"    ({smr.idx_a:2d},{smr.idx_b:2d})  "
              f"|| {smr.num_tentative:8d} {srr.num_inliers:8d} "
              f"{sc:6.2f} {se:7.2f}  "
              f"|| {dmr.num_tentative:8d} {drr.num_inliers:8d} "
              f"{dc:6.2f} {de:7.2f}")

    sift_agg = _aggregate_metrics(sift_mr, sift_rr, img_h, img_w)
    deep_agg = _aggregate_metrics(deep_mr, deep_rr, img_h, img_w)

    print("\n    Aggregate (mean over pairs):")
    print(f"    {'method':>22} | {'matches':>8} | {'inliers':>8} | "
          f"{'coverage':>8} | {'mean_err_px':>11} | {'F_ok':>5} | {'success':>7}")
    print("    " + "-" * 80)
    for name, agg in (("SIFT + grid", sift_agg),
                      ("SuperPoint + LightGlue", deep_agg)):
        err = agg["mean_err_mean"]
        err_s = f"{err:11.3f}" if not np.isnan(err) else f"{'nan':>11}"
        print(f"    {name:>22} | {agg['matches_mean']:8.0f} | "
              f"{agg['inliers_mean']:8.0f} | {agg['coverage_mean']:8.2f} | "
              f"{err_s} | "
              f"{agg['pairs_with_F']:3d}/{agg['n_pairs']:<2d} | "
              f"{agg['pairs_success']:3d}/{agg['n_pairs']:<2d}")

    delta_match = deep_agg["matches_mean"] - sift_agg["matches_mean"]
    delta_inl = deep_agg["inliers_mean"] - sift_agg["inliers_mean"]
    delta_cov = deep_agg["coverage_mean"] - sift_agg["coverage_mean"]
    print(f"\n    Delta (SuperPoint - SIFT): "
          f"matches {delta_match:+.1f}  inliers {delta_inl:+.1f}  "
          f"coverage {delta_cov:+.2f}")


def _save_representative_visualizations(
    match_results: Sequence[FrameMatchResult],
    tag: str,
    representatives: Sequence[Tuple[Tuple[int, int], str]],
) -> None:
    rep_map = {(r.idx_a, r.idx_b): r for r in match_results}
    print(f"\n    Saving {tag} visualizations:")
    for (i, j), label in representatives:
        if (i, j) not in rep_map:
            continue
        r = rep_map[(i, j)]
        img_a = cv2.imread(str(CLEAN_FOLDER / r.name_a))
        img_b = cv2.imread(str(CLEAN_FOLDER / r.name_b))
        vis = draw_tentative_matches(img_a, img_b, r, max_draw=80)
        out = OUTPUT_DIR / f"phase4_pair_{label}_{tag}_{i:02d}_{j:02d}.png"
        cv2.imwrite(str(out), vis)
        print(f"      {label} ({i},{j}): {out.name}")


# -------------------------------------------------------------------------- #
# SIFT-only legacy validation (sections [1]–[5]) — unchanged behavior        #
# -------------------------------------------------------------------------- #
def _run_sift_only_validation(frames, regions, cal, matcher: str, ratio: float) -> int:
    print("Extracting SIFT features from cleaned frames ...")
    feature_sets = extract_features_for_frames(
        frames, method="sift", use_mask=True,
        regions=regions, calibration_size=cal, source_dir=CLEAN_FOLDER,
    )

    img_h, img_w = feature_sets[0].image_shape
    session_pairs = select_pairs(frames, policy="session")

    print(f"\n[1] Baseline matching (NO grid filter) — {len(session_pairs)} session pairs")
    results_base = match_frame_pairs(
        feature_sets, session_pairs,
        method=matcher, ratio=ratio, mutual=False,
        grid_filter=False, progress=False,
    )
    tents_base = [r.num_tentative for r in results_base]
    cov_base = [_cell_coverage(r, img_h, img_w, GRID_ROWS, GRID_COLS) for r in results_base]
    ystd_base = [_y_std_normalized(r, img_h) for r in results_base]
    print(f"    Tentative  : mean={np.mean(tents_base):.0f}  min={min(tents_base)}  max={max(tents_base)}")
    print(f"    Cell cover : mean={np.mean(cov_base):.2f}  min={min(cov_base):.2f}  max={max(cov_base):.2f}")
    print(f"    Y-std norm : mean={np.mean(ystd_base):.3f}  min={min(ystd_base):.3f}  max={max(ystd_base):.3f}")

    print(f"\n[2] Grid-filtered matching (grid={GRID_ROWS}x{GRID_COLS}  max_per_cell={GRID_MAX_PER_CELL}) "
          f"— {len(session_pairs)} session pairs")
    results_grid = match_frame_pairs(
        feature_sets, session_pairs,
        method=matcher, ratio=ratio, mutual=False,
        grid_filter=True,
        grid_rows=GRID_ROWS, grid_cols=GRID_COLS, grid_max_per_cell=GRID_MAX_PER_CELL,
        progress=False,
    )
    tents_grid = [r.num_tentative for r in results_grid]
    cov_grid = [_cell_coverage(r, img_h, img_w, GRID_ROWS, GRID_COLS) for r in results_grid]
    ystd_grid = [_y_std_normalized(r, img_h) for r in results_grid]
    print(f"    Tentative  : mean={np.mean(tents_grid):.0f}  min={min(tents_grid)}  max={max(tents_grid)}")
    print(f"    Cell cover : mean={np.mean(cov_grid):.2f}  min={min(cov_grid):.2f}  max={max(cov_grid):.2f}")
    print(f"    Y-std norm : mean={np.mean(ystd_grid):.3f}  min={min(ystd_grid):.3f}  max={max(ystd_grid):.3f}")

    print(f"\n[3] Per-pair spatial coverage comparison "
          f"({TOTAL_CELLS} cells):")
    hdr = f"    {'pair':>7}  {'tent_base':>9}  {'cov_base':>8}  "
    hdr += f"{'ystd_base':>9}  ||  {'tent_grid':>9}  {'cov_grid':>8}  {'ystd_grid':>9}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))

    below_coverage: List[Tuple[int, int]] = []
    for rb, rg, cb, cg, yb, yg in zip(results_base, results_grid, cov_base, cov_grid, ystd_base, ystd_grid):
        flag = "  <-- LOW" if cg < MIN_CELL_COVERAGE_WARN else ""
        if cg < MIN_CELL_COVERAGE_WARN:
            below_coverage.append((rg.idx_a, rg.idx_b))
        print(f"    ({rb.idx_a:2d},{rb.idx_b:2d})  "
              f"{rb.num_tentative:9d}  {cb:8.2f}  {yb:9.3f}  ||  "
              f"{rg.num_tentative:9d}  {cg:8.2f}  {yg:9.3f}{flag}")

    delta_cov = [cg - cb for cg, cb in zip(cov_grid, cov_base)]
    delta_ystd = [yg - yb for yg, yb in zip(ystd_grid, ystd_base)]
    print(f"\n    Spatial improvement (grid vs base):")
    print(f"      Cell coverage diff: mean={np.mean(delta_cov):+.3f}  min={min(delta_cov):+.3f}  max={max(delta_cov):+.3f}")
    print(f"      Y-std-norm diff   : mean={np.mean(delta_ystd):+.3f}  min={min(delta_ystd):+.3f}  max={max(delta_ystd):+.3f}")

    first = next(r for r in results_grid if r.num_tentative > 0)
    pa = first.points_a()
    pb = first.points_b()
    assert pa.shape == (first.num_tentative, 2), "points_a shape mismatch"
    assert pb.shape == (first.num_tentative, 2), "points_b shape mismatch"
    assert pa.dtype == np.float32 and pb.dtype == np.float32, "points arrays must be float32"
    print(f"\n[4] Handoff sanity: points_a.shape={pa.shape} dtype={pa.dtype} OK.")

    zero_pairs = [r for r in results_grid if r.num_tentative == 0]
    assert len(zero_pairs) == 0, (
        f"{len(zero_pairs)} pair(s) have 0 tentative matches after grid filter."
    )

    representatives: List[Tuple[Tuple[int, int], str]] = [
        ((0, 1), "easy"), ((3, 7), "medium"), ((4, 10), "difficult"),
    ]
    rep_pairs_idx = [p for p, _ in representatives]

    results_rep_base = match_frame_pairs(
        feature_sets, rep_pairs_idx,
        method=matcher, ratio=ratio, mutual=False,
        grid_filter=False,
    )
    results_rep_grid = match_frame_pairs(
        feature_sets, rep_pairs_idx,
        method=matcher, ratio=ratio, mutual=False,
        grid_filter=True, grid_rows=GRID_ROWS, grid_cols=GRID_COLS,
        grid_max_per_cell=GRID_MAX_PER_CELL,
    )
    rep_base_map = {(r.idx_a, r.idx_b): r for r in results_rep_base}
    rep_grid_map = {(r.idx_a, r.idx_b): r for r in results_rep_grid}

    print(f"\n[5] Representative pairs — before vs after grid filter:")
    print(f"    {'label':12}  {'pair':>7}  "
          f"{'base_tent':>9}  {'base_cov':>8}  {'base_ystd':>9}  ||  "
          f"{'grid_tent':>9}  {'grid_cov':>8}  {'grid_ystd':>9}")
    for (i, j), label in representatives:
        rb = rep_base_map[(i, j)]
        rg = rep_grid_map[(i, j)]
        cb = _cell_coverage(rb, img_h, img_w, GRID_ROWS, GRID_COLS)
        cg = _cell_coverage(rg, img_h, img_w, GRID_ROWS, GRID_COLS)
        yb = _y_std_normalized(rb, img_h)
        yg = _y_std_normalized(rg, img_h)
        print(f"    {label:12}  ({i:2d},{j:2d})  "
              f"{rb.num_tentative:9d}  {cb:8.2f}  {yb:9.3f}  ||  "
              f"{rg.num_tentative:9d}  {cg:8.2f}  {yg:9.3f}")

    print(f"\n    Saving visualizations:")
    for (i, j), label in representatives:
        img_a = cv2.imread(str(CLEAN_FOLDER / rep_grid_map[(i, j)].name_a))
        img_b = cv2.imread(str(CLEAN_FOLDER / rep_grid_map[(i, j)].name_b))
        for tag, result in (("base", rep_base_map[(i, j)]),
                            ("grid", rep_grid_map[(i, j)])):
            vis = draw_tentative_matches(img_a, img_b, result, max_draw=80)
            out = OUTPUT_DIR / f"phase4_pair_{label}_{tag}_{i:02d}_{j:02d}.png"
            cv2.imwrite(str(out), vis)
        print(f"      {label} ({i},{j}): "
              f"phase4_pair_{label}_base_{i:02d}_{j:02d}.png  "
              f"phase4_pair_{label}_grid_{i:02d}_{j:02d}.png")
    return 0


# -------------------------------------------------------------------------- #
# Main                                                                       #
# -------------------------------------------------------------------------- #
def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4 validation")
    parser.add_argument("--method", default="sift", choices=("sift", "superpoint"),
                        help="Pipeline to validate. 'sift' keeps the original "
                             "grid-filter comparison; 'superpoint' runs the deep "
                             "pipeline AND compares it head-to-head with SIFT. "
                             "Default: sift.")
    parser.add_argument("--matcher", default="flann", choices=("flann", "bf"),
                        help="Matcher algorithm for the SIFT path (default: flann).")
    parser.add_argument("--ratio", type=float, default=DEFAULT_RATIO)
    parser.add_argument("--regions", type=Path, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print(f" PHASE 4 VALIDATION  [method={args.method}]")
    print("=" * 70)

    regions, cal = _resolve_regions(args.regions)

    frames = load_frames(INPUT_FOLDER)
    assert len(frames) > 0, "No input frames found."
    assert CLEAN_FOLDER.is_dir(), f"Run Phase 2 first: {CLEAN_FOLDER} missing."
    print(f"Loaded {len(frames)} frames.")

    if args.method == "sift":
        ret = _run_sift_only_validation(frames, regions, cal, args.matcher, args.ratio)
        print("\n" + "=" * 70)
        print(" PHASE 4 VALIDATION: PASSED")
        print("=" * 70)
        return ret

    # --- method == "superpoint" --------------------------------------------- #
    session_pairs = select_pairs(frames, policy="session")
    print(f"Session pairs   : {len(session_pairs)}")

    # Extract features once per pipeline, then reuse for both the session
    # comparison and the representative-pair visualizations.
    sift_fs, _ = _extract_once(frames, regions, cal, pipeline="sift")
    deep_fs, _ = _extract_once(frames, regions, cal, pipeline="superpoint")
    img_h, img_w = sift_fs[0].image_shape

    sift_mr, sift_rr = _match_and_ransac(sift_fs, session_pairs, "sift", args.matcher, args.ratio)
    deep_mr, deep_rr = _match_and_ransac(deep_fs, session_pairs, "superpoint", args.matcher, args.ratio)

    # Handoff sanity on the deep path (still points_a/points_b contract)
    first = next(r for r in deep_mr if r.num_tentative > 0)
    pa = first.points_a()
    pb = first.points_b()
    assert pa.shape == (first.num_tentative, 2), "points_a shape mismatch"
    assert pb.shape == (first.num_tentative, 2), "points_b shape mismatch"
    assert pa.dtype == np.float32 and pb.dtype == np.float32, "points arrays must be float32"
    print(f"\n[OK] Deep-pipeline handoff sanity: points_a.shape={pa.shape} dtype={pa.dtype}")

    zero_pairs = [r for r in deep_mr if r.num_tentative == 0]
    assert len(zero_pairs) == 0, (
        f"{len(zero_pairs)} pair(s) have 0 LightGlue matches."
    )

    _print_comparison_table(sift_mr, sift_rr, deep_mr, deep_rr, img_h, img_w)

    representatives: List[Tuple[Tuple[int, int], str]] = [
        ((0, 1), "easy"), ((3, 7), "medium"), ((4, 10), "difficult"),
    ]

    # The representative set may contain cross-session pairs (e.g. (4,10))
    # that are not in the session-only comparison above. Re-match them
    # explicitly (reusing the already-extracted features, no re-extraction)
    # so all three visualizations are produced regardless of session membership.
    rep_idx = [p for p, _ in representatives]
    sift_rep_mr, _ = _match_and_ransac(sift_fs, rep_idx, "sift", args.matcher, args.ratio)
    deep_rep_mr, _ = _match_and_ransac(deep_fs, rep_idx, "superpoint", args.matcher, args.ratio)
    _save_representative_visualizations(sift_rep_mr, "sift", representatives)
    _save_representative_visualizations(deep_rep_mr, "superpoint", representatives)

    print("\n" + "=" * 70)
    print(" PHASE 4 VALIDATION: PASSED  (SIFT vs SuperPoint comparison done)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
