"""Phase 5 validation: RANSAC / fundamental-matrix evaluation.

Evaluates both configurations:
  [A] baseline matching (no grid filter) + RANSAC
  [B] grid-filtered matching            + RANSAC

For each config we report per-pair RANSAC stats (tentative, inliers,
inlier ratio, F-estimated, success) and spatial-distribution metrics on
the inlier set (cell coverage, Y-std-normalized).

For the 3 representative pairs (easy / medium / difficult) we save:
  - before RANSAC (all tentative matches)
  - after RANSAC  (inliers only)
  - epipolar lines for a handful of sampled inliers

Finally we flag near-degenerate pairs and write a verdict on which
configuration is better suited for the next milestone (epipolar-guided
point transfer).
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
    DEFAULT_RATIO,
    select_pairs,
    match_frame_pairs,
    draw_tentative_matches,
    DEFAULT_F_METHOD,
    DEFAULT_F_THRESHOLD,
    DEFAULT_F_CONFIDENCE,
    DEFAULT_MIN_INLIERS,
    estimate_fundamental_for_matches,
    draw_inlier_matches,
    draw_epipolar_lines,
    is_near_degenerate,
)


INPUT_FOLDER = REPO_ROOT / "drones_images_input"
CLEAN_FOLDER = REPO_ROOT / "outputs" / "clean_frames"
OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_REGIONS_JSON = REPO_ROOT / "config" / "overlay_regions.json"

GRID_ROWS = 4
GRID_COLS = 5
GRID_MAX = 15
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


def _cell_coverage(pts: np.ndarray, image_h: int, image_w: int,
                   rows: int = GRID_ROWS, cols: int = GRID_COLS) -> float:
    if pts is None or len(pts) == 0:
        return 0.0
    cell_h = image_h / rows
    cell_w = image_w / cols
    occupied = set()
    for x, y in pts:
        r = min(int(y / cell_h), rows - 1)
        c = min(int(x / cell_w), cols - 1)
        occupied.add((r, c))
    return len(occupied) / (rows * cols)


def _y_std_norm(pts: np.ndarray, image_h: int) -> float:
    if pts is None or len(pts) == 0:
        return 0.0
    return float(np.std(pts[:, 1]) / image_h)


def _x_std_norm(pts: np.ndarray, image_w: int) -> float:
    if pts is None or len(pts) == 0:
        return 0.0
    return float(np.std(pts[:, 0]) / image_w)


def _run_config(
    label: str,
    feature_sets,
    pairs: Sequence[Tuple[int, int]],
    grid_filter: bool,
    img_h: int, img_w: int,
    method: str, threshold: float, confidence: float, min_inliers: int,
):
    print(f"\n[{label}] matching ({'grid-filtered' if grid_filter else 'no grid filter'}) "
          f"+ RANSAC over {len(pairs)} pairs ...")
    mrs = match_frame_pairs(
        feature_sets, pairs,
        method="flann", ratio=DEFAULT_RATIO, mutual=False,
        grid_filter=grid_filter,
        grid_rows=GRID_ROWS, grid_cols=GRID_COLS, grid_max_per_cell=GRID_MAX,
    )
    rrs = estimate_fundamental_for_matches(
        mrs, method=method, threshold=threshold,
        confidence=confidence, min_inliers=min_inliers,
    )

    # Per-pair metrics on inlier set
    metrics = []
    for mr, rr in zip(mrs, rrs):
        pts_a = (rr.inlier_points_a(mr.fs_a_filtered)
                 if rr.f_estimated and rr.num_inliers > 0
                 else np.zeros((0, 2), dtype=np.float32))
        cov = _cell_coverage(pts_a, img_h, img_w)
        ystd = _y_std_norm(pts_a, img_h)
        xstd = _x_std_norm(pts_a, img_w)
        degen, reason = is_near_degenerate(rr, pts_a, (img_h, img_w),
                                           min_inliers=min_inliers)
        metrics.append({
            "mr": mr, "rr": rr,
            "cov": cov, "ystd": ystd, "xstd": xstd,
            "degen": degen, "reason": reason,
        })

    # Aggregate summary
    n = len(rrs)
    n_ok = sum(1 for r in rrs if r.success)
    n_f  = sum(1 for r in rrs if r.f_estimated)
    n_deg = sum(1 for m in metrics if m["degen"])
    inls   = [r.num_inliers for r in rrs if r.f_estimated]
    ratios = [r.inlier_ratio for r in rrs if r.f_estimated]
    covs   = [m["cov"] for m in metrics if m["rr"].success]
    ystds  = [m["ystd"] for m in metrics if m["rr"].success]

    print(f"    success (>= {min_inliers} inl)  : {n_ok}/{n}  ({100 * n_ok / n:.1f}%)")
    print(f"    F estimated              : {n_f}/{n}  ({100 * n_f / n:.1f}%)")
    print(f"    near-degenerate flagged  : {n_deg}/{n}  ({100 * n_deg / n:.1f}%)")
    if inls:
        print(f"    inliers / pair           : min={min(inls)}  "
              f"mean={np.mean(inls):.0f}  max={max(inls)}  median={int(np.median(inls))}")
        print(f"    inlier ratio             : mean={100*np.mean(ratios):.1f}%  "
              f"min={100*min(ratios):.1f}%  max={100*max(ratios):.1f}%")
    if covs:
        print(f"    inlier cell coverage     : mean={np.mean(covs):.2f}  "
              f"min={min(covs):.2f}  max={max(covs):.2f}")
        print(f"    inlier Y-std normalized  : mean={np.mean(ystds):.3f}  "
              f"min={min(ystds):.3f}  max={max(ystds):.3f}")

    return metrics


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Phase 5 validation")
    parser.add_argument("--method", default=DEFAULT_F_METHOD)
    parser.add_argument("--threshold", type=float, default=DEFAULT_F_THRESHOLD)
    parser.add_argument("--confidence", type=float, default=DEFAULT_F_CONFIDENCE)
    parser.add_argument("--min-inliers", type=int, default=DEFAULT_MIN_INLIERS)
    parser.add_argument("--regions", type=Path, default=None)
    args = parser.parse_args()

    print("=" * 74)
    print(" PHASE 5 VALIDATION - RANSAC / Fundamental Matrix (both configs)")
    print("=" * 74)

    regions, cal = _resolve_regions(args.regions)

    frames = load_frames(INPUT_FOLDER)
    assert len(frames) > 0
    assert CLEAN_FOLDER.is_dir(), f"Run Phase 2 first: {CLEAN_FOLDER} missing."

    print(f"Loaded {len(frames)} frames.")
    print("Extracting SIFT features from cleaned frames ...")
    feature_sets = extract_features_for_frames(
        frames, method="sift", use_mask=True,
        regions=regions, calibration_size=cal, source_dir=CLEAN_FOLDER,
    )
    img_h, img_w = feature_sets[0].image_shape

    print(f"\nRANSAC config    : method={args.method}  threshold={args.threshold}px  "
          f"confidence={args.confidence}  min_inliers={args.min_inliers}")

    session_pairs = select_pairs(frames, policy="session")

    # ------------------------------------------------------------------ #
    # [A] Baseline: no grid filter                                         #
    # ------------------------------------------------------------------ #
    metrics_base = _run_config(
        "A", feature_sets, session_pairs, grid_filter=False,
        img_h=img_h, img_w=img_w,
        method=args.method, threshold=args.threshold,
        confidence=args.confidence, min_inliers=args.min_inliers,
    )

    # ------------------------------------------------------------------ #
    # [B] Grid-filtered                                                    #
    # ------------------------------------------------------------------ #
    metrics_grid = _run_config(
        "B", feature_sets, session_pairs, grid_filter=True,
        img_h=img_h, img_w=img_w,
        method=args.method, threshold=args.threshold,
        confidence=args.confidence, min_inliers=args.min_inliers,
    )

    # ------------------------------------------------------------------ #
    # [C] Per-pair side-by-side                                            #
    # ------------------------------------------------------------------ #
    print(f"\n[C] Per-pair side-by-side (both configs):")
    hdr = (f"    {'pair':>7}  "
           f"{'A_tent':>6}  {'A_inl':>5}  {'A_rat':>6}  {'A_cov':>5}  {'A_y':>5}  {'A':>4}  "
           f"||  "
           f"{'B_tent':>6}  {'B_inl':>5}  {'B_rat':>6}  {'B_cov':>5}  {'B_y':>5}  {'B':>4}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for ma, mb in zip(metrics_base, metrics_grid):
        a_tag = "OK" if ma["rr"].success else ("F" if ma["rr"].f_estimated else "FAIL")
        b_tag = "OK" if mb["rr"].success else ("F" if mb["rr"].f_estimated else "FAIL")
        if ma["degen"]: a_tag += "*"
        if mb["degen"]: b_tag += "*"
        print(f"    ({ma['rr'].idx_a:2d},{ma['rr'].idx_b:2d})  "
              f"{ma['rr'].num_tentative:6d}  {ma['rr'].num_inliers:5d}  "
              f"{100*ma['rr'].inlier_ratio:5.1f}%  {ma['cov']:5.2f}  {ma['ystd']:5.3f}  {a_tag:>4}  "
              f"||  "
              f"{mb['rr'].num_tentative:6d}  {mb['rr'].num_inliers:5d}  "
              f"{100*mb['rr'].inlier_ratio:5.1f}%  {mb['cov']:5.2f}  {mb['ystd']:5.3f}  {b_tag:>4}")
    print("    (trailing * = flagged as near-degenerate)")

    # ------------------------------------------------------------------ #
    # [D] Representative pairs                                             #
    # ------------------------------------------------------------------ #
    representatives: List[Tuple[Tuple[int, int], str]] = [
        ((0, 1), "easy"),
        ((3, 7), "medium"),
        ((4, 10), "difficult"),
    ]
    rep_pairs = [p for p, _ in representatives]

    # Re-run matching + RANSAC for the 3 pairs under both configs (needed to save
    # visualizations — the earlier runs discarded the intermediate match results).
    print(f"\n[D] Representative pair visualizations (both configs) ...")
    for grid_on, tag in ((False, "base"), (True, "grid")):
        mrs = match_frame_pairs(
            feature_sets, rep_pairs,
            method="flann", ratio=DEFAULT_RATIO, mutual=False,
            grid_filter=grid_on,
            grid_rows=GRID_ROWS, grid_cols=GRID_COLS, grid_max_per_cell=GRID_MAX,
        )
        rrs = estimate_fundamental_for_matches(
            mrs, method=args.method, threshold=args.threshold,
            confidence=args.confidence, min_inliers=args.min_inliers,
        )
        mr_map = {(m.idx_a, m.idx_b): m for m in mrs}
        rr_map = {(r.idx_a, r.idx_b): r for r in rrs}
        for (i, j), label in representatives:
            mr = mr_map[(i, j)]
            rr = rr_map[(i, j)]
            img_a = cv2.imread(str(CLEAN_FOLDER / mr.name_a))
            img_b = cv2.imread(str(CLEAN_FOLDER / mr.name_b))

            # Before RANSAC
            vis_before = draw_tentative_matches(img_a, img_b, mr, max_draw=80)
            out_before = OUTPUT_DIR / f"phase5_{label}_{tag}_before_{i:02d}_{j:02d}.png"
            cv2.imwrite(str(out_before), vis_before)

            # After RANSAC
            vis_after = draw_inlier_matches(img_a, img_b, mr, rr, max_draw=80)
            out_after = OUTPUT_DIR / f"phase5_{label}_{tag}_after_{i:02d}_{j:02d}.png"
            cv2.imwrite(str(out_after), vis_after)

            # Epipolar lines
            if rr.success:
                ea, eb = draw_epipolar_lines(img_a, img_b, mr, rr, num_samples=8)
                combined = np.hstack([ea, eb])
                out_epi = OUTPUT_DIR / f"phase5_{label}_{tag}_epipolar_{i:02d}_{j:02d}.png"
                cv2.imwrite(str(out_epi), combined)

            inl_pct = 100 * rr.inlier_ratio
            print(f"    {label:10} [{tag}] ({i:2d},{j:2d}): "
                  f"tent={rr.num_tentative:3d}  inl={rr.num_inliers:3d}  "
                  f"({inl_pct:5.1f}%)  "
                  f"{'OK' if rr.success else ('F' if rr.f_estimated else 'FAIL')}")

    # ------------------------------------------------------------------ #
    # [E] Commentary                                                       #
    # ------------------------------------------------------------------ #
    n = len(metrics_base)
    ok_A  = sum(1 for m in metrics_base if m["rr"].success)
    ok_B  = sum(1 for m in metrics_grid if m["rr"].success)
    deg_A = sum(1 for m in metrics_base if m["degen"])
    deg_B = sum(1 for m in metrics_grid if m["degen"])
    inl_A = [m["rr"].num_inliers for m in metrics_base if m["rr"].success]
    inl_B = [m["rr"].num_inliers for m in metrics_grid if m["rr"].success]
    cov_A = [m["cov"] for m in metrics_base if m["rr"].success]
    cov_B = [m["cov"] for m in metrics_grid if m["rr"].success]
    ys_A  = [m["ystd"] for m in metrics_base if m["rr"].success]
    ys_B  = [m["ystd"] for m in metrics_grid if m["rr"].success]

    def _safe_mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    print("\n[E] Commentary")
    print("    ----------")
    print(f"    Success rate                  : A={ok_A}/{n}  B={ok_B}/{n}")
    print(f"    Near-degenerate flagged       : A={deg_A}/{n}  B={deg_B}/{n}")
    print(f"    Mean inliers (successful)     : A={_safe_mean(inl_A):.0f}  "
          f"B={_safe_mean(inl_B):.0f}")
    print(f"    Mean inlier cell coverage     : A={_safe_mean(cov_A):.2f}  "
          f"B={_safe_mean(cov_B):.2f}  (of {TOTAL_CELLS} cells)")
    print(f"    Mean inlier Y-std normalized  : A={_safe_mean(ys_A):.3f}  "
          f"B={_safe_mean(ys_B):.3f}")

    # Decide which config to recommend
    def _score(ok_count, mean_inl, mean_cov, mean_ystd):
        # Normalized composite: success rate, inlier count, spatial diversity.
        return (
            ok_count / max(n, 1)
            + 0.002 * mean_inl
            + mean_cov
            + mean_ystd * 5.0
        )

    sA = _score(ok_A, _safe_mean(inl_A), _safe_mean(cov_A), _safe_mean(ys_A))
    sB = _score(ok_B, _safe_mean(inl_B), _safe_mean(cov_B), _safe_mean(ys_B))
    better = "A (no grid filter)" if sA >= sB else "B (grid filter)"
    print(f"\n    Composite score               : A={sA:.3f}  B={sB:.3f}")
    print(f"    Recommended for next stage    : {better}")

    # Degenerate pairs — list them for both configs
    print(f"\n    Near-degenerate pairs under A : ", end="")
    print(", ".join(f"({m['rr'].idx_a},{m['rr'].idx_b})"
                     for m in metrics_base if m["degen"]) or "(none)")
    print(f"    Near-degenerate pairs under B : ", end="")
    print(", ".join(f"({m['rr'].idx_a},{m['rr'].idx_b})"
                     for m in metrics_grid if m["degen"]) or "(none)")

    print("\n" + "=" * 74)
    print(" PHASE 5 VALIDATION: DONE")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
