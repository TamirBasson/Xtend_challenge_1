"""Phase 6 validation: epipolar-guided point transfer.

For each representative pair (easy / medium / difficult):
  * take K source pixels from the RANSAC inlier set (they carry ground-truth
    correspondences in the target frame),
  * run `src.transfer.transfer_point` with an NCC patch comparator,
  * measure pixel error between the predicted and the ground-truth target
    pixel, and report aggregate statistics,
  * save a side-by-side visualization per query point.

This is a baseline sanity-check. We are not yet doing sub-pixel refinement,
multi-frame fusion, or learned descriptors — that is the job of later phases.
"""

from __future__ import annotations

import argparse
import csv
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
    match_frame_pairs,
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

REPRESENTATIVES: List[Tuple[Tuple[int, int], str]] = [
    ((0, 1),  "easy"),
    ((3, 7),  "medium"),
    ((4, 10), "difficult"),
]

GRID_ROWS = 4
GRID_COLS = 5
GRID_MAX = 15

# Success thresholds used for the "within-X-px" rate.
SUCCESS_THRESHOLDS_PX = (5.0, 10.0, 20.0)


def _resolve_regions(path):
    if path is None and DEFAULT_REGIONS_JSON.is_file():
        path = DEFAULT_REGIONS_JSON
    if path is not None:
        regions, cal = load_regions_from_json(path)
        print(f"Overlay regions : loaded {len(regions)} from {path}")
        return regions, cal
    print(f"Overlay regions : built-in defaults ({len(DEFAULT_OVERLAY_REGIONS)} regions)")
    return list(DEFAULT_OVERLAY_REGIONS), CALIBRATION_SIZE


def _spread_picks(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    image_shape: Tuple[int, int],
    k: int,
    patch_margin: int,
    rows: int = 3,
    cols: int = 3,
) -> List[int]:
    """Pick up to k indices into pts_a/pts_b spread across a rows x cols grid.

    Falls back to any remaining in-bounds inliers if the grid does not yield
    enough picks.
    """
    h, w = image_shape
    if len(pts_a) == 0 or k <= 0:
        return []

    cell_h = h / rows
    cell_w = w / cols
    picked: List[int] = []
    used = set()

    for r in range(rows):
        for c in range(cols):
            if len(picked) >= k:
                break
            for i, (x, y) in enumerate(pts_a):
                if i in used:
                    continue
                if not (patch_margin <= x <= w - 1 - patch_margin
                        and patch_margin <= y <= h - 1 - patch_margin):
                    continue
                if int(y / cell_h) == r and int(x / cell_w) == c:
                    picked.append(i)
                    used.add(i)
                    break

    if len(picked) < k:
        for i, (x, y) in enumerate(pts_a):
            if i in used:
                continue
            if not (patch_margin <= x <= w - 1 - patch_margin
                    and patch_margin <= y <= h - 1 - patch_margin):
                continue
            picked.append(i)
            used.add(i)
            if len(picked) >= k:
                break

    return picked[:k]


def _summarize_errors(errors: Sequence[float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    arr = np.asarray([e for e in errors if np.isfinite(e)], dtype=np.float64)
    out["n"] = float(len(arr))
    if arr.size == 0:
        for t in SUCCESS_THRESHOLDS_PX:
            out[f"le_{int(t)}px_pct"] = 0.0
        out["median_px"] = float("nan")
        out["mean_px"] = float("nan")
        out["p90_px"] = float("nan")
        out["max_px"] = float("nan")
        return out
    out["median_px"] = float(np.median(arr))
    out["mean_px"] = float(np.mean(arr))
    out["p90_px"] = float(np.percentile(arr, 90))
    out["max_px"] = float(np.max(arr))
    for t in SUCCESS_THRESHOLDS_PX:
        out[f"le_{int(t)}px_pct"] = float(100.0 * np.mean(arr <= t))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6 validation")
    parser.add_argument("--num-points", type=int, default=6,
                        help="Query points per pair (default: 6).")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE,
                        help=f"NCC patch size (odd int, default: {DEFAULT_PATCH_SIZE}).")
    parser.add_argument("--step", type=float, default=DEFAULT_STEP,
                        help=f"Sampling step along the epipolar line (default: {DEFAULT_STEP}).")
    parser.add_argument("--regions", type=Path, default=None)
    parser.add_argument("--method", default=DEFAULT_F_METHOD)
    parser.add_argument("--threshold", type=float, default=DEFAULT_F_THRESHOLD)
    parser.add_argument("--confidence", type=float, default=DEFAULT_F_CONFIDENCE)
    parser.add_argument("--min-inliers", type=int, default=DEFAULT_MIN_INLIERS)
    parser.add_argument("--draw-samples", action="store_true",
                        help="Overlay sampled candidate points on the target image.")
    args = parser.parse_args()

    print("=" * 74)
    print(" PHASE 6 VALIDATION - Epipolar-guided point transfer (baseline)")
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

    pairs = [p for p, _ in REPRESENTATIVES]
    print(f"\nPairs            : {pairs}")
    print(f"Query points/pair: {args.num_points}")
    print(f"Patch size       : {args.patch_size}  | step: {args.step} px")
    print(f"RANSAC           : method={args.method}  threshold={args.threshold}px  "
          f"confidence={args.confidence}  min_inliers={args.min_inliers}")

    print("\nMatching + RANSAC on representative pairs ...")
    mrs = match_frame_pairs(
        feature_sets, pairs,
        method="flann", ratio=DEFAULT_RATIO, mutual=False,
        grid_filter=True,
        grid_rows=GRID_ROWS, grid_cols=GRID_COLS, grid_max_per_cell=GRID_MAX,
    )
    rrs = estimate_fundamental_for_matches(
        mrs, method=args.method, threshold=args.threshold,
        confidence=args.confidence, min_inliers=args.min_inliers,
    )
    mr_map = {(m.idx_a, m.idx_b): m for m in mrs}
    rr_map = {(r.idx_a, r.idx_b): r for r in rrs}

    per_pair: Dict[str, Dict[str, float]] = {}
    all_rows: List[List] = []
    all_errors: List[float] = []

    print("\n[A] Per-pair transfer results")
    print("    " + "-" * 70)
    for (i, j), label in REPRESENTATIVES:
        mr = mr_map.get((i, j))
        rr = rr_map.get((i, j))
        if mr is None or rr is None or not rr.success or rr.F is None:
            reason = "no result" if rr is None else (
                "RANSAC failed" if not rr.success else "no F")
            print(f"    {label:10} ({i:2d},{j:2d}): {reason}, skipped.")
            per_pair[label] = {"n": 0}
            continue

        img_a = cv2.imread(str(CLEAN_FOLDER / rr.name_a))
        img_b = cv2.imread(str(CLEAN_FOLDER / rr.name_b))
        if img_a is None or img_b is None:
            print(f"    {label:10} ({i:2d},{j:2d}): could not load images, skipped.")
            per_pair[label] = {"n": 0}
            continue

        pts_a_all = rr.inlier_points_a(mr.fs_a_filtered)
        pts_b_all = rr.inlier_points_b(mr.fs_b_filtered)

        picks = _spread_picks(
            pts_a_all, pts_b_all,
            image_shape=img_a.shape[:2],
            k=args.num_points,
            patch_margin=args.patch_size // 2,
        )

        errs: List[float] = []
        print(f"    {label:10} ({i:2d},{j:2d}): inliers={rr.num_inliers}  "
              f"picks={len(picks)}")

        for q_idx, pi in enumerate(picks):
            src_px = pts_a_all[pi]
            gt_px = pts_b_all[pi]
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
            errs.append(err)
            all_errors.append(err)

            vis = draw_transfer(
                img_a, img_b, result,
                ground_truth=tuple(gt_px.tolist()),
                draw_samples=args.draw_samples,
            )
            out = OUTPUT_DIR / f"phase6_{label}_q{q_idx}_{i:02d}_{j:02d}.png"
            cv2.imwrite(str(out), vis)

            print(f"        q{q_idx}: src=({src_px[0]:6.1f},{src_px[1]:6.1f})  "
                  f"pred="
                  f"{'(%6.1f,%6.1f)' % result.predicted_pixel if result.predicted_pixel else '     none    '}"
                  f"  gt=({gt_px[0]:6.1f},{gt_px[1]:6.1f})  "
                  f"score={result.score:.3f}  err={err:.1f}px")

            all_rows.append([
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

        per_pair[label] = _summarize_errors(errs)

    # -------- Per-pair summary --------
    print("\n[B] Per-pair error summary")
    print("    " + "-" * 70)
    hdr = (f"    {'pair':10}  {'n':>3}  {'med':>6}  {'mean':>6}  "
           f"{'p90':>6}  {'max':>6}   "
           + "  ".join(f"<={int(t)}px" for t in SUCCESS_THRESHOLDS_PX))
    print(hdr)
    for (_, label) in REPRESENTATIVES:
        s = per_pair.get(label, {"n": 0})
        if s.get("n", 0) == 0:
            print(f"    {label:10}   -    skipped")
            continue
        rates = "  ".join(f"{s[f'le_{int(t)}px_pct']:5.1f}%"
                          for t in SUCCESS_THRESHOLDS_PX)
        print(f"    {label:10}  {int(s['n']):>3}  "
              f"{s['median_px']:6.1f}  {s['mean_px']:6.1f}  "
              f"{s['p90_px']:6.1f}  {s['max_px']:6.1f}   {rates}")

    # -------- Aggregate summary --------
    agg = _summarize_errors(all_errors)
    print("\n[C] Overall summary")
    print("    " + "-" * 70)
    if agg.get("n", 0) == 0:
        print("    No successful transfers — cannot report aggregate error.")
    else:
        print(f"    queries           : {int(agg['n'])}")
        print(f"    median error      : {agg['median_px']:.1f} px")
        print(f"    mean error        : {agg['mean_px']:.1f} px")
        print(f"    90-th percentile  : {agg['p90_px']:.1f} px")
        print(f"    worst case        : {agg['max_px']:.1f} px")
        for t in SUCCESS_THRESHOLDS_PX:
            print(f"    rate <= {int(t):>3} px    : {agg[f'le_{int(t)}px_pct']:5.1f}%")

    # -------- Persist CSV --------
    csv_path = OUTPUT_DIR / "phase6_transfer_validation.csv"
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
        w.writerows(all_rows)
    print(f"\nSaved validation CSV : {csv_path}")

    print("\n" + "=" * 74)
    print(" PHASE 6 VALIDATION: DONE")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
