"""Phase 7 (EXPERIMENTAL): telemetry-based epipolar prior vs vision baseline.

This script is intentionally kept OUTSIDE the main Phase 5/6 pipeline. It
does not replace the RANSAC-based geometry; it runs alongside it and
produces diagnostic outputs:

    outputs/phase7/telemetry_parsed.csv        per-frame parsed telemetry
    outputs/phase7/relative_pose_diag.csv      per-pair coarse relative pose
    outputs/phase7/epipolar_compare.csv        per-(pair, query) comparison
    outputs/phase7/compare_<label>_q<k>.png    side-by-side visualization

Pipeline (per run):
  1. Load telemetry JSON (and optionally run best-effort OCR to fill
     missing fields, if pytesseract is available).
  2. Build an ENU reference and per-frame camera poses.
  3. For each requested pair:
       a. Run the standard vision pipeline (matching + RANSAC) on the
          cleaned frames to obtain F_vision.
       b. Pick K query points from the RANSAC inlier set (so both the
          vision line and the telemetry line are anchored on the same
          source pixel).
       c. Compute the telemetry-based epipolar prior for that pixel.
       d. Compare the two.

Usage (from repository root):
    python scripts/run_phase7_telemetry_epipolar.py
    python scripts/run_phase7_telemetry_epipolar.py --pairs "0,1 3,7" --num-points 3
    python scripts/run_phase7_telemetry_epipolar.py --run-ocr
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    parse_pairs_arg,
    DEFAULT_F_METHOD,
    DEFAULT_F_THRESHOLD,
    DEFAULT_F_CONFIDENCE,
    DEFAULT_MIN_INLIERS,
    estimate_fundamental_for_matches,
)
from src.telemetry import (  # noqa: E402
    CameraIntrinsics,
    FrameTelemetry,
    PYTESSERACT_AVAILABLE,
    build_camera_pose,
    compare_epipolar_lines,
    draw_epipolar_comparison,
    epipolar_line_from_pose,
    load_camera_intrinsics,
    load_frame_telemetry_json,
    ocr_extract_frame_telemetry,
    pose_completeness,
    relative_pose,
    save_frame_telemetry_json,
)
from src.telemetry.pose import pick_reference_origin  # noqa: E402


INPUT_FOLDER = REPO_ROOT / "drones_images_input"
CLEAN_FOLDER = REPO_ROOT / "outputs" / "clean_frames"
OUTPUT_DIR = REPO_ROOT / "outputs" / "phase7"
DEFAULT_REGIONS_JSON = REPO_ROOT / "config" / "overlay_regions.json"
DEFAULT_TELEMETRY_JSON = REPO_ROOT / "config" / "frame_telemetry.json"
DEFAULT_INTRINSICS_JSON = REPO_ROOT / "config" / "camera_intrinsics.json"

DEFAULT_PAIRS: List[Tuple[int, int]] = [(0, 1), (3, 7), (4, 10)]
LABELS = {(0, 1): "easy", (3, 7): "medium", (4, 10): "difficult"}

GRID_ROWS = 4
GRID_COLS = 5
GRID_MAX = 15


# ---------------------------------------------------------------------- #
# Helpers                                                                 #
# ---------------------------------------------------------------------- #

def _resolve_regions(path):
    if path is None and DEFAULT_REGIONS_JSON.is_file():
        path = DEFAULT_REGIONS_JSON
    if path is not None:
        regions, cal = load_regions_from_json(path)
        return regions, cal
    return list(DEFAULT_OVERLAY_REGIONS), CALIBRATION_SIZE


def _regions_as_dict(regions) -> Dict[str, Tuple[int, int, int, int]]:
    """Flatten list of OverlayRegion into a {name: (x, y, w, h)} map."""
    return {r.name: (r.x, r.y, r.w, r.h) for r in regions}


def _nan_to_str(x) -> str:
    if isinstance(x, float) and (math.isnan(x) or not math.isfinite(x)):
        return ""
    return f"{x}"


def _select_source_pixel(
    inlier_pts_a: np.ndarray,
    image_shape: Tuple[int, int],
    target_xy: Tuple[float, float],
) -> Optional[np.ndarray]:
    """Pick the inlier in A closest to `target_xy` (ideal ground-truth-like)."""
    if len(inlier_pts_a) == 0:
        return None
    dx = inlier_pts_a[:, 0] - target_xy[0]
    dy = inlier_pts_a[:, 1] - target_xy[1]
    d2 = dx * dx + dy * dy
    return inlier_pts_a[int(np.argmin(d2))]


# ---------------------------------------------------------------------- #
# Main                                                                    #
# ---------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 7 (experimental): telemetry-based coarse epipolar prior."
    )
    parser.add_argument("--pairs", default=None,
                        help='Pairs to run, e.g. "0,1 3,7 4,10" (default: 3 representatives).')
    parser.add_argument("--num-points", type=int, default=3,
                        help="Query points per pair (default: 3).")
    parser.add_argument("--telemetry", type=Path, default=DEFAULT_TELEMETRY_JSON,
                        help=f"Per-frame telemetry JSON (default: {DEFAULT_TELEMETRY_JSON.name}).")
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS_JSON,
                        help=f"Camera intrinsics JSON (default: {DEFAULT_INTRINSICS_JSON.name}).")
    parser.add_argument("--regions", type=Path, default=None)
    parser.add_argument("--run-ocr", action="store_true",
                        help="Try OCR extraction to fill missing telemetry fields.")
    parser.add_argument("--save-ocr-to", type=Path, default=None,
                        help="Optional JSON path to dump OCR-extracted telemetry.")
    parser.add_argument("--method", default=DEFAULT_F_METHOD)
    parser.add_argument("--threshold", type=float, default=DEFAULT_F_THRESHOLD)
    parser.add_argument("--confidence", type=float, default=DEFAULT_F_CONFIDENCE)
    parser.add_argument("--min-inliers", type=int, default=DEFAULT_MIN_INLIERS)
    args = parser.parse_args()

    explicit = parse_pairs_arg(args.pairs)
    pairs = explicit if explicit is not None else list(DEFAULT_PAIRS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- inputs ------------------------------------------------------
    frames = load_frames(INPUT_FOLDER)
    if not CLEAN_FOLDER.is_dir():
        print(f"ERROR: {CLEAN_FOLDER} not found. Run Phase 2 first.")
        return 1

    regions, cal = _resolve_regions(args.regions)
    region_map = _regions_as_dict(regions)

    intrinsics: Optional[CameraIntrinsics] = None
    if args.intrinsics.is_file():
        intrinsics = load_camera_intrinsics(args.intrinsics)
        print(f"Intrinsics       : loaded {args.intrinsics.name} "
              f"(fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f} "
              f"cx={intrinsics.cx:.1f} cy={intrinsics.cy:.1f} "
              f"{intrinsics.width}x{intrinsics.height})")
    else:
        print(f"Intrinsics       : {args.intrinsics} not found - epipolar prior "
              "will be limited to directional mode only.")

    frame_names = [f.name for f in frames]
    telemetry_list: List[FrameTelemetry]
    meta: Dict = {}
    if args.telemetry.is_file():
        telemetry_list, meta = load_frame_telemetry_json(args.telemetry, frame_names)
        print(f"Telemetry        : loaded {args.telemetry.name}")
    else:
        telemetry_list = [FrameTelemetry(frame_name=n) for n in frame_names]
        print(f"Telemetry        : no file, starting from NaN skeleton.")

    # ---- optional OCR pass -------------------------------------------
    if args.run_ocr:
        if not PYTESSERACT_AVAILABLE:
            print("OCR              : pytesseract not available, skipping.")
        else:
            print("OCR              : running per-frame extraction on ORIGINAL frames ...")
            for i, f in enumerate(frames):
                img = f.load_image()
                ocr_tel = ocr_extract_frame_telemetry(
                    f.name, img, region_map, calibration_size=cal,
                )
                # Merge: only fill fields that the JSON declared as NaN.
                tel = telemetry_list[i]
                for attr in ("latitude", "longitude", "altitude_m",
                             "heading_deg", "pitch_deg", "roll_deg",
                             "gimbal_pitch_deg", "gimbal_yaw_deg", "gimbal_roll_deg"):
                    cur = getattr(tel, attr)
                    if isinstance(cur, float) and math.isnan(cur):
                        setattr(tel, attr, getattr(ocr_tel, attr))
                tel.raw_ocr.update(ocr_tel.raw_ocr)
                if not tel.notes:
                    tel.notes = ocr_tel.notes
            if args.save_ocr_to is not None:
                ref = pick_reference_origin(telemetry_list)
                save_frame_telemetry_json(
                    telemetry_list, args.save_ocr_to, reference=ref,
                )
                print(f"OCR              : saved merged telemetry -> {args.save_ocr_to}")

    # ---- parsed telemetry CSV ----------------------------------------
    tele_csv = OUTPUT_DIR / "telemetry_parsed.csv"
    with tele_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "index", "frame", "lat", "lon", "alt_m", "alt_is_agl",
            "heading_deg", "pitch_deg", "roll_deg",
            "gimbal_pitch_deg", "gimbal_yaw_deg", "gimbal_roll_deg",
            "has_gps", "has_altitude", "has_heading", "timestamp", "notes",
        ])
        for i, t in enumerate(telemetry_list):
            w.writerow([
                i, t.frame_name,
                _nan_to_str(t.latitude), _nan_to_str(t.longitude),
                _nan_to_str(t.altitude_m), t.altitude_is_agl,
                _nan_to_str(t.heading_deg), _nan_to_str(t.pitch_deg),
                _nan_to_str(t.roll_deg),
                _nan_to_str(t.gimbal_pitch_deg),
                _nan_to_str(t.gimbal_yaw_deg),
                _nan_to_str(t.gimbal_roll_deg),
                t.has_gps(), t.has_altitude(), t.has_heading(),
                t.timestamp or "", t.notes,
            ])
    print(f"Parsed telemetry -> {tele_csv}")

    # ---- pose building -----------------------------------------------
    ref = None
    if "reference" in meta and meta["reference"].get("latitude") is not None:
        r = meta["reference"]
        ref = (float(r["latitude"]), float(r["longitude"]),
               float(r.get("altitude_m", 0.0)))
    if ref is None:
        ref = pick_reference_origin(telemetry_list)
    if ref is None:
        ref = (0.0, 0.0, 0.0)
        print("Pose origin      : no GPS in telemetry -> using (0, 0, 0). "
              "All poses will lack position until lat/lon are provided.")
    else:
        print(f"Pose origin (ENU): lat={ref[0]:.6f} lon={ref[1]:.6f} alt={ref[2]:.2f} m")

    default_pitch = float(meta.get("default_pitch_deg", 0.0))
    default_roll  = float(meta.get("default_roll_deg", 0.0))

    poses = [
        build_camera_pose(
            t, ref[0], ref[1], ref[2],
            default_pitch_deg=default_pitch, default_roll_deg=default_roll,
        )
        for t in telemetry_list
    ]

    print("\nPer-frame pose completeness:")
    for i, p in enumerate(poses):
        print(f"  [{i:2d}] {p.frame_name}  -> {pose_completeness(p)}")

    # ---- vision pipeline (matching + RANSAC) on the same pairs --------
    print("\nVision pipeline (SIFT + FLANN + RANSAC) on requested pairs ...")
    feature_sets = extract_features_for_frames(
        frames, method="sift", use_mask=True,
        regions=regions, calibration_size=cal, source_dir=CLEAN_FOLDER,
    )
    match_results = match_frame_pairs(
        feature_sets, pairs,
        method="flann", ratio=DEFAULT_RATIO, mutual=False,
        grid_filter=True,
        grid_rows=GRID_ROWS, grid_cols=GRID_COLS, grid_max_per_cell=GRID_MAX,
    )
    ransac_results = estimate_fundamental_for_matches(
        match_results,
        method=args.method, threshold=args.threshold,
        confidence=args.confidence, min_inliers=args.min_inliers,
    )
    mr_map = {(m.idx_a, m.idx_b): m for m in match_results}
    rr_map = {(r.idx_a, r.idx_b): r for r in ransac_results}

    # ---- per-pair diagnostics ----------------------------------------
    pose_csv = OUTPUT_DIR / "relative_pose_diag.csv"
    cmp_csv  = OUTPUT_DIR / "epipolar_compare.csv"
    pose_rows: List[List] = []
    cmp_rows: List[List] = []

    print("\nPair-wise telemetry-vs-vision comparison:")
    for (i, j) in pairs:
        label = LABELS.get((i, j), f"pair_{i:02d}_{j:02d}")
        mr = mr_map.get((i, j))
        rr = rr_map.get((i, j))
        vision_ok = (mr is not None and rr is not None
                     and rr.f_estimated and rr.F is not None
                     and rr.num_inliers >= 6)

        rel = relative_pose(poses[i], poses[j], idx_a=i, idx_b=j)
        pose_rows.append([
            i, j, rel.name_a, rel.name_b,
            pose_completeness(poses[i]), pose_completeness(poses[j]),
            rel.orientation_valid, rel.position_valid,
            _nan_to_str(rel.baseline_m),
            _nan_to_str(float(rel.t_rel[0])),
            _nan_to_str(float(rel.t_rel[1])),
            _nan_to_str(float(rel.t_rel[2])),
        ])
        print(f"  {label:10} ({i:2d},{j:2d}): "
              f"pose_A={pose_completeness(poses[i])}  "
              f"pose_B={pose_completeness(poses[j])}  "
              f"baseline={_nan_to_str(rel.baseline_m)} m  "
              f"orient_ok={rel.orientation_valid}  pos_ok={rel.position_valid}  "
              f"vision_ok={vision_ok}"
              + ("" if rr is None else f" (inl={rr.num_inliers}/{rr.num_tentative})"))

        name_a = rr.name_a if rr is not None else frames[i].name
        name_b = rr.name_b if rr is not None else frames[j].name
        img_a = cv2.imread(str(CLEAN_FOLDER / name_a))
        img_b = cv2.imread(str(CLEAN_FOLDER / name_b))
        if img_a is None or img_b is None:
            print(f"    -> missing cleaned images, skipping visualization.")
            continue

        # Scale intrinsics to the actual image resolution.
        if intrinsics is not None:
            h_b, w_b = img_b.shape[:2]
            h_a, w_a = img_a.shape[:2]
            K_a = intrinsics.scaled_to(w_a, h_a)
            K_b = intrinsics.scaled_to(w_b, h_b)
        else:
            K_a = K_b = None

        pts_a = (rr.inlier_points_a(mr.fs_a_filtered) if vision_ok
                 else np.zeros((0, 2), dtype=np.float32))

        # Pick up to `num_points` source pixels spread across image A.
        h, w = img_a.shape[:2]
        targets = [
            (0.30 * w, 0.30 * h),
            (0.70 * w, 0.30 * h),
            (0.50 * w, 0.60 * h),
            (0.30 * w, 0.80 * h),
            (0.70 * w, 0.80 * h),
        ][: args.num_points]

        for q_idx, tgt in enumerate(targets):
            # Prefer an actual RANSAC inlier near the target, fall back to
            # the target pixel itself when vision has no inliers.
            src_px_pref = _select_source_pixel(pts_a, (h, w), tgt)
            if src_px_pref is not None:
                src_px = src_px_pref
            else:
                src_px = np.array(tgt, dtype=np.float32)

            # Vision line only when RANSAC succeeded. Otherwise use NaNs so
            # the comparison is visibly "vision unavailable".
            if vision_ok:
                vis_line = cv2.computeCorrespondEpilines(
                    src_px.reshape(1, 1, 2).astype(np.float32), 1, rr.F,
                ).reshape(3).astype(np.float64)
            else:
                vis_line = np.array([np.nan, np.nan, np.nan], dtype=np.float64)

            if K_a is not None and K_b is not None:
                prior = epipolar_line_from_pose(
                    source_pixel=tuple(src_px.tolist()),
                    rel=rel,
                    intrinsics_a=K_a, intrinsics_b=K_b,
                    pose_a=poses[i], pose_b=poses[j],
                    telemetry_a=telemetry_list[i], telemetry_b=telemetry_list[j],
                )
            else:
                # No intrinsics -> directional-only prior.
                from src.telemetry.epipolar import directional_prior_from_translation
                az, tag = directional_prior_from_translation(
                    poses[i], poses[j],
                    telemetry_list[i], telemetry_list[j],
                )
                from src.telemetry.types import EpipolarPrior
                prior = EpipolarPrior(
                    source_pixel=tuple(src_px.tolist()),
                    mode="directional" if az is not None else "none",
                    direction_deg=az,
                    note=tag,
                )

            cmp = compare_epipolar_lines(vis_line, prior)

            vis = draw_epipolar_comparison(
                img_a, img_b,
                source_pixel=tuple(src_px.tolist()),
                vision_line=vis_line, prior=prior, cmp=cmp,
            )
            out_png = OUTPUT_DIR / f"compare_{label}_q{q_idx}_{i:02d}_{j:02d}.png"
            cv2.imwrite(str(out_png), vis)

            cmp_rows.append([
                i, j, label, q_idx,
                float(src_px[0]), float(src_px[1]),
                prior.mode,
                _nan_to_str(prior.direction_deg),
                _nan_to_str(prior.epipole_hint[0] if prior.epipole_hint else float("nan")),
                _nan_to_str(prior.epipole_hint[1] if prior.epipole_hint else float("nan")),
                _nan_to_str(cmp.angle_diff_deg),
                _nan_to_str(cmp.epipole_hint_distance_px),
                prior.note,
                out_png.name,
            ])
            print(f"      q{q_idx}: src=({src_px[0]:6.1f},{src_px[1]:6.1f})  "
                  f"mode={prior.mode}  "
                  f"angle_diff={_nan_to_str(cmp.angle_diff_deg)} deg  "
                  f"epipole_dist={_nan_to_str(cmp.epipole_hint_distance_px)} px "
                  f"-> {out_png.name}")

    with pose_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "idx_a", "idx_b", "name_a", "name_b",
            "pose_a", "pose_b", "orientation_valid", "position_valid",
            "baseline_m", "t_rel_x", "t_rel_y", "t_rel_z",
        ])
        w.writerows(pose_rows)
    with cmp_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "idx_a", "idx_b", "label", "q",
            "src_u", "src_v", "prior_mode",
            "prior_direction_deg", "epipole_hint_u", "epipole_hint_v",
            "angle_diff_deg", "epipole_hint_distance_px",
            "note", "visualization",
        ])
        w.writerows(cmp_rows)
    print(f"\nRelative pose CSV -> {pose_csv}")
    print(f"Comparison CSV    -> {cmp_csv}")

    if not any(t.has_gps() for t in telemetry_list):
        print("\nNOTE: all telemetry is NaN. Fill in "
              f"{args.telemetry.relative_to(REPO_ROOT)} or run --run-ocr to populate "
              "GPS/heading before meaningful comparisons can be produced.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
