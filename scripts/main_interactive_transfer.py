"""Interactive pixel transfer: click a pixel in a source frame and transfer it
to every other frame using the existing offline pipeline.

Pipeline per target frame:
    1. Feature extraction on cleaned source/target frame
         * `--method superpoint` (default) -> SuperPoint
         * `--method sift`                 -> SIFT (masked by overlay regions)
    2. Pairwise descriptor matching
         * SuperPoint  -> LightGlue (learned matcher; SuperGlue-family)
         * SIFT        -> FLANN + Lowe's ratio + grid filter
    3. RANSAC fundamental-matrix estimation (defines the epipolar line).
    4. Local-affine point transfer (see `src.local_transfer`):
         click -> deep matches -> epipolar-band filter (soft, ~20 px)
               -> K matches nearest the click -> local affine
               -> apply affine to the clicked point -> final target pixel
       No NCC, no scanning along the epipolar line.
    5. Side-by-side visualization via `src.transfer.draw_transfer`.

All outputs for a single session are written to:
    outputs/YYYYMMDD_HHMMSS/
        source_<idx>_<name>.png        (source frame + clicked pixel marker)
        target_<idx>_<name>.png        (side-by-side source|target visualization)
        transfer_results.csv           (per-frame log)

Usage (from repository root):
    python scripts/main_interactive_transfer.py
    python scripts/main_interactive_transfer.py --source-index 4
    python scripts/main_interactive_transfer.py --epipolar-band 25 --k-neighbors 12
    python scripts/main_interactive_transfer.py --method sift

Controls:
    Left-click inside the source image to pick a pixel.
    Press 'r' to clear the current click and re-pick.
    Press ENTER or SPACE to confirm and proceed.
    Press ESC or 'q' to abort without transferring.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

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
    SUPPORTED_PIPELINES,
    match_pair,
    DEFAULT_F_METHOD,
    DEFAULT_F_THRESHOLD,
    DEFAULT_F_CONFIDENCE,
    estimate_fundamental,
    DEFAULT_PATCH_SIZE,
    DEFAULT_STEP,
    transfer_point,
    draw_transfer,
    transfer_point_local_affine,
    DEFAULT_EPIPOLAR_BAND_PX,
    DEFAULT_K_NEIGHBORS,
)


INPUT_FOLDER = REPO_ROOT / "drones_images_input"
CLEAN_FOLDER = REPO_ROOT / "outputs" / "clean_frames"
OUTPUT_ROOT = REPO_ROOT / "outputs"
DEFAULT_REGIONS_JSON = REPO_ROOT / "config" / "overlay_regions.json"

GRID_ROWS = 4
GRID_COLS = 5
GRID_MAX = 15
DEFAULT_MIN_INLIERS =  15       # relaxed w.r.t. Phase 5; F is what we actually need
MIN_EPIPOLAR_BAND_PX = 10.0
MIN_BAND_MATCHES_TO_SHOW = 8
MIN_BAND_MATCHES_BY_SOURCE_INDEX = {
    8: 4,
}


# ---------------------------------------------------------------------- #
# Helpers                                                                 #
# ---------------------------------------------------------------------- #

def _resolve_regions(path: Optional[Path]):
    if path is None and DEFAULT_REGIONS_JSON.is_file():
        path = DEFAULT_REGIONS_JSON
    if path is not None:
        regions, cal = load_regions_from_json(path)
        print(f"Overlay regions : loaded {len(regions)} from {path}")
        return regions, cal
    print(f"Overlay regions : built-in defaults ({len(DEFAULT_OVERLAY_REGIONS)} regions)")
    return list(DEFAULT_OVERLAY_REGIONS), CALIBRATION_SIZE


def _read_clean_image(frame_name: str) -> Optional[np.ndarray]:
    path = CLEAN_FOLDER / frame_name
    if not path.is_file():
        return None
    return cv2.imread(str(path))


class _Picker:
    """Mouse-click picker with an OpenCV window."""

    def __init__(self, image: np.ndarray, window_name: str):
        self.window = window_name
        self.base = image
        self.point: Optional[Tuple[int, int]] = None
        self.done = False
        self.aborted = False

    def _redraw(self) -> None:
        canvas = self.base.copy()
        hint = "Click a pixel | r=reset | ENTER/SPACE=confirm | ESC/q=abort"
        cv2.putText(canvas, hint, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        if self.point is not None:
            u, v = self.point
            cv2.drawMarker(canvas, (u, v), (0, 0, 255),
                           cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
            cv2.circle(canvas, (u, v), 8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"({u}, {v})", (u + 12, v - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(self.window, canvas)

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point = (int(x), int(y))
            self._redraw()

    def run(self) -> Optional[Tuple[int, int]]:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        h, w = self.base.shape[:2]
        cv2.resizeWindow(self.window, min(w, 1600), min(h, 900))
        cv2.setMouseCallback(self.window, self._on_mouse)
        self._redraw()
        try:
            while True:
                key = cv2.waitKey(20) & 0xFF
                if key in (13, 32):                 # ENTER or SPACE
                    if self.point is not None:
                        self.done = True
                        break
                elif key in (27, ord("q"), ord("Q")):  # ESC or q
                    self.aborted = True
                    break
                elif key in (ord("r"), ord("R")):
                    self.point = None
                    self._redraw()
        finally:
            cv2.destroyWindow(self.window)
            cv2.waitKey(1)
        if self.aborted:
            return None
        return self.point


def _timestamped_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_ROOT / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_source_image(img: np.ndarray, point: Tuple[int, int],
                       out_path: Path) -> None:
    canvas = img.copy()
    u, v = point
    cv2.drawMarker(canvas, (u, v), (0, 0, 255),
                   cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
    cv2.circle(canvas, (u, v), 8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"source ({u}, {v})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def _extract_band_match_count(note: str) -> Optional[int]:
    """Extract the kept band-match count from a transfer note string.

    Supports both note styles:
      - "band(10px)=4/94, ..."
      - "band_filter(10px): 0 matches (< 2) of 57"
    """
    if not note:
        return None
    m = re.search(r"band\([^)]*\)\s*=\s*(\d+)\s*/\s*\d+", note)
    if m:
        return int(m.group(1))
    m = re.search(r"band_filter\([^)]*\):\s*(\d+)\s+matches", note)
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------- #
# Main                                                                    #
# ---------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive pixel transfer: click in source -> transfer to all frames."
    )
    parser.add_argument("--source-index", type=int, default=0,
                        help="Index of the source frame in the loaded list (default: 0).")
    parser.add_argument("--min-inliers", type=int, default=DEFAULT_MIN_INLIERS,
                        help=f"Minimum RANSAC inliers to accept F (default: {DEFAULT_MIN_INLIERS}).")
    parser.add_argument("--regions", type=Path, default=None,
                        help="Overlay regions JSON (default: config/overlay_regions.json).")
    parser.add_argument("--method", default="superpoint", choices=SUPPORTED_PIPELINES,
                        help="Feature/matching pipeline: "
                             "'superpoint' (SuperPoint + LightGlue, default; the "
                             "primary signal for the new pipeline) or "
                             "'sift' (classical SIFT + FLANN/ratio + grid filter).")
    parser.add_argument("--ransac-method", default=DEFAULT_F_METHOD,
                        help=f"RANSAC variant for F estimation "
                             f"(default: {DEFAULT_F_METHOD}).")
    parser.add_argument("--threshold", type=float, default=DEFAULT_F_THRESHOLD,
                        help=f"RANSAC reprojection threshold in px (default: {DEFAULT_F_THRESHOLD}).")
    parser.add_argument("--confidence", type=float, default=DEFAULT_F_CONFIDENCE,
                        help=f"RANSAC confidence (default: {DEFAULT_F_CONFIDENCE}).")
    parser.add_argument("--draw-samples", action="store_true",
                        help="Overlay the band-filtered target keypoints on the target image.")
    parser.add_argument("--epipolar-band", type=float, default=DEFAULT_EPIPOLAR_BAND_PX,
                        help=f"Half-width (px) of the epipolar tolerance band used to "
                             f"filter deep matches (soft geometric filter, NOT exact-line "
                             f"rejection). Default: {DEFAULT_EPIPOLAR_BAND_PX:.0f} px.")
    parser.add_argument("--k-neighbors", type=int, default=1,
                        help=f"Number of band-filtered matches nearest the click used "
                             f"for the local-affine fit (default: 1).")
    parser.add_argument("--click", default=None,
                        help='Non-interactive click "u,v" (skip the picker). '
                             'Useful to reproduce / compare runs.')

    args = parser.parse_args()
    args.epipolar_band = max(float(args.epipolar_band), MIN_EPIPOLAR_BAND_PX)

    regions, cal = _resolve_regions(args.regions)

    if not CLEAN_FOLDER.is_dir():
        print(f"ERROR: clean frames not found at {CLEAN_FOLDER}. Run Phase 2 first.")
        return 1

    frames = load_frames(INPUT_FOLDER)
    if not frames:
        print(f"ERROR: no frames found under {INPUT_FOLDER}.")
        return 1
    if not (0 <= args.source_index < len(frames)):
        print(f"ERROR: --source-index {args.source_index} out of range "
              f"[0, {len(frames) - 1}].")
        return 1

    src_idx = args.source_index
    src_frame = frames[src_idx]
    pipeline = args.method.lower()
    min_band_matches_to_show = MIN_BAND_MATCHES_BY_SOURCE_INDEX.get(
        src_idx, MIN_BAND_MATCHES_TO_SHOW
    )
    print(f"Frames           : {len(frames)}")
    print(f"Source frame     : [{src_idx}] {src_frame.name}")
    print(f"Pipeline         : {pipeline} -> local-affine transfer (no NCC)")
    print(f"Epipolar band    : +-{args.epipolar_band:.1f} px  | "
          f"K neighbors: {args.k_neighbors}")
    print(f"RANSAC           : method={args.ransac_method}  threshold={args.threshold}px  "
          f"confidence={args.confidence}  min_inliers={args.min_inliers}")

    img_src = _read_clean_image(src_frame.name)
    if img_src is None:
        print(f"ERROR: could not read clean source image "
              f"{CLEAN_FOLDER / src_frame.name}")
        return 1

    # ------------------------------------------------------------------ #
    # Pixel selection (interactive picker, or --click "u,v" to bypass)    #
    # ------------------------------------------------------------------ #
    if args.click is not None:
        try:
            u_str, v_str = args.click.split(",")
            point = (int(round(float(u_str))), int(round(float(v_str))))
        except Exception:
            print(f"ERROR: --click expects 'u,v', got {args.click!r}")
            return 1
        h_s, w_s = img_src.shape[:2]
        if not (0 <= point[0] < w_s and 0 <= point[1] < h_s):
            print(f"ERROR: --click {point} out of image bounds ({w_s}x{h_s}).")
            return 1
        print(f"Click (non-interactive): ({point[0]}, {point[1]})")
    else:
        print("\nOpening interactive picker ...")
        picker = _Picker(img_src, window_name=f"Pick pixel in {src_frame.name}")
        point = picker.run()
        if point is None:
            print("Aborted by user; nothing saved.")
            return 2
    u, v = point
    print(f"Picked pixel     : ({u}, {v})")

    # ------------------------------------------------------------------ #
    # Feature extraction (shared across all pairs)                        #
    # ------------------------------------------------------------------ #
    feat_label = "SIFT" if pipeline == "sift" else "SuperPoint"
    print(f"\nExtracting {feat_label} features on cleaned frames ...")
    feature_sets = extract_features_for_frames(
        frames, method=pipeline, use_mask=True,
        regions=regions, calibration_size=cal, source_dir=CLEAN_FOLDER,
    )

    # ------------------------------------------------------------------ #
    # Output folder                                                        #
    # ------------------------------------------------------------------ #
    out_dir = _timestamped_output_dir()
    print(f"\nOutput folder    : {out_dir}")

    source_stem = Path(src_frame.name).stem
    source_png = out_dir / f"source_{src_idx:02d}_{source_stem}.png"
    _save_source_image(img_src, point, source_png)
    print(f"Saved source     : {source_png.name}")

    # ------------------------------------------------------------------ #
    # Per-target pipeline                                                  #
    # ------------------------------------------------------------------ #
    csv_rows: List[List] = []

    print(f"\nTransferring pixel ({u}, {v}) to {len(frames) - 1} target frame(s):")
    for tgt_idx, tgt_frame in enumerate(frames):
        if tgt_idx == src_idx:
            continue
        tgt_stem = Path(tgt_frame.name).stem
        tag = f"[{tgt_idx:02d}] {tgt_frame.name}"

        img_tgt = _read_clean_image(tgt_frame.name)
        if img_tgt is None:
            print(f"  {tag}: missing clean image, skipped.")
            csv_rows.append([tgt_idx, tgt_frame.name, False,
                             "", "", "", "", "no clean image"])
            continue

        fs_a = feature_sets[src_idx]
        fs_b = feature_sets[tgt_idx]
        if fs_a.num_keypoints < 2 or fs_b.num_keypoints < 2:
            print(f"  {tag}: too few keypoints "
                  f"(src={fs_a.num_keypoints}, tgt={fs_b.num_keypoints}), skipped.")
            csv_rows.append([tgt_idx, tgt_frame.name, False,
                             "", "", "", "", "too few keypoints"])
            continue

        mr = match_pair(
            fs_a, fs_b, idx_a=src_idx, idx_b=tgt_idx,
            method="flann", ratio=DEFAULT_RATIO, mutual=False,
            grid_filter=True,
            grid_rows=GRID_ROWS, grid_cols=GRID_COLS, grid_max_per_cell=GRID_MAX,
            pipeline=pipeline,
        )
        rr = estimate_fundamental(
            mr, method=args.ransac_method, threshold=args.threshold,
            confidence=args.confidence, min_inliers=args.min_inliers,
        )

        if not rr.f_estimated or rr.F is None:
            print(f"  {tag}: F estimation failed "
                  f"(tent={mr.num_tentative}, inl={rr.num_inliers}), skipped.")
            csv_rows.append([tgt_idx, tgt_frame.name, False,
                             "", "", "", "", "F estimation failed"])
            continue
        if rr.num_inliers < args.min_inliers:
            print(f"  {tag}: too few inliers "
                  f"({rr.num_inliers} < {args.min_inliers}), skipped.")
            csv_rows.append([tgt_idx, tgt_frame.name, False,
                             "", "", "", "", f"inliers<{args.min_inliers}"])
            continue

        match_pts_a = mr.points_a()
        match_pts_b = mr.points_b()

        result = transfer_point_local_affine(
            source_pixel=(float(u), float(v)),
            F=rr.F,
            match_pts_a=match_pts_a, match_pts_b=match_pts_b,
            source_is_a=True,
            epipolar_band_px=args.epipolar_band,
            k_neighbors=args.k_neighbors,
        )

        # Strict fallback: if local-affine fails specifically because the
        # epipolar-band gate kept fewer than 2 matches, run the original
        # NCC-on-epipolar-line transfer with fixed baseline defaults.
        is_band_failure = (
            (not result.success)
            and ("band_filter(" in (result.note or ""))
            and ("(< 2)" in (result.note or ""))
        )
        if is_band_failure:
            fallback = transfer_point(
                source_pixel=(float(u), float(v)),
                image_src=img_src,
                image_dst=img_tgt,
                F=rr.F,
                source_is_a=True,
                patch_size=DEFAULT_PATCH_SIZE,
                step=DEFAULT_STEP,
            )
            fallback.note = (
                f"fallback_ncc: {fallback.note} "
                f"(from local_affine failure: {result.note})"
            )
            result = fallback

        out_png = out_dir / f"target_{tgt_idx:02d}_{tgt_stem}.png"
        band_kept = _extract_band_match_count(result.note)
        hide_due_to_low_band = (band_kept is not None and band_kept < min_band_matches_to_show)
        if not hide_due_to_low_band:
            vis = draw_transfer(
                img_src, img_tgt, result,
                ground_truth=None,
                draw_samples=args.draw_samples,
            )
            label = f"src [{src_idx:02d}] {src_frame.name}  ->  tgt [{tgt_idx:02d}] {tgt_frame.name}"
            cv2.putText(vis, label, (10, vis.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(out_png), vis)

        pred_u, pred_v = ("", "")
        if result.predicted_pixel is not None:
            pred_u, pred_v = result.predicted_pixel
            if hide_due_to_low_band:
                print(f"  {tag}: inl={rr.num_inliers}  "
                      f"pred=({pred_u:6.1f},{pred_v:6.1f})  "
                      f"[{result.note}]  -> hidden (band<{min_band_matches_to_show})")
            else:
                print(f"  {tag}: inl={rr.num_inliers}  "
                      f"pred=({pred_u:6.1f},{pred_v:6.1f})  "
                      f"[{result.note}]  -> {out_png.name}")
        else:
            if hide_due_to_low_band:
                print(f"  {tag}: inl={rr.num_inliers}  "
                      f"transfer failed ({result.note})  -> hidden (band<{min_band_matches_to_show})")
            else:
                print(f"  {tag}: inl={rr.num_inliers}  "
                      f"transfer failed ({result.note})  -> {out_png.name}")

        csv_rows.append([
            tgt_idx, tgt_frame.name,
            bool(result.success),
            (f"{result.score:.6f}" if np.isfinite(result.score) else ""),
            (f"{pred_u:.3f}" if pred_u != "" else ""),
            (f"{pred_v:.3f}" if pred_v != "" else ""),
            rr.num_inliers,
            result.note,
        ])

    # ------------------------------------------------------------------ #
    # CSV log                                                              #
    # ------------------------------------------------------------------ #
    csv_path = out_dir / "transfer_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "frame_name", "success",
                    "score", "predicted_x", "predicted_y",
                    "ransac_inliers", "note"])
        w.writerows(csv_rows)
    print(f"\nSaved CSV        : {csv_path}")

    n_ok = sum(1 for r in csv_rows if r[2] is True)
    n_total = len(csv_rows)
    print(f"Summary          : {n_ok}/{n_total} targets transferred successfully.")
    print(f"Done.            : outputs under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
