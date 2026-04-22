"""Mask-gated local-affine point transfer (SAM2 as spatial filter).

This module is a thin additive layer on top of `src.local_transfer`. The
epipolar geometry is unchanged: the fundamental matrix F, the epipolar
line, and the soft epipolar-band filter all come from the existing
`src.local_transfer` logic.

The only new behavior is a **spatial gate** driven by SAM2 masks of the
object of interest (e.g. the van):

    1. Keep only deep matches whose source keypoint is inside `mask_src`
       AND whose target keypoint is inside `mask_dst`.
    2. Also keep only matches whose target keypoint lies within the
       epipolar band (identical to the non-masked path).
    3. Fit the local affine from the K matches nearest the click (in the
       source image) out of that gated set.
    4. Apply the affine to the clicked point -> predicted target pixel.
    5. If the predicted pixel lands outside the target mask, snap it to
       the mask pixel closest to the epipolar line (geometric fallback).

Robustness chain (degrades gracefully):
    * gated set >= MIN_BAND_MATCHES               -> masked local-affine
    * gated set < MIN_BAND_MATCHES                -> fall back to the
                                                     classical (band-only)
                                                     path from
                                                     `transfer_point_local_affine`
                                                     so behavior never gets
                                                     worse than the baseline
    * predicted pixel outside `mask_dst`          -> snap to line, then to
                                                     nearest mask pixel

Nothing else in the pipeline (feature extraction, matching, F estimation,
`draw_transfer`) is touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .transfer import (
    TransferResult,
    DEFAULT_PATCH_SIZE,
    DEFAULT_STEP,
    compute_epipolar_line,
)
from .local_transfer import (
    transfer_point_local_affine,
    DEFAULT_EPIPOLAR_BAND_PX,
    DEFAULT_K_NEIGHBORS,
    MIN_BAND_MATCHES,
    _fit_local_affine,
    _epipolar_distance,
)
from .sam2_mask import (
    filter_samples_by_mask,
    nearest_mask_pixel_to_line,
    closest_mask_point_to_xy,
)


DEFAULT_MASK_DILATE_PX = 0  # grow masks before gating (0 = off; 2-4 absorbs SAM2 boundary slop)


@dataclass
class MaskedLocalAffineDebug:
    """Diagnostic bundle returned alongside the TransferResult."""
    n_matches_total: int = 0
    n_band_matches: int = 0
    n_gated_matches: int = 0
    n_used_for_fit: int = 0
    used_fallback: bool = False
    fallback_reason: str = ""
    snapped_to_mask: bool = False
    mask_src_area_px: int = 0
    mask_dst_area_px: int = 0


# ---------------------------------------------------------------------- #
# Helpers                                                                 #
# ---------------------------------------------------------------------- #

def _dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0 or mask is None or mask.size == 0:
        return mask
    k = 2 * int(radius_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _snap_point_to_line(xy: Tuple[float, float],
                        line_abc: np.ndarray) -> Tuple[float, float]:
    """Perpendicular projection of (x, y) onto a*x + b*y + c = 0."""
    a = float(line_abc[0]); b = float(line_abc[1]); c = float(line_abc[2])
    denom = a * a + b * b
    if denom < 1e-12:
        return (float(xy[0]), float(xy[1]))
    t = (a * xy[0] + b * xy[1] + c) / denom
    return (float(xy[0] - t * a), float(xy[1] - t * b))


def _empty_result(source_pixel, line, note: str) -> TransferResult:
    return TransferResult(
        source_pixel=(float(source_pixel[0]), float(source_pixel[1])),
        epipolar_line=line,
        samples=np.zeros((0, 2), dtype=np.float64),
        scores=np.zeros((0,), dtype=np.float64),
        predicted_pixel=None,
        score=float("nan"),
        patch_size=DEFAULT_PATCH_SIZE,
        step=DEFAULT_STEP,
        source_patch_valid=False,
        success=False,
        note=note,
    )


# ---------------------------------------------------------------------- #
# Core: mask-gated local-affine                                           #
# ---------------------------------------------------------------------- #

def transfer_point_masked_local_affine(
    source_pixel: Tuple[float, float],
    F: np.ndarray,
    match_pts_a: np.ndarray,
    match_pts_b: np.ndarray,
    mask_src: Optional[np.ndarray],
    mask_dst: Optional[np.ndarray],
    *,
    source_is_a: bool = True,
    epipolar_band_px: float = DEFAULT_EPIPOLAR_BAND_PX,
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
    mask_dilate_px: int = DEFAULT_MASK_DILATE_PX,
    snap_to_mask: bool = True,
    return_debug: bool = False,
) -> "TransferResult | Tuple[TransferResult, MaskedLocalAffineDebug]":
    """Mask-gated variant of `transfer_point_local_affine`.

    Epipolar geometry is untouched. SAM2 masks act *only* as a spatial
    filter over the deep-match set and as a final sanity snap.

    Parameters
    ----------
    source_pixel, F, match_pts_a, match_pts_b, source_is_a,
    epipolar_band_px, k_neighbors
        Same semantics as `src.local_transfer.transfer_point_local_affine`.
    mask_src, mask_dst
        (H, W) bool/uint8 masks of the object in the source / target
        images respectively. Either may be None, in which case the
        corresponding side's gate is disabled. If both are None the
        function is exactly equivalent to the classical path.
    mask_dilate_px
        Optional morphological dilation applied to both masks before
        gating, to absorb small SAM2 boundary inaccuracies.
    snap_to_mask
        If True and the affine prediction lands outside `mask_dst`, the
        prediction is (a) projected onto the epipolar line, then (b)
        replaced with the mask pixel closest to that projection. This
        guarantees the returned point lies on the target object.
    return_debug
        If True, additionally returns a `MaskedLocalAffineDebug` bundle.

    Returns
    -------
    TransferResult or (TransferResult, MaskedLocalAffineDebug).

    Robustness
    ----------
    * If mask gating leaves too few matches to fit an affine, the function
      transparently falls back to the classical (band-only) path so the
      output is never worse than the pre-SAM2 baseline.
    * If either mask is None or empty, the function is equivalent to the
      classical path (no-op SAM2 gate).
    """
    if F is None or F.shape != (3, 3):
        raise ValueError("F must be a 3x3 matrix.")

    u, v = float(source_pixel[0]), float(source_pixel[1])
    line = compute_epipolar_line((u, v), F, source_is_a=source_is_a)

    pts_a = np.asarray(match_pts_a, dtype=np.float64).reshape(-1, 2)
    pts_b = np.asarray(match_pts_b, dtype=np.float64).reshape(-1, 2)
    n_total = int(pts_a.shape[0])

    dbg = MaskedLocalAffineDebug(n_matches_total=n_total)

    # Normalize masks.
    def _prep(m):
        if m is None:
            return None
        m = np.asarray(m)
        if m.size == 0:
            return None
        mb = m.astype(bool)
        if mask_dilate_px > 0:
            mb = _dilate_mask(mb, mask_dilate_px)
        return mb

    mask_src_b = _prep(mask_src)
    mask_dst_b = _prep(mask_dst)
    dbg.mask_src_area_px = int(mask_src_b.sum()) if mask_src_b is not None else 0
    dbg.mask_dst_area_px = int(mask_dst_b.sum()) if mask_dst_b is not None else 0

    masks_unavailable = (
        (mask_src_b is None or not mask_src_b.any())
        and (mask_dst_b is None or not mask_dst_b.any())
    )

    # --- Shortcut: no masks -> exact classical behaviour. ---
    if masks_unavailable:
        result = transfer_point_local_affine(
            source_pixel=(u, v), F=F,
            match_pts_a=pts_a, match_pts_b=pts_b,
            source_is_a=source_is_a,
            epipolar_band_px=epipolar_band_px,
            k_neighbors=k_neighbors,
        )
        result.note = f"masked(no-mask) -> {result.note}"
        dbg.used_fallback = True
        dbg.fallback_reason = "no usable mask; classical path"
        dbg.n_band_matches = int(result.samples.shape[0])
        return (result, dbg) if return_debug else result

    if n_total == 0 or pts_a.shape[0] != pts_b.shape[0]:
        if return_debug:
            return _empty_result((u, v), line,
                                 f"masked: no matches "
                                 f"(|A|={pts_a.shape[0]}, |B|={pts_b.shape[0]})"), dbg
        return _empty_result((u, v), line,
                             f"masked: no matches "
                             f"(|A|={pts_a.shape[0]}, |B|={pts_b.shape[0]})")

    # 1. Soft epipolar-band filter on the target-side keypoint
    #    (same convention as the classical local-affine path).
    d_epi = _epipolar_distance(pts_b, line)
    band_mask = d_epi <= float(epipolar_band_px)
    dbg.n_band_matches = int(band_mask.sum())

    # 2. Mask gates on source and target keypoints.
    if mask_src_b is not None and mask_src_b.any():
        in_src = filter_samples_by_mask(pts_a, mask_src_b)
    else:
        in_src = np.ones(pts_a.shape[0], dtype=bool)
    if mask_dst_b is not None and mask_dst_b.any():
        in_dst = filter_samples_by_mask(pts_b, mask_dst_b)
    else:
        in_dst = np.ones(pts_b.shape[0], dtype=bool)

    gated = band_mask & in_src & in_dst
    dbg.n_gated_matches = int(gated.sum())

    # 3. If mask gating is too aggressive, fall back to the classical
    #    band-only path. SAM2 then contributes *only* through the final
    #    snap (if enabled). This guarantees we never do worse.
    if dbg.n_gated_matches < MIN_BAND_MATCHES:
        dbg.used_fallback = True
        dbg.fallback_reason = (
            f"masked gate too strict "
            f"({dbg.n_gated_matches} < {MIN_BAND_MATCHES}); "
            f"falling back to band-only local-affine"
        )
        result = transfer_point_local_affine(
            source_pixel=(u, v), F=F,
            match_pts_a=pts_a, match_pts_b=pts_b,
            source_is_a=source_is_a,
            epipolar_band_px=epipolar_band_px,
            k_neighbors=k_neighbors,
        )
        result.note = f"masked -> fallback ({dbg.fallback_reason}) -> {result.note}"

        # Even on fallback, we can still snap to the mask if the mask-dst
        # is available and the baseline lands outside it.
        if (snap_to_mask and result.predicted_pixel is not None
                and mask_dst_b is not None and mask_dst_b.any()):
            pred = result.predicted_pixel
            if not _point_in_mask(pred, mask_dst_b):
                snapped = _snap_prediction_to_mask(pred, line, mask_dst_b)
                if snapped is not None:
                    result.predicted_pixel = snapped
                    result.note += f" | snapped to mask at ({snapped[0]:.1f},{snapped[1]:.1f})"
                    dbg.snapped_to_mask = True
        return (result, dbg) if return_debug else result

    # 4. Fit the local affine from the K nearest GATED matches to the click.
    gated_a = pts_a[gated]
    gated_b = pts_b[gated]
    A, k_used, fit_kind = _fit_local_affine(
        (u, v), gated_a, gated_b, k_neighbors=k_neighbors,
    )
    dbg.n_used_for_fit = int(k_used)

    if A is None:
        # Extremely unlikely (we already have >= MIN_BAND_MATCHES inside the
        # gate) but be safe and degrade to classical path.
        dbg.used_fallback = True
        dbg.fallback_reason = f"gated affine fit failed ({fit_kind})"
        result = transfer_point_local_affine(
            source_pixel=(u, v), F=F,
            match_pts_a=pts_a, match_pts_b=pts_b,
            source_is_a=source_is_a,
            epipolar_band_px=epipolar_band_px,
            k_neighbors=k_neighbors,
        )
        result.note = f"masked -> fallback ({dbg.fallback_reason}) -> {result.note}"
        return (result, dbg) if return_debug else result

    # 5. Apply the affine.
    pred_x = A[0, 0] * u + A[0, 1] * v + A[0, 2]
    pred_y = A[1, 0] * u + A[1, 1] * v + A[1, 2]
    if not (np.isfinite(pred_x) and np.isfinite(pred_y)):
        dbg.used_fallback = True
        dbg.fallback_reason = "non-finite gated prediction; classical path"
        result = transfer_point_local_affine(
            source_pixel=(u, v), F=F,
            match_pts_a=pts_a, match_pts_b=pts_b,
            source_is_a=source_is_a,
            epipolar_band_px=epipolar_band_px,
            k_neighbors=k_neighbors,
        )
        result.note = f"masked -> fallback ({dbg.fallback_reason}) -> {result.note}"
        return (result, dbg) if return_debug else result

    pred: Tuple[float, float] = (float(pred_x), float(pred_y))

    # 6. Optional snap: keep prediction inside mask_dst.
    snap_note = ""
    if (snap_to_mask and mask_dst_b is not None and mask_dst_b.any()
            and not _point_in_mask(pred, mask_dst_b)):
        snapped = _snap_prediction_to_mask(pred, line, mask_dst_b)
        if snapped is not None:
            pred = snapped
            dbg.snapped_to_mask = True
            snap_note = f" | snapped to mask at ({pred[0]:.1f},{pred[1]:.1f})"

    samples = gated_b.astype(np.float64, copy=False)
    scores = np.full((samples.shape[0],), np.nan, dtype=np.float64)

    note = (f"masked band({epipolar_band_px:.0f}px)={dbg.n_band_matches}/{n_total}"
            f", gated(src&dst mask)={dbg.n_gated_matches}"
            f", local_affine({fit_kind},K={k_used}){snap_note}")

    result = TransferResult(
        source_pixel=(u, v),
        epipolar_line=line,
        samples=samples,
        scores=scores,
        predicted_pixel=pred,
        score=float("nan"),
        patch_size=DEFAULT_PATCH_SIZE,
        step=DEFAULT_STEP,
        source_patch_valid=True,
        success=True,
        note=note,
    )
    return (result, dbg) if return_debug else result


# ---------------------------------------------------------------------- #
# Snap helpers                                                            #
# ---------------------------------------------------------------------- #

def _point_in_mask(xy: Tuple[float, float], mask: np.ndarray) -> bool:
    h, w = mask.shape[:2]
    x = int(round(float(xy[0])))
    y = int(round(float(xy[1])))
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    return bool(mask[y, x])


def _snap_prediction_to_mask(
    pred: Tuple[float, float],
    line_abc: np.ndarray,
    mask: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """Try to pull `pred` inside `mask` using line/mask geometry.

    Order of attempts (first that lands inside `mask` wins):
        1. Perpendicular projection of `pred` onto the epipolar line.
        2. Mask pixel closest to the projected point.
        3. Mask pixel closest to the epipolar line globally.
    """
    proj = _snap_point_to_line(pred, line_abc)
    if _point_in_mask(proj, mask):
        return proj

    near_proj = closest_mask_point_to_xy(mask, proj)
    if near_proj is not None:
        return near_proj

    return nearest_mask_pixel_to_line(line_abc, mask)


# ---------------------------------------------------------------------- #
# Target-side SAM2 seed (to prompt SAM2 in the target frame)              #
# ---------------------------------------------------------------------- #

def pick_target_seed(
    source_pixel: Tuple[float, float],
    F: np.ndarray,
    match_pts_a: np.ndarray,
    match_pts_b: np.ndarray,
    *,
    source_is_a: bool = True,
    epipolar_band_px: float = DEFAULT_EPIPOLAR_BAND_PX,
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
    image_shape_dst: Optional[Tuple[int, int]] = None,
) -> Tuple[Optional[Tuple[float, float]], str]:
    """Pick a seed xy inside the target object for SAM2 prompting.

    Strategy: run the existing, un-masked `transfer_point_local_affine`
    and use its prediction as the SAM2 positive point prompt in the
    target frame. This is the single strongest signal we have before
    SAM2 sees the image, and it is already what the baseline uses, so
    SAM2 can only *improve* over it.

    Returns (xy, note). `xy` is None if no reliable seed could be
    produced (caller should skip SAM2 for that target in that case).
    """
    baseline = transfer_point_local_affine(
        source_pixel=source_pixel, F=F,
        match_pts_a=match_pts_a, match_pts_b=match_pts_b,
        source_is_a=source_is_a,
        epipolar_band_px=epipolar_band_px,
        k_neighbors=k_neighbors,
    )
    if baseline.predicted_pixel is not None:
        if image_shape_dst is not None:
            h, w = image_shape_dst[:2]
            x, y = baseline.predicted_pixel
            if not (0 <= x < w and 0 <= y < h):
                return None, f"seed outside image: {baseline.note}"
        return baseline.predicted_pixel, f"seed=local_affine ({baseline.note})"
    return None, f"seed unavailable ({baseline.note})"
