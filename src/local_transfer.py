"""Deep-match-driven point transfer with a local-affine prior.

Pipeline
--------
    click  ->  SuperPoint + deep matches (e.g. LightGlue / SuperGlue)
           ->  filter matches using an epipolar *band* (soft, +/- tolerance)
           ->  select the K matches whose source keypoint is closest to the click
           ->  fit a local affine A from those corresponding pairs
           ->  final point = A * (u, v, 1)   (in the target image)

Important properties
--------------------
* The epipolar line is used as a **soft geometric filter**, not a hard
  on-line rejection: matches whose target keypoint lies within
  `epipolar_band_px` pixels of the line are kept.
* No NCC is used anywhere in this pipeline.
* No sampling / scanning along the epipolar line.
* The source of geometric signal is the deep-match set passed in by the
  caller (typically SuperPoint+LightGlue tentative matches); F is only
  needed to define the epipolar band.

Return type is `TransferResult` (from `src.transfer`) so the existing
`draw_transfer` visualization keeps working unchanged. NCC-specific
fields (`scores`, `patch_size`, `step`) are populated with neutral
sentinels since they no longer apply.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .transfer import (
    TransferResult,
    DEFAULT_PATCH_SIZE,
    DEFAULT_STEP,
    compute_epipolar_line,
)


DEFAULT_EPIPOLAR_BAND_PX = 20.0
DEFAULT_K_NEIGHBORS = 8
DEFAULT_AFFINE_RANSAC_PX = 2.0
MIN_BAND_MATCHES = 2  # need >= 2 correspondences to fit any affine
THIRD_POINT_RELATIVE_FAR_RATIO = 2.5
THIRD_POINT_ABSOLUTE_FAR_PX = 50.0


# ---------------------------------------------------------------------- #
# Helpers                                                                 #
# ---------------------------------------------------------------------- #

def _epipolar_distance(points_xy: np.ndarray, line_abc: np.ndarray) -> np.ndarray:
    """Signed-magnitude perpendicular distance from each (x, y) to a*x + b*y + c = 0."""
    a = float(line_abc[0]); b = float(line_abc[1]); c = float(line_abc[2])
    denom = float(np.sqrt(a * a + b * b))
    if denom < 1e-12:
        return np.full(points_xy.shape[0], np.inf, dtype=np.float64)
    return np.abs(a * points_xy[:, 0] + b * points_xy[:, 1] + c) / denom


def _fit_local_affine(
    click_xy: Tuple[float, float],
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
    ransac_px: float = DEFAULT_AFFINE_RANSAC_PX,
) -> Tuple[Optional[np.ndarray], int, str]:
    """Fit a 2x3 affine mapping using the K nearest matches to the click.

    Falls through a chain by available K:
        K >= 3 -> cv2.estimateAffine2D         (6-DoF affine, RANSAC)
        K >= 2 -> cv2.estimateAffinePartial2D  (similarity,  RANSAC)
        K == 1 -> translation-only from the single nearest correspondence
        K == 0 -> (None, 0, "no matches")
    """
    if pts_a is None or len(pts_a) == 0:
        return None, 0, "no matches"
    pts_a = np.asarray(pts_a, dtype=np.float32).reshape(-1, 2)
    pts_b = np.asarray(pts_b, dtype=np.float32).reshape(-1, 2)

    d = np.linalg.norm(pts_a - np.asarray(click_xy, dtype=np.float32), axis=1)
    order = np.argsort(d)
    k_eff = int(min(k_neighbors, len(pts_a)))

    # Adaptive K for the common k=3 case:
    # if the 3rd nearest source match is clearly far from the first two, use
    # only the first two correspondences (more stable local similarity fit).
    if k_eff >= 3 and int(k_neighbors) == 3:
        i0, i1, i2 = int(order[0]), int(order[1]), int(order[2])
        p0, p1, p2 = pts_a[i0], pts_a[i1], pts_a[i2]
        base = float(np.linalg.norm(p0 - p1))
        third_to_pair = float(min(np.linalg.norm(p2 - p0), np.linalg.norm(p2 - p1)))
        rel_gate = THIRD_POINT_RELATIVE_FAR_RATIO * max(base, 1.0)
        if third_to_pair > rel_gate or third_to_pair > THIRD_POINT_ABSOLUTE_FAR_PX:
            k_eff = 2

    idx = order[:k_eff]
    pa, pb = pts_a[idx], pts_b[idx]

    if k_eff >= 3:
        A, _ = cv2.estimateAffine2D(
            pa, pb, method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_px),
        )
        if A is not None and np.all(np.isfinite(A)):
            return A.astype(np.float64), k_eff, "affine"

    if k_eff >= 2:
        A, _ = cv2.estimateAffinePartial2D(
            pa, pb, method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_px),
        )
        if A is not None and np.all(np.isfinite(A)):
            return A.astype(np.float64), k_eff, "similarity"

    if k_eff >= 1:
        delta = pb[0] - pa[0]
        A = np.array(
            [[1.0, 0.0, float(delta[0])], [0.0, 1.0, float(delta[1])]],
            dtype=np.float64,
        )
        return A, 1, "translation"

    return None, 0, "degenerate"


def _empty_result(
    source_pixel: Tuple[float, float],
    line: np.ndarray,
    note: str,
) -> TransferResult:
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
# Core                                                                     #
# ---------------------------------------------------------------------- #

def transfer_point_local_affine(
    source_pixel: Tuple[float, float],
    F: np.ndarray,
    match_pts_a: np.ndarray,
    match_pts_b: np.ndarray,
    source_is_a: bool = True,
    epipolar_band_px: float = DEFAULT_EPIPOLAR_BAND_PX,
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
) -> TransferResult:
    """Transfer a clicked source pixel to the target image via a local affine.

    Parameters
    ----------
    source_pixel : (u, v) in the source image.
    F : 3x3 fundamental matrix consistent with `x_b^T F x_a = 0` when
        `source_is_a=True`. Used only to define the epipolar band filter.
    match_pts_a, match_pts_b : (N, 2) float arrays of corresponding
        deep-match keypoints in the source / target images respectively.
        Typically SuperPoint+LightGlue/SuperGlue tentative matches; RANSAC
        inliers work equally well.
    source_is_a : mirrors the original `transfer_point`.
    epipolar_band_px : half-width of the tolerance band around the
        epipolar line in the target image. Matches whose target keypoint
        lies within this distance of the line are kept. Soft filter, not
        strict on-line rejection.
    k_neighbors : number of nearest (in the source image) band-filtered
        matches used for the affine fit.

    Returns
    -------
    TransferResult (shape-compatible with `src.transfer.transfer_point`).
    `samples` holds the kept target-side matches (useful for debugging /
    drawing); `scores` is filled with NaN since no NCC is computed.
    """
    if F is None or F.shape != (3, 3):
        raise ValueError("F must be a 3x3 matrix.")

    u, v = float(source_pixel[0]), float(source_pixel[1])
    line = compute_epipolar_line((u, v), F, source_is_a=source_is_a)

    pts_a = np.asarray(match_pts_a, dtype=np.float64).reshape(-1, 2)
    pts_b = np.asarray(match_pts_b, dtype=np.float64).reshape(-1, 2)
    if pts_a.shape[0] == 0 or pts_a.shape[0] != pts_b.shape[0]:
        return _empty_result(
            (u, v), line,
            note=f"no matches (|A|={pts_a.shape[0]}, |B|={pts_b.shape[0]})",
        )

    # 1. Soft epipolar-band filter on the target-side keypoint.
    d_epi = _epipolar_distance(pts_b, line)
    band_mask = d_epi <= float(epipolar_band_px)
    n_band = int(band_mask.sum())
    if n_band < MIN_BAND_MATCHES:
        return _empty_result(
            (u, v), line,
            note=(f"band_filter({epipolar_band_px:.0f}px): "
                  f"{n_band} matches (< {MIN_BAND_MATCHES}) of {pts_a.shape[0]}"),
        )
    band_a = pts_a[band_mask]
    band_b = pts_b[band_mask]

    # 2. Fit local affine from the K nearest band-filtered matches to the click.
    A, k_used, fit_kind = _fit_local_affine(
        (u, v), band_a, band_b, k_neighbors=k_neighbors,
    )
    if A is None:
        return _empty_result(
            (u, v), line,
            note=(f"band_filter({epipolar_band_px:.0f}px)={n_band}, "
                  f"local_affine: {fit_kind}"),
        )

    # 3. Apply affine to the clicked point -> predicted target pixel.
    pred_x = A[0, 0] * u + A[0, 1] * v + A[0, 2]
    pred_y = A[1, 0] * u + A[1, 1] * v + A[1, 2]
    if not (np.isfinite(pred_x) and np.isfinite(pred_y)):
        return _empty_result(
            (u, v), line,
            note=f"local_affine({fit_kind},K={k_used}): non-finite prediction",
        )

    # Samples = the band-filtered target-side matches (for visualization /
    # debugging). No NCC scores -> NaN array of the same length.
    samples = band_b.astype(np.float64, copy=False)
    scores = np.full((samples.shape[0],), np.nan, dtype=np.float64)

    note = (f"band({epipolar_band_px:.0f}px)={n_band}/{pts_a.shape[0]}, "
            f"local_affine({fit_kind},K={k_used})")

    return TransferResult(
        source_pixel=(u, v),
        epipolar_line=line,
        samples=samples,
        scores=scores,
        predicted_pixel=(float(pred_x), float(pred_y)),
        score=float("nan"),
        patch_size=DEFAULT_PATCH_SIZE,
        step=DEFAULT_STEP,
        source_patch_valid=True,
        success=True,
        note=note,
    )
