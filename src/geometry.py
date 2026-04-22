"""Phase 5 geometric filtering (Milestone 2 Step 3).

RANSAC-based fundamental-matrix estimation on top of tentative matches
produced by `src.matching`. This module produces `RansacResult` objects
that carry the inlier subset, the estimated F, and the sign convention
required for the next milestone (epipolar-guided point transfer).

Sign convention (OpenCV):
    For a match (x_a, x_b) we have  x_b^T F x_a = 0.
    `cv2.computeCorrespondEpilines(pts, whichImage=1, F)` returns the
    epipolar line in image B (frame b) for a point in image A (frame a).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .features import FeatureSet
from .matching import FrameMatchResult


# OpenCV RANSAC method constants (USAC_MAGSAC available since OpenCV 4.5)
SUPPORTED_F_METHODS = {
    "ransac": cv2.FM_RANSAC,
    "lmeds": cv2.FM_LMEDS,
    "usac_magsac": getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC),
    "usac_default": getattr(cv2, "USAC_DEFAULT", cv2.FM_RANSAC),
}

DEFAULT_F_METHOD = "ransac"
DEFAULT_F_THRESHOLD = 1.0       # reprojection error in pixels
DEFAULT_F_CONFIDENCE = 0.999
DEFAULT_MIN_INLIERS = 15        # practical robustness floor for F


@dataclass
class RansacResult:
    """Outcome of RANSAC-based fundamental-matrix estimation for a pair.

    `F` is consistent with `(points_a, points_b)` such that
    `points_b.T @ F @ points_a = 0` up to noise. Compute epipolar lines in
    image B for points in image A with `whichImage=1`.

    `inlier_matches` holds the cv2.DMatch objects that survived RANSAC,
    indexed against the same FeatureSets that the underlying
    FrameMatchResult used (i.e. `match_result.fs_a_filtered` /
    `.fs_b_filtered`).
    """
    idx_a: int
    idx_b: int
    name_a: str
    name_b: str
    num_tentative: int
    num_inliers: int
    F: Optional[np.ndarray]
    inlier_mask: Optional[np.ndarray]          # shape (num_tentative,) bool
    inlier_matches: List[cv2.DMatch] = field(default_factory=list)
    method: str = DEFAULT_F_METHOD
    threshold: float = DEFAULT_F_THRESHOLD
    confidence: float = DEFAULT_F_CONFIDENCE
    min_inliers: int = DEFAULT_MIN_INLIERS
    f_estimated: bool = False                  # did findFundamentalMat return a valid F
    success: bool = False                      # F estimated AND num_inliers >= min_inliers

    @property
    def inlier_ratio(self) -> float:
        if self.num_tentative == 0:
            return 0.0
        return self.num_inliers / self.num_tentative

    def inlier_points_a(self, fs_a: FeatureSet) -> np.ndarray:
        """(M, 2) float32 inlier coordinates in frame A."""
        return np.float32([fs_a.keypoints[m.queryIdx].pt
                           for m in self.inlier_matches]).reshape(-1, 2)

    def inlier_points_b(self, fs_b: FeatureSet) -> np.ndarray:
        """(M, 2) float32 inlier coordinates in frame B."""
        return np.float32([fs_b.keypoints[m.trainIdx].pt
                           for m in self.inlier_matches]).reshape(-1, 2)


def estimate_fundamental(
    match_result: FrameMatchResult,
    method: str = DEFAULT_F_METHOD,
    threshold: float = DEFAULT_F_THRESHOLD,
    confidence: float = DEFAULT_F_CONFIDENCE,
    min_inliers: int = DEFAULT_MIN_INLIERS,
) -> RansacResult:
    """Estimate F from a FrameMatchResult's tentative matches.

    The match indices in `match_result.tentative_matches` reference the
    FeatureSets stored inside the result (`fs_a_filtered` / `fs_b_filtered`),
    so no external FeatureSet is needed here — we just pull 2D coordinates
    via `match_result.points_a()` / `points_b()`.
    """
    if method.lower() not in SUPPORTED_F_METHODS:
        raise ValueError(
            f"Unsupported method {method!r}; expected one of "
            f"{sorted(SUPPORTED_F_METHODS)}."
        )
    m = method.lower()
    cv_method = SUPPORTED_F_METHODS[m]

    pts_a = match_result.points_a()
    pts_b = match_result.points_b()
    n_tent = len(pts_a)

    base_result = RansacResult(
        idx_a=match_result.idx_a, idx_b=match_result.idx_b,
        name_a=match_result.name_a, name_b=match_result.name_b,
        num_tentative=n_tent, num_inliers=0,
        F=None, inlier_mask=None, inlier_matches=[],
        method=m, threshold=threshold, confidence=confidence,
        min_inliers=min_inliers,
        f_estimated=False, success=False,
    )

    # 8-point algorithm minimum (7 for FM_7POINT; OpenCV uses 8 internally for RANSAC).
    if n_tent < 8:
        return base_result

    F, mask = cv2.findFundamentalMat(
        pts_a, pts_b,
        method=cv_method,
        ransacReprojThreshold=threshold,
        confidence=confidence,
    )

    if F is None or mask is None:
        return base_result

    # OpenCV sometimes returns a 9x3 matrix (multiple candidates from FM_7POINT).
    # For RANSAC / USAC we expect 3x3.
    if F.shape != (3, 3):
        return base_result

    inlier_mask = mask.ravel().astype(bool)
    inlier_matches = [m for m, keep in zip(match_result.tentative_matches, inlier_mask) if keep]
    n_inl = int(inlier_mask.sum())

    return RansacResult(
        idx_a=match_result.idx_a, idx_b=match_result.idx_b,
        name_a=match_result.name_a, name_b=match_result.name_b,
        num_tentative=n_tent, num_inliers=n_inl,
        F=F, inlier_mask=inlier_mask, inlier_matches=inlier_matches,
        method=m, threshold=threshold, confidence=confidence,
        min_inliers=min_inliers,
        f_estimated=True,
        success=(n_inl >= min_inliers),
    )


def estimate_fundamental_for_matches(
    match_results: Sequence[FrameMatchResult],
    method: str = DEFAULT_F_METHOD,
    threshold: float = DEFAULT_F_THRESHOLD,
    confidence: float = DEFAULT_F_CONFIDENCE,
    min_inliers: int = DEFAULT_MIN_INLIERS,
    progress: bool = False,
) -> List[RansacResult]:
    """Apply `estimate_fundamental()` to a batch of match results."""
    out: List[RansacResult] = []
    for i, mr in enumerate(match_results):
        rr = estimate_fundamental(
            mr, method=method, threshold=threshold,
            confidence=confidence, min_inliers=min_inliers,
        )
        out.append(rr)
        if progress:
            tag = "OK " if rr.success else ("F  " if rr.f_estimated else "FAIL")
            print(f"  [{i + 1:3d}/{len(match_results)}] {tag} "
                  f"({rr.idx_a},{rr.idx_b})  "
                  f"tent={rr.num_tentative:4d}  "
                  f"inl={rr.num_inliers:4d}  "
                  f"({100 * rr.inlier_ratio:5.1f}%)")
    return out


# ---------------------------------------------------------------------- #
# Visualization                                                          #
# ---------------------------------------------------------------------- #

def draw_inlier_matches(
    img_a: np.ndarray,
    img_b: np.ndarray,
    match_result: FrameMatchResult,
    ransac_result: RansacResult,
    max_draw: int = 80,
) -> np.ndarray:
    """Draw only the RANSAC-inlier subset of a FrameMatchResult."""
    matches = ransac_result.inlier_matches
    if len(matches) > max_draw:
        idx = np.linspace(0, len(matches) - 1, max_draw, dtype=int)
        matches = [matches[i] for i in idx]
    kps_a = match_result.fs_a_filtered.keypoints
    kps_b = match_result.fs_b_filtered.keypoints
    return cv2.drawMatches(
        img_a, kps_a, img_b, kps_b,
        matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def _draw_line_through_image(img: np.ndarray, line: np.ndarray, color) -> None:
    """Clip the infinite epipolar line a x + b y + c = 0 to the image and draw."""
    a, b, c = float(line[0]), float(line[1]), float(line[2])
    h, w = img.shape[:2]
    # Endpoints at x=0 and x=w (prefer horizontal endpoints unless b == 0)
    if abs(b) > 1e-6:
        x0, x1 = 0, w - 1
        y0 = int(round(-(a * x0 + c) / b))
        y1 = int(round(-(a * x1 + c) / b))
        cv2.line(img, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
    elif abs(a) > 1e-6:
        y0, y1 = 0, h - 1
        x0 = int(round(-(b * y0 + c) / a))
        x1 = int(round(-(b * y1 + c) / a))
        cv2.line(img, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)


def draw_epipolar_lines(
    img_a: np.ndarray,
    img_b: np.ndarray,
    match_result: FrameMatchResult,
    ransac_result: RansacResult,
    num_samples: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (img_a_with_lines, img_b_with_lines) for a few sampled inliers.

    Each sampled inlier in A is shown as a coloured circle; the matching
    epipolar line (of that point) is drawn in the same colour on B.
    And vice versa for points in B -> lines in A.
    """
    if not ransac_result.success or ransac_result.F is None:
        return img_a.copy(), img_b.copy()

    pts_a = ransac_result.inlier_points_a(match_result.fs_a_filtered)
    pts_b = ransac_result.inlier_points_b(match_result.fs_b_filtered)
    n = len(pts_a)
    if n == 0:
        return img_a.copy(), img_b.copy()

    k = min(num_samples, n)
    sel = np.linspace(0, n - 1, k, dtype=int)
    pa = pts_a[sel].reshape(-1, 1, 2).astype(np.float32)
    pb = pts_b[sel].reshape(-1, 1, 2).astype(np.float32)

    lines_in_b = cv2.computeCorrespondEpilines(pa, 1, ransac_result.F).reshape(-1, 3)
    lines_in_a = cv2.computeCorrespondEpilines(pb, 2, ransac_result.F).reshape(-1, 3)

    out_a = img_a.copy()
    out_b = img_b.copy()
    rng = np.random.default_rng(42)
    for i in range(k):
        color = tuple(int(x) for x in rng.integers(64, 256, size=3))
        pa_i = tuple(int(x) for x in pa[i, 0])
        pb_i = tuple(int(x) for x in pb[i, 0])
        _draw_line_through_image(out_a, lines_in_a[i], color)
        _draw_line_through_image(out_b, lines_in_b[i], color)
        cv2.circle(out_a, pa_i, 5, color, 2, cv2.LINE_AA)
        cv2.circle(out_b, pb_i, 5, color, 2, cv2.LINE_AA)

    return out_a, out_b


def is_near_degenerate(
    ransac_result: RansacResult,
    inlier_pts_a: np.ndarray,
    image_shape: Tuple[int, int],
    min_inliers: int = 15,
    min_y_std_norm: float = 0.04,
    min_x_std_norm: float = 0.04,
) -> Tuple[bool, str]:
    """Heuristically flag near-degenerate pairs.

    Degeneracy indicators (any one flags the pair):
      * F estimation failed.
      * Too few inliers.
      * Inliers concentrated on a near-horizontal line (low Y spread).
      * Inliers concentrated on a near-vertical line (low X spread).

    `inlier_pts_a` is the (M, 2) array of inlier coordinates in frame A
    (obtained via `ransac_result.inlier_points_a(fs_a)`).
    """
    if not ransac_result.f_estimated:
        return True, "F estimation failed"
    if ransac_result.num_inliers < min_inliers:
        return True, f"too few inliers ({ransac_result.num_inliers} < {min_inliers})"

    if inlier_pts_a.size == 0:
        return True, "no inlier points"

    h, w = image_shape
    y_std = float(np.std(inlier_pts_a[:, 1]) / h)
    x_std = float(np.std(inlier_pts_a[:, 0]) / w)
    if y_std < min_y_std_norm:
        return True, f"inliers on near-horizontal line (y_std_norm={y_std:.3f})"
    if x_std < min_x_std_norm:
        return True, f"inliers on near-vertical line (x_std_norm={x_std:.3f})"
    return False, "ok"
