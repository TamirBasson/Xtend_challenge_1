"""Phase 6 / Milestone 3: epipolar-guided point transfer.

Given a source pixel in frame A and a fundamental matrix F consistent with
`x_b^T F x_a = 0`, predict the matching pixel in frame B by:

    1. computing the epipolar line in B (`cv2.computeCorrespondEpilines`)
    2. sampling candidate points along that line (clipped to the image)
    3. scoring each candidate by patch similarity (NCC)
    4. returning the best candidate + its score

The module is intentionally minimal:
  * grayscale patches, fixed patch size
  * 1-pixel sampling step by default
  * NCC implemented directly (no OpenCV template matching)
  * no multi-frame fusion, no sub-pixel refinement (deferred to later phases)

The direction is configurable via `source_is_a`:
  * True  -> source point is in frame A, line is computed in frame B (whichImage=1)
  * False -> source point is in frame B, line is computed in frame A (whichImage=2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np


DEFAULT_PATCH_SIZE = 21
DEFAULT_STEP = 1.0


@dataclass
class TransferResult:
    """Outcome of a single point-transfer query.

    Attributes
    ----------
    source_pixel : (u, v) float coordinates queried in the source image.
    epipolar_line : (3,) float array [a, b, c] with a*x + b*y + c = 0 in the target image.
    samples : (K, 2) float array of candidate (x, y) points along the line.
    scores : (K,) float array of NCC scores for each sample in [-1, 1].
               NaN entries mark samples whose target patch was out of bounds.
    predicted_pixel : best-scoring (u, v) in the target image, or None if none valid.
    score : NCC score at `predicted_pixel`, or NaN.
    patch_size : size of the (square) patch used for matching.
    step : sampling step along the epipolar line (pixels).
    source_patch_valid : whether the source patch was extractable (in bounds & variance > 0).
    success : True iff source patch was valid AND at least one target sample was scored.
    """
    source_pixel: Tuple[float, float]
    epipolar_line: np.ndarray
    samples: np.ndarray
    scores: np.ndarray
    predicted_pixel: Optional[Tuple[float, float]] = None
    score: float = float("nan")
    patch_size: int = DEFAULT_PATCH_SIZE
    step: float = DEFAULT_STEP
    source_patch_valid: bool = False
    success: bool = False
    note: str = ""

    @property
    def num_samples(self) -> int:
        return int(self.samples.shape[0]) if self.samples.size else 0

    @property
    def num_scored(self) -> int:
        if self.scores.size == 0:
            return 0
        return int(np.isfinite(self.scores).sum())


# ---------------------------------------------------------------------- #
# Low-level helpers                                                       #
# ---------------------------------------------------------------------- #

def compute_epipolar_line(
    source_pixel: Tuple[float, float],
    F: np.ndarray,
    source_is_a: bool = True,
) -> np.ndarray:
    """Return the epipolar line [a, b, c] in the target image for `source_pixel`.

    `source_is_a=True` uses whichImage=1 (source in A, line in B) which matches
    the sign convention produced by `src.geometry.estimate_fundamental` where
    `x_b^T F x_a = 0`. Flip to False to query in the reverse direction.
    """
    pt = np.asarray(source_pixel, dtype=np.float32).reshape(1, 1, 2)
    which = 1 if source_is_a else 2
    line = cv2.computeCorrespondEpilines(pt, which, F).reshape(3)
    return line.astype(np.float64)


def sample_line_in_image(
    line: np.ndarray,
    image_shape: Tuple[int, int],
    step: float = DEFAULT_STEP,
    margin: int = 0,
) -> np.ndarray:
    """Sample points along `a*x + b*y + c = 0`, clipped to the image (with margin).

    The line is parameterized along the axis with the larger coefficient
    to avoid numerical issues near horizontal/vertical lines.

    Parameters
    ----------
    line : (3,) [a, b, c].
    image_shape : (H, W) of the target image.
    step : sampling step in pixels along the chosen axis.
    margin : pixels to keep away from the border (e.g. patch_size // 2) so
             that downstream patch extraction stays in bounds.

    Returns
    -------
    (K, 2) float64 array of (x, y) points, possibly empty.
    """
    h, w = image_shape[:2]
    a, b, c = float(line[0]), float(line[1]), float(line[2])
    step = max(float(step), 1e-3)

    x_lo = float(margin)
    x_hi = float(w - 1 - margin)
    y_lo = float(margin)
    y_hi = float(h - 1 - margin)
    if x_hi < x_lo or y_hi < y_lo:
        return np.zeros((0, 2), dtype=np.float64)

    # Parameterize along whichever axis varies the most (i.e. the smaller
    # coefficient on that axis in the implicit equation).
    if abs(b) >= abs(a):
        xs = np.arange(x_lo, x_hi + 0.5, step, dtype=np.float64)
        if abs(b) < 1e-12:
            return np.zeros((0, 2), dtype=np.float64)
        ys = -(a * xs + c) / b
        valid = (ys >= y_lo) & (ys <= y_hi)
        xs, ys = xs[valid], ys[valid]
    else:
        ys = np.arange(y_lo, y_hi + 0.5, step, dtype=np.float64)
        if abs(a) < 1e-12:
            return np.zeros((0, 2), dtype=np.float64)
        xs = -(b * ys + c) / a
        valid = (xs >= x_lo) & (xs <= x_hi)
        xs, ys = xs[valid], ys[valid]

    return np.stack([xs, ys], axis=1)


def _extract_patch(
    gray: np.ndarray,
    center: Tuple[float, float],
    patch_size: int,
) -> Optional[np.ndarray]:
    """Extract a (patch_size, patch_size) patch centred at `center`.

    Returns None if the patch would fall outside the image. Uses nearest-pixel
    rounding (simple, sufficient for a baseline).
    """
    if patch_size < 3 or patch_size % 2 == 0:
        raise ValueError(f"patch_size must be an odd integer >= 3, got {patch_size}")
    h, w = gray.shape[:2]
    half = patch_size // 2
    cx = int(round(center[0]))
    cy = int(round(center[1]))
    if cx - half < 0 or cy - half < 0:
        return None
    if cx + half >= w or cy + half >= h:
        return None
    return gray[cy - half: cy + half + 1, cx - half: cx + half + 1]


def ncc(patch_a: np.ndarray, patch_b: np.ndarray) -> float:
    """Normalized cross-correlation of two equally-sized grayscale patches.

    Returns a value in [-1, 1]; returns NaN if either patch has zero variance.
    """
    if patch_a.shape != patch_b.shape:
        raise ValueError(f"Patch shape mismatch: {patch_a.shape} vs {patch_b.shape}")
    a = patch_a.astype(np.float32).ravel()
    b = patch_b.astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------- #
# Core: epipolar-guided transfer                                         #
# ---------------------------------------------------------------------- #

def transfer_point(
    source_pixel: Tuple[float, float],
    image_src: np.ndarray,
    image_dst: np.ndarray,
    F: np.ndarray,
    source_is_a: bool = True,
    patch_size: int = DEFAULT_PATCH_SIZE,
    step: float = DEFAULT_STEP,
) -> TransferResult:
    """Transfer `source_pixel` from the source image to the target image via F.

    Parameters
    ----------
    source_pixel : (u, v) in the source image (float ok).
    image_src, image_dst : BGR or grayscale uint8 images.
    F : 3x3 fundamental matrix consistent with `x_b^T F x_a = 0` for
        `source_is_a=True`. For the reverse direction, set `source_is_a=False`.
    patch_size : odd integer, size of the square patch used for NCC.
    step : sampling step along the epipolar line, in pixels.

    Returns
    -------
    TransferResult
    """
    if F is None or F.shape != (3, 3):
        raise ValueError("F must be a 3x3 matrix.")
    if patch_size < 3 or patch_size % 2 == 0:
        raise ValueError(f"patch_size must be an odd integer >= 3, got {patch_size}")

    gray_src = _to_gray(image_src)
    gray_dst = _to_gray(image_dst)

    line = compute_epipolar_line(source_pixel, F, source_is_a=source_is_a)

    src_patch = _extract_patch(gray_src, source_pixel, patch_size)
    source_ok = src_patch is not None and float(src_patch.std()) > 1e-6

    samples = sample_line_in_image(
        line, gray_dst.shape, step=step, margin=patch_size // 2,
    )
    scores = np.full((samples.shape[0],), np.nan, dtype=np.float64)

    if not source_ok:
        note = ("source patch out of bounds"
                if src_patch is None
                else "source patch has zero variance")
        return TransferResult(
            source_pixel=(float(source_pixel[0]), float(source_pixel[1])),
            epipolar_line=line,
            samples=samples,
            scores=scores,
            predicted_pixel=None,
            score=float("nan"),
            patch_size=patch_size,
            step=step,
            source_patch_valid=False,
            success=False,
            note=note,
        )

    # Vectorized NCC along the line (loop is fine for baseline simplicity).
    # Precompute normalized source vector once.
    src_vec = src_patch.astype(np.float32).ravel()
    src_vec = src_vec - src_vec.mean()
    src_norm = float(np.linalg.norm(src_vec))

    for k, (cx, cy) in enumerate(samples):
        dst_patch = _extract_patch(gray_dst, (cx, cy), patch_size)
        if dst_patch is None:
            continue
        dv = dst_patch.astype(np.float32).ravel()
        dv -= dv.mean()
        dn = float(np.linalg.norm(dv))
        if dn < 1e-6 or src_norm < 1e-6:
            continue
        scores[k] = float(np.dot(src_vec, dv) / (src_norm * dn))

    if not np.isfinite(scores).any():
        return TransferResult(
            source_pixel=(float(source_pixel[0]), float(source_pixel[1])),
            epipolar_line=line,
            samples=samples,
            scores=scores,
            predicted_pixel=None,
            score=float("nan"),
            patch_size=patch_size,
            step=step,
            source_patch_valid=True,
            success=False,
            note="no valid target samples",
        )

    best = int(np.nanargmax(scores))
    cx, cy = samples[best]
    return TransferResult(
        source_pixel=(float(source_pixel[0]), float(source_pixel[1])),
        epipolar_line=line,
        samples=samples,
        scores=scores,
        predicted_pixel=(float(cx), float(cy)),
        score=float(scores[best]),
        patch_size=patch_size,
        step=step,
        source_patch_valid=True,
        success=True,
        note="ok",
    )


# ---------------------------------------------------------------------- #
# Visualization                                                           #
# ---------------------------------------------------------------------- #

def _draw_line_in_image(img: np.ndarray, line: np.ndarray, color, thickness: int = 1) -> None:
    a, b, c = float(line[0]), float(line[1]), float(line[2])
    h, w = img.shape[:2]
    if abs(b) > 1e-6:
        x0, x1 = 0, w - 1
        y0 = int(round(-(a * x0 + c) / b))
        y1 = int(round(-(a * x1 + c) / b))
        cv2.line(img, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)
    elif abs(a) > 1e-6:
        y0, y1 = 0, h - 1
        x0 = int(round(-(b * y0 + c) / a))
        x1 = int(round(-(b * y1 + c) / a))
        cv2.line(img, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)


def draw_transfer(
    image_src: np.ndarray,
    image_dst: np.ndarray,
    result: TransferResult,
    ground_truth: Optional[Tuple[float, float]] = None,
    draw_samples: bool = False,
) -> np.ndarray:
    """Side-by-side visualization for a single transfer query.

    Left  : source image with the source pixel marked (red).
    Right : target image with the epipolar line (yellow), sampled candidates
            (optional, light blue dots), predicted point (green), and the
            optional ground-truth point (magenta).
    """
    if image_src.ndim == 2:
        img_a = cv2.cvtColor(image_src, cv2.COLOR_GRAY2BGR)
    else:
        img_a = image_src.copy()
    if image_dst.ndim == 2:
        img_b = cv2.cvtColor(image_dst, cv2.COLOR_GRAY2BGR)
    else:
        img_b = image_dst.copy()

    COLOR_SRC   = (0, 0, 255)       # red
    COLOR_LINE  = (0, 255, 255)     # yellow
    COLOR_PRED  = (0, 255, 0)       # green
    COLOR_GT    = (255, 0, 255)     # magenta
    COLOR_SAMPL = (255, 200, 0)     # light blue (BGR)

    su, sv = int(round(result.source_pixel[0])), int(round(result.source_pixel[1]))
    cv2.drawMarker(img_a, (su, sv), COLOR_SRC, cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
    cv2.circle(img_a, (su, sv), 6, COLOR_SRC, 2, cv2.LINE_AA)

    _draw_line_in_image(img_b, result.epipolar_line, COLOR_LINE, thickness=1)

    if draw_samples and result.samples.size:
        mask = np.isfinite(result.scores)
        pts = result.samples[mask]
        stride = max(1, len(pts) // 200)
        for (cx, cy) in pts[::stride]:
            cv2.circle(img_b, (int(round(cx)), int(round(cy))), 1, COLOR_SAMPL, -1)

    if result.predicted_pixel is not None:
        pu, pv = (int(round(result.predicted_pixel[0])),
                  int(round(result.predicted_pixel[1])))
        cv2.drawMarker(img_b, (pu, pv), COLOR_PRED, cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
        cv2.circle(img_b, (pu, pv), 7, COLOR_PRED, 2, cv2.LINE_AA)

    if ground_truth is not None:
        gu, gv = int(round(ground_truth[0])), int(round(ground_truth[1]))
        cv2.drawMarker(img_b, (gu, gv), COLOR_GT, cv2.MARKER_TILTED_CROSS, 22, 2, cv2.LINE_AA)
        cv2.circle(img_b, (gu, gv), 9, COLOR_GT, 2, cv2.LINE_AA)

    score_text = f"score={result.score:.3f}" if np.isfinite(result.score) else "score=n/a"
    if ground_truth is not None and result.predicted_pixel is not None:
        err = float(np.hypot(result.predicted_pixel[0] - ground_truth[0],
                             result.predicted_pixel[1] - ground_truth[1]))
        score_text += f"   err={err:.1f}px"
    cv2.putText(img_b, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)

    if img_a.shape[0] != img_b.shape[0]:
        h = min(img_a.shape[0], img_b.shape[0])
        img_a = img_a[:h]
        img_b = img_b[:h]
    return np.hstack([img_a, img_b])
