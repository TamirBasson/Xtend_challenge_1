"""Compare a telemetry epipolar prior to a vision-based epipolar line."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from .types import EpipolarComparison, EpipolarPrior


def _line_direction_deg(line: np.ndarray) -> Optional[float]:
    a, b = float(line[0]), float(line[1])
    norm = math.hypot(a, b)
    if norm < 1e-9:
        return None
    return math.degrees(math.atan2(-a, b))  # tangent direction = (-a, b) rotated


def _signed_angle_diff_deg(a: float, b: float, wrap_180: bool = True) -> float:
    """Smallest absolute angle difference, optionally treating lines as
    undirected (wrap to [0, 90] -> [0, 180])."""
    d = (a - b + 180.0) % 360.0 - 180.0
    d = abs(d)
    if wrap_180 and d > 90.0:
        d = 180.0 - d
    return d


def _distance_point_to_line(line: np.ndarray, pt: Tuple[float, float]) -> float:
    a, b, c = float(line[0]), float(line[1]), float(line[2])
    norm = math.hypot(a, b)
    if norm < 1e-9:
        return float("inf")
    return abs(a * pt[0] + b * pt[1] + c) / norm


def compare_epipolar_lines(
    vision_line: np.ndarray,
    prior: EpipolarPrior,
) -> EpipolarComparison:
    """Quantify the agreement between a vision line and a telemetry prior.

    When `vision_line` contains NaNs (vision pipeline unavailable), the
    comparison fields are left as NaN but the prior is still returned
    so callers can still visualize the telemetry output.
    """
    vision_line = np.asarray(vision_line, dtype=np.float64)
    cmp = EpipolarComparison(
        source_pixel=prior.source_pixel,
        vision_line=vision_line,
        prior=prior,
        mode=prior.mode,
    )

    if not np.all(np.isfinite(vision_line)):
        cmp.note = "vision line unavailable"
        return cmp

    vis_dir = _line_direction_deg(cmp.vision_line)
    if vis_dir is not None and prior.direction_deg is not None:
        if prior.mode == "directional":
            # Compass bearing (0=N cw) converted to image-plane angle
            # only makes sense with an intrinsics/orientation chain.
            # We still report the raw compass-vs-line angle for diagnostic
            # purposes but wrap to [0, 90] so it stays interpretable.
            cmp.angle_diff_deg = _signed_angle_diff_deg(
                vis_dir, prior.direction_deg, wrap_180=True
            )
        else:
            cmp.angle_diff_deg = _signed_angle_diff_deg(
                vis_dir, prior.direction_deg, wrap_180=True
            )

    # Epipole-hint distance to vision line.
    if prior.epipole_hint is not None:
        cmp.epipole_hint_distance_px = _distance_point_to_line(
            cmp.vision_line, prior.epipole_hint
        )

    # Point-line distance: source_pixel should not lie on the vision line
    # (different image), but we report the source-pixel distance to the
    # prior line for full-pose mode.
    if prior.line is not None:
        cmp.point_line_distance_px = _distance_point_to_line(
            cmp.vision_line, prior.epipole_hint or prior.source_pixel,
        )
        cmp.note = "full_pose comparison"
    elif prior.direction_deg is not None:
        cmp.note = "directional comparison (compass bearing only)"
    else:
        cmp.note = "prior has no geometric output"

    return cmp


# ---------------------------------------------------------------------- #
# Visualization                                                           #
# ---------------------------------------------------------------------- #

def _clip_line_to_image(
    line: np.ndarray, shape: Tuple[int, int]
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    a, b, c = float(line[0]), float(line[1]), float(line[2])
    h, w = shape[:2]
    if abs(b) > 1e-6:
        x0, x1 = 0, w - 1
        y0 = int(round(-(a * x0 + c) / b))
        y1 = int(round(-(a * x1 + c) / b))
        return (x0, y0), (x1, y1)
    if abs(a) > 1e-6:
        y0, y1 = 0, h - 1
        x0 = int(round(-(b * y0 + c) / a))
        x1 = int(round(-(b * y1 + c) / a))
        return (x0, y0), (x1, y1)
    return None


def draw_epipolar_comparison(
    image_src: np.ndarray,
    image_dst: np.ndarray,
    source_pixel: Tuple[float, float],
    vision_line: np.ndarray,
    prior: EpipolarPrior,
    cmp: Optional[EpipolarComparison] = None,
) -> np.ndarray:
    """Side-by-side: source marker on the left, both lines on the right.

    Yellow = vision-based epipolar line (the reference).
    Cyan   = telemetry-based epipolar line (full_pose mode only).
    Magenta arrow = compass-bearing direction (directional mode).
    """
    img_a = image_src.copy() if image_src.ndim == 3 else cv2.cvtColor(image_src, cv2.COLOR_GRAY2BGR)
    img_b = image_dst.copy() if image_dst.ndim == 3 else cv2.cvtColor(image_dst, cv2.COLOR_GRAY2BGR)

    COLOR_SRC     = (0, 0, 255)
    COLOR_VISION  = (0, 255, 255)
    COLOR_PRIOR   = (255, 255, 0)
    COLOR_DIR     = (255, 0, 255)
    COLOR_EPIPOLE = (0, 200, 0)

    su, sv = int(round(source_pixel[0])), int(round(source_pixel[1]))
    cv2.drawMarker(img_a, (su, sv), COLOR_SRC, cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
    cv2.circle(img_a, (su, sv), 6, COLOR_SRC, 2, cv2.LINE_AA)

    # Vision line (ground-truth reference here). May be NaN when RANSAC fails.
    vision_finite = np.all(np.isfinite(np.asarray(vision_line, dtype=np.float64)))
    if vision_finite:
        seg = _clip_line_to_image(vision_line, img_b.shape)
        if seg is not None:
            cv2.line(img_b, seg[0], seg[1], COLOR_VISION, 2, cv2.LINE_AA)

    # Telemetry line (full_pose mode).
    if prior.line is not None:
        seg = _clip_line_to_image(prior.line, img_b.shape)
        if seg is not None:
            cv2.line(img_b, seg[0], seg[1], COLOR_PRIOR, 2, cv2.LINE_AA)

    # Telemetry epipole hint.
    if prior.epipole_hint is not None:
        eu, ev = int(round(prior.epipole_hint[0])), int(round(prior.epipole_hint[1]))
        if -10_000 < eu < 10_000 and -10_000 < ev < 10_000:
            cv2.circle(img_b, (eu, ev), 8, COLOR_EPIPOLE, 2, cv2.LINE_AA)

    # Directional-mode arrow: draw from image centre outward in the
    # compass bearing direction (top = north).
    if prior.mode == "directional" and prior.direction_deg is not None:
        h, w = img_b.shape[:2]
        cx, cy = w // 2, h // 2
        ang = math.radians(prior.direction_deg)
        L = int(0.35 * min(h, w))
        ex = int(cx + L * math.sin(ang))
        ey = int(cy - L * math.cos(ang))
        cv2.arrowedLine(img_b, (cx, cy), (ex, ey), COLOR_DIR, 2, cv2.LINE_AA, tipLength=0.08)

    # Text overlay.
    lines = [f"mode: {prior.mode}"]
    if not vision_finite:
        lines.append("vision: unavailable")
    if cmp is not None:
        if math.isfinite(cmp.angle_diff_deg):
            lines.append(f"angle_diff: {cmp.angle_diff_deg:.1f} deg")
        if math.isfinite(cmp.epipole_hint_distance_px):
            lines.append(f"epipole->vision line: {cmp.epipole_hint_distance_px:.1f} px")
    if prior.note:
        lines.append(prior.note)
    y = 28
    for text in lines:
        cv2.putText(img_b, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img_b, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26

    if img_a.shape[0] != img_b.shape[0]:
        h = min(img_a.shape[0], img_b.shape[0])
        img_a = img_a[:h]
        img_b = img_b[:h]
    return np.hstack([img_a, img_b])
