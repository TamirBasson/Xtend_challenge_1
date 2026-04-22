"""Derive an approximate epipolar prior from telemetry-based relative pose.

The main public entry point is `epipolar_line_from_pose`, which branches on
pose completeness:

  * full pose on both frames   -> essential matrix -> fundamental matrix
                                  -> proper epipolar line in the target image
  * position only              -> directional prior: compass bearing from
                                  frame A to frame B (useful as a sanity
                                  check on the vision epipole orientation)
  * nothing                    -> EpipolarPrior with mode="none"

This module does NOT try to replace the RANSAC-based fundamental matrix.
Its output is explicitly labelled as a coarse prior and is intended to be
compared against the vision pipeline, not to supersede it.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from .types import (
    CameraIntrinsics,
    CameraPose,
    EpipolarPrior,
    FrameTelemetry,
    RelativePose,
)
from .geodetic import azimuth_between_wgs84


# ---------------------------------------------------------------------- #
# Essential / Fundamental from relative pose                              #
# ---------------------------------------------------------------------- #

def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([
        [0.0, -z,   y],
        [z,    0.0, -x],
        [-y,   x,   0.0],
    ], dtype=np.float64)


def essential_from_relative_pose(rel: RelativePose) -> Optional[np.ndarray]:
    """E = [t_rel]_x @ R_rel, or None if pose is not fully valid."""
    if not (rel.orientation_valid and rel.position_valid):
        return None
    t = np.asarray(rel.t_rel, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(t))
    if not np.isfinite(norm) or norm < 1e-9:
        return None
    # Normalize translation: essential matrix is only defined up to scale,
    # and this keeps downstream F well-conditioned.
    t_hat = t / norm
    return _skew(t_hat) @ rel.R_rel


def fundamental_from_relative_pose(
    rel: RelativePose,
    K_a: np.ndarray,
    K_b: np.ndarray,
) -> Optional[np.ndarray]:
    """F = K_b^{-T} @ E @ K_a^{-1}, matching the `x_b^T F x_a = 0` convention."""
    E = essential_from_relative_pose(rel)
    if E is None:
        return None
    K_a_inv = np.linalg.inv(np.asarray(K_a, dtype=np.float64))
    K_b_inv_T = np.linalg.inv(np.asarray(K_b, dtype=np.float64)).T
    F = K_b_inv_T @ E @ K_a_inv
    # Scale for numerical stability; sign / scale are irrelevant for lines.
    n = float(np.linalg.norm(F))
    if not np.isfinite(n) or n < 1e-12:
        return None
    return F / n


# ---------------------------------------------------------------------- #
# Epipole hint from translation only                                      #
# ---------------------------------------------------------------------- #

def directional_prior_from_translation(
    pose_a: CameraPose,
    pose_b: CameraPose,
    telemetry_a: Optional[FrameTelemetry] = None,
    telemetry_b: Optional[FrameTelemetry] = None,
) -> Tuple[Optional[float], str]:
    """Return a compass direction (degrees, 0=N cw) from A to B, if possible.

    Prefers the great-circle bearing if we have lat/lon on both frames,
    otherwise falls back to ENU baseline direction.
    """
    if (telemetry_a is not None and telemetry_b is not None
            and telemetry_a.has_gps() and telemetry_b.has_gps()):
        az = azimuth_between_wgs84(
            telemetry_a.latitude, telemetry_a.longitude,
            telemetry_b.latitude, telemetry_b.longitude,
        )
        return float(az), "azimuth_wgs84"

    if pose_a.has_position and pose_b.has_position:
        d = pose_b.position_enu - pose_a.position_enu
        # ENU: x=E, y=N -> compass = atan2(E, N) then wrap to [0, 360).
        az = (math.degrees(math.atan2(d[0], d[1])) + 360.0) % 360.0
        return float(az), "azimuth_enu"

    return None, "no_translation_available"


# ---------------------------------------------------------------------- #
# Main entry point                                                        #
# ---------------------------------------------------------------------- #

def epipolar_line_from_pose(
    source_pixel: Tuple[float, float],
    rel: RelativePose,
    intrinsics_a: CameraIntrinsics,
    intrinsics_b: CameraIntrinsics,
    pose_a: Optional[CameraPose] = None,
    pose_b: Optional[CameraPose] = None,
    telemetry_a: Optional[FrameTelemetry] = None,
    telemetry_b: Optional[FrameTelemetry] = None,
) -> EpipolarPrior:
    """Compute an epipolar prior for `source_pixel` in frame A, in frame B."""
    src = (float(source_pixel[0]), float(source_pixel[1]))

    # ---- full-pose branch ------------------------------------------------
    if rel.orientation_valid and rel.position_valid:
        F = fundamental_from_relative_pose(rel, intrinsics_a.K(), intrinsics_b.K())
        if F is not None:
            pt = np.array([[src]], dtype=np.float32)
            line = cv2.computeCorrespondEpilines(pt, 1, F).reshape(3).astype(np.float64)

            # Epipole in B = K_b * t_rel (homogeneous) up to scale.
            t = np.asarray(rel.t_rel, dtype=np.float64).reshape(3)
            e_b_h = intrinsics_b.K() @ t
            epipole_hint = None
            if abs(e_b_h[2]) > 1e-9:
                epipole_hint = (float(e_b_h[0] / e_b_h[2]),
                                float(e_b_h[1] / e_b_h[2]))

            # Line direction in image: perpendicular to [a, b].
            a, b = float(line[0]), float(line[1])
            norm = math.hypot(a, b)
            direction_deg = None
            if norm > 1e-9:
                direction_deg = math.degrees(math.atan2(-a, b))

            return EpipolarPrior(
                source_pixel=src,
                line=line,
                epipole_hint=epipole_hint,
                direction_deg=direction_deg,
                mode="full_pose",
                note=f"baseline={rel.baseline_m:.2f} m",
            )

    # ---- directional fallback -------------------------------------------
    if pose_a is not None and pose_b is not None:
        az, tag = directional_prior_from_translation(
            pose_a, pose_b, telemetry_a, telemetry_b,
        )
        if az is not None:
            return EpipolarPrior(
                source_pixel=src,
                line=None,
                epipole_hint=None,
                direction_deg=az,
                mode="directional",
                note=f"compass_bearing_deg={az:.1f} ({tag})",
            )

    return EpipolarPrior(
        source_pixel=src,
        line=None,
        epipole_hint=None,
        direction_deg=None,
        mode="none",
        note="insufficient telemetry",
    )
