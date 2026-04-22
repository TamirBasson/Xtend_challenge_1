"""Build approximate camera poses from parsed telemetry.

Conventions
-----------
World frame : local ENU (East, North, Up) centred at a chosen reference
              (typically the first GPS-valid frame).
Camera frame: OpenCV convention (x = right, y = down, z = forward).

The drone HUD reports heading as a compass yaw (0 = North, clockwise).
The camera orientation is built as a chain of rotations applied to a
"nominal" forward-North / camera-aligned-with-ENU orientation:

    1. Rotate by heading around the Up axis (compass yaw -> yaw in ENU).
    2. Apply gimbal/airframe pitch (negative = looking down).
    3. Apply roll around the resulting optical axis.

When orientation data is missing the rotation is replaced by a safe
fallback (identity or heading-only) and the `has_orientation` flag on
the returned pose is set to False so downstream stages can choose a
directional prior instead of a full essential-matrix computation.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from .types import CameraPose, FrameTelemetry, RelativePose
from .geodetic import wgs84_to_enu


# ---------------------------------------------------------------------- #
# Rotation helpers                                                        #
# ---------------------------------------------------------------------- #

def _rot_x(angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _R_world_from_camera(heading_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Build R_wc: columns are the camera axes expressed in the ENU world
    frame, given compass heading, pitch and roll in degrees.

    Nominal camera (heading = 0, pitch = 0, roll = 0):
        x_cam (right)   -> +East
        y_cam (down)    -> -Up
        z_cam (forward) -> +North

    so R_nominal has those as its three columns.

    Rotations are composed as:
        R_wc = R_yaw_world @ R_nominal @ R_pitch_cam @ R_roll_cam

      * R_yaw_world = Rz(-heading) in ENU (compass is cw-from-N, ENU z
        is right-handed Up, so positive compass heading is a negative
        rotation about +z in ENU).
      * R_pitch_cam = Rx(pitch) is applied in the *camera* frame; with
        the standard aviation convention (positive = nose up) this
        rotates the forward axis upward. Negative pitch (e.g. -90) is
        a nadir-looking camera.
      * R_roll_cam = Rz(roll) rotates about the optical axis.

    Any NaN input is treated as 0.
    """
    hd = 0.0 if math.isnan(heading_deg) else heading_deg
    pt = 0.0 if math.isnan(pitch_deg)   else pitch_deg
    rl = 0.0 if math.isnan(roll_deg)    else roll_deg

    R_nominal = np.array([
        [1.0, 0.0,  0.0],    # col 0 = x_cam in world = +E
        [0.0, 0.0,  1.0],    # col 1 = y_cam in world = -U (so down)
        [0.0, -1.0, 0.0],    # col 2 = z_cam in world = +N
    ], dtype=np.float64)

    # Columns of R_nominal are (E, -U, N): verify programmatically if in doubt.
    R_yaw = _rot_z(-hd)
    R_pitch = _rot_x(pt)
    R_roll = _rot_z(rl)

    return R_yaw @ R_nominal @ R_pitch @ R_roll


# ---------------------------------------------------------------------- #
# Public API                                                              #
# ---------------------------------------------------------------------- #

def build_camera_pose(
    telemetry: FrameTelemetry,
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_alt_m: float,
    default_pitch_deg: float = 0.0,
    default_roll_deg: float = 0.0,
) -> CameraPose:
    """Build a `CameraPose` from parsed telemetry.

    `ref_lat_deg`, `ref_lon_deg`, `ref_alt_m` define the origin of the
    local ENU world frame. `default_pitch_deg` / `default_roll_deg` are
    used when the telemetry is silent about those axes — typical values
    are 0 for a forward-looking camera or -90 for a nadir-looking gimbal.
    """
    # --- position in ENU -----------------------------------------------
    if telemetry.has_gps():
        alt = telemetry.altitude_m if telemetry.has_altitude() else 0.0
        if telemetry.altitude_is_agl:
            alt = ref_alt_m + alt
        position = wgs84_to_enu(
            telemetry.latitude, telemetry.longitude, alt,
            ref_lat_deg, ref_lon_deg, ref_alt_m,
        )
        position_valid = True
    else:
        position = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        position_valid = False

    # --- orientation ---------------------------------------------------
    yaw = telemetry.effective_yaw()
    pitch = telemetry.effective_pitch()
    roll = telemetry.effective_roll()

    has_orientation = telemetry.has_heading()  # heading is the minimum we need
    use_pitch = pitch if not math.isnan(pitch) else default_pitch_deg
    use_roll  = roll  if not math.isnan(roll)  else default_roll_deg
    use_yaw   = yaw   if not math.isnan(yaw)   else 0.0

    R_wc = _R_world_from_camera(use_yaw, use_pitch, use_roll)
    R_cw = R_wc.T  # world -> camera

    if position_valid:
        t = -R_cw @ position
    else:
        t = np.array([np.nan, np.nan, np.nan], dtype=np.float64)

    return CameraPose(
        frame_name=telemetry.frame_name,
        position_enu=position,
        R=R_cw,
        t=t,
        has_orientation=has_orientation,
        has_position=position_valid,
        source=("telemetry" if (position_valid or has_orientation) else "empty"),
    )


def relative_pose(
    pose_a: CameraPose,
    pose_b: CameraPose,
    idx_a: int = -1,
    idx_b: int = -1,
) -> RelativePose:
    """Return pose B expressed relative to pose A.

    If X_A / X_B are the same world point expressed in camera A / B
    frames, then  X_B = R_rel @ X_A + t_rel. This is the form OpenCV
    wants for `cv2.stereoRectify` / essential-matrix construction.
    """
    orientation_valid = pose_a.has_orientation and pose_b.has_orientation
    position_valid = pose_a.has_position and pose_b.has_position

    if orientation_valid:
        R_rel = pose_b.R @ pose_a.R.T
    else:
        R_rel = np.eye(3, dtype=np.float64)

    if position_valid:
        # Camera centres in world -> baseline in world frame, then expressed
        # in camera-A coordinates.
        baseline_world = pose_b.position_enu - pose_a.position_enu
        if pose_a.has_orientation:
            t_rel = -R_rel @ (pose_a.R @ baseline_world) * 0.0 + \
                    (pose_b.t - R_rel @ pose_a.t)
        else:
            # No orientation on A: express baseline in "world-parallel"
            # camera frame, which is still a meaningful translation
            # direction even if rotationally uncalibrated.
            t_rel = baseline_world.copy()
        baseline_m = float(np.linalg.norm(baseline_world))
    else:
        t_rel = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        baseline_m = float("nan")

    return RelativePose(
        idx_a=idx_a, idx_b=idx_b,
        name_a=pose_a.frame_name, name_b=pose_b.frame_name,
        R_rel=R_rel, t_rel=t_rel,
        baseline_m=baseline_m,
        orientation_valid=orientation_valid,
        position_valid=position_valid,
    )


def pose_completeness(pose: CameraPose) -> str:
    """Human-readable tag for a pose: 'full' | 'position_only' | 'orientation_only' | 'none'."""
    if pose.has_position and pose.has_orientation:
        return "full"
    if pose.has_position:
        return "position_only"
    if pose.has_orientation:
        return "orientation_only"
    return "none"


def pick_reference_origin(
    telemetry_list: Sequence[FrameTelemetry],
) -> Optional[Tuple[float, float, float]]:
    """Pick a reasonable ENU origin: the first frame with GPS.

    Altitude for the origin uses whatever the frame declares; if the
    altitude is AGL we still anchor ENU at the take-off datum
    (alt contribution cancels for relative positions anyway).
    """
    for t in telemetry_list:
        if t.has_gps():
            alt = t.altitude_m if (t.has_altitude() and not t.altitude_is_agl) else 0.0
            return (t.latitude, t.longitude, alt)
    return None
