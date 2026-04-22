"""Shared dataclasses for the experimental telemetry subpackage.

All angles are stored in DEGREES unless explicitly noted. NaN is used as
the "missing" sentinel for optional numeric fields, so `math.isnan(x)`
is the canonical "is this unknown?" test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------- #
# FrameTelemetry                                                         #
# ---------------------------------------------------------------------- #

@dataclass
class FrameTelemetry:
    """Coarse telemetry parsed (or declared) for a single frame.

    Any field may be NaN to signal "not available for this frame". The
    downstream pose/epipolar stages treat NaN as "missing" and degrade
    to a coarser prior rather than failing.

    Coordinate / angle conventions
    ------------------------------
    * latitude  : decimal degrees, WGS84, +N / -S
    * longitude : decimal degrees, WGS84, +E / -W
    * altitude_m: metres. If `altitude_is_agl` is True the value is
      treated as height above ground / take-off point and combined
      with `reference_altitude_m` inside pose building; otherwise it
      is assumed to be WGS84 ellipsoidal / geodetic altitude.
    * heading_deg : compass yaw in degrees, clockwise from North
      (0 = North, 90 = East, 180 = South, 270 = West). This is the
      convention used by most drone HUDs.
    * pitch_deg   : camera/gimbal pitch in degrees; 0 = horizon,
                    negative = looking down, positive = looking up.
                    (Typical nadir-looking drone imagery: pitch ≈ -90.)
    * roll_deg    : camera/gimbal roll in degrees around the optical axis.
    """
    frame_name: str
    latitude: float = float("nan")
    longitude: float = float("nan")
    altitude_m: float = float("nan")
    altitude_is_agl: bool = True

    heading_deg: float = float("nan")
    pitch_deg: float = float("nan")
    roll_deg: float = float("nan")

    gimbal_pitch_deg: float = float("nan")
    gimbal_yaw_deg: float = float("nan")
    gimbal_roll_deg: float = float("nan")

    timestamp: Optional[str] = None
    raw_ocr: dict = field(default_factory=dict)
    notes: str = ""

    # ------- convenience predicates ------------------------------------
    def has_gps(self) -> bool:
        return not (math.isnan(self.latitude) or math.isnan(self.longitude))

    def has_altitude(self) -> bool:
        return not math.isnan(self.altitude_m)

    def has_heading(self) -> bool:
        return not math.isnan(self.heading_deg)

    def effective_pitch(self) -> float:
        """Gimbal pitch if present, else airframe pitch, else NaN."""
        if not math.isnan(self.gimbal_pitch_deg):
            return self.gimbal_pitch_deg
        return self.pitch_deg

    def effective_roll(self) -> float:
        if not math.isnan(self.gimbal_roll_deg):
            return self.gimbal_roll_deg
        return self.roll_deg

    def effective_yaw(self) -> float:
        """Gimbal yaw if present, else airframe heading, else NaN."""
        if not math.isnan(self.gimbal_yaw_deg):
            return self.gimbal_yaw_deg
        return self.heading_deg

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------- #
# CameraIntrinsics                                                        #
# ---------------------------------------------------------------------- #

@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics for the source frames.

    `fx`, `fy` are focal lengths in pixels, `cx`, `cy` the principal
    point in pixels. `width`, `height` are the nominal image size the
    intrinsics are calibrated against.
    """
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float) -> "CameraIntrinsics":
        """Build a square-pixel intrinsics matrix from horizontal FOV."""
        f = 0.5 * width / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(fx=f, fy=f, cx=0.5 * width, cy=0.5 * height,
                   width=int(width), height=int(height))

    def K(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,     1.0    ],
        ], dtype=np.float64)

    def scaled_to(self, width: int, height: int) -> "CameraIntrinsics":
        """Return a copy scaled to a different image size."""
        sx = width / self.width
        sy = height / self.height
        return CameraIntrinsics(
            fx=self.fx * sx, fy=self.fy * sy,
            cx=self.cx * sx, cy=self.cy * sy,
            width=int(width), height=int(height),
        )


# ---------------------------------------------------------------------- #
# Pose + results                                                          #
# ---------------------------------------------------------------------- #

@dataclass
class CameraPose:
    """Approximate world->camera pose in a local ENU frame.

    Convention
    ----------
    * World frame is local ENU (East, North, Up) centred at a chosen
      reference lat/lon/alt.
    * Camera frame: x=right, y=down, z=forward (OpenCV convention).
    * `R` maps a world point to the camera frame:  X_cam = R @ X_world + t.
    * `position_enu` is the camera centre in the world (ENU) frame;
       t = -R @ position_enu.
    """
    frame_name: str
    position_enu: np.ndarray              # (3,)
    R: np.ndarray                          # (3, 3)
    t: np.ndarray                          # (3,) = -R @ position_enu
    has_orientation: bool                  # False => R is identity-ish placeholder
    has_position: bool                     # False => position is NaN-ish
    source: str = ""                       # free-form provenance tag


@dataclass
class RelativePose:
    """Relative pose between two camera poses (A -> B).

    Given X_B = R_rel @ X_A + t_rel where points are expressed in
    camera-A coordinates on the left and camera-B coordinates on the
    right. When only position is known, `R_rel` is the identity and
    `orientation_valid` is False.
    """
    idx_a: int
    idx_b: int
    name_a: str
    name_b: str
    R_rel: np.ndarray                      # (3, 3)
    t_rel: np.ndarray                      # (3,) in camera-A units (metres)
    baseline_m: float
    orientation_valid: bool
    position_valid: bool


@dataclass
class EpipolarPrior:
    """Telemetry-derived epipolar prior for a single query pixel.

    Two flavours of output:
      * `line` : a proper epipolar line [a, b, c] in the target image
                 (only filled when we had full pose on both frames).
      * `direction_deg` : a coarse angular direction in degrees
                          (0 = pointing +x / right, 90 = +y / down)
                          that the line should travel along, together
                          with `epipole_hint` if available. This is the
                          fallback when we only have position.
    """
    source_pixel: Tuple[float, float]
    line: Optional[np.ndarray] = None           # (3,) or None
    epipole_hint: Optional[Tuple[float, float]] = None   # (u, v) or None
    direction_deg: Optional[float] = None
    mode: str = "unknown"                       # "full_pose" | "directional" | "none"
    note: str = ""


@dataclass
class EpipolarComparison:
    """Diagnostics comparing a telemetry prior against a vision line."""
    source_pixel: Tuple[float, float]
    vision_line: np.ndarray
    prior: EpipolarPrior

    angle_diff_deg: float = float("nan")        # angle between the two line directions
    point_line_distance_px: float = float("nan")  # distance from epipole_hint / source
                                                  # direction to the vision line
    epipole_hint_distance_px: float = float("nan")  # |epipole_hint - vision_line nearest point|
    mode: str = "unknown"
    note: str = ""
