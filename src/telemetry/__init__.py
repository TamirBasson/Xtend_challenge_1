"""Experimental telemetry -> coarse epipolar prior subpackage.

This package is EXPERIMENTAL. It is intentionally decoupled from the main
vision pipeline (`src.geometry`, `src.transfer`) and is NOT imported from
`src/__init__.py`. Import it explicitly as `from src.telemetry import ...`.

Goal
----
Use on-screen telemetry (latitude, longitude, altitude, heading, and any
orientation cues) to estimate a coarse inter-frame camera relationship
and derive an approximate epipolar line as a prior that can be compared
against the RANSAC-based geometry.

Modules
-------
- types      : FrameTelemetry, CameraIntrinsics dataclasses
- parsing    : JSON loader + optional pytesseract-based OCR extraction
- geodetic   : WGS84 <-> ECEF <-> local ENU conversion
- pose       : Build an approximate camera pose from telemetry
- epipolar   : Essential / Fundamental matrix from relative pose +
               directional prior fallback when orientation is missing
- compare    : Compare telemetry-based vs vision-based epipolar lines
"""

from .types import (
    FrameTelemetry,
    CameraIntrinsics,
    CameraPose,
    RelativePose,
    EpipolarPrior,
    EpipolarComparison,
)
from .parsing import (
    load_frame_telemetry_json,
    save_frame_telemetry_json,
    load_camera_intrinsics,
    ocr_extract_frame_telemetry,
    PYTESSERACT_AVAILABLE,
)
from .geodetic import (
    wgs84_to_ecef,
    ecef_to_enu,
    wgs84_to_enu,
    enu_basis_at,
)
from .pose import (
    build_camera_pose,
    relative_pose,
    pose_completeness,
)
from .epipolar import (
    essential_from_relative_pose,
    fundamental_from_relative_pose,
    epipolar_line_from_pose,
    directional_prior_from_translation,
)
from .compare import (
    compare_epipolar_lines,
    draw_epipolar_comparison,
)

__all__ = [
    "FrameTelemetry",
    "CameraIntrinsics",
    "CameraPose",
    "RelativePose",
    "EpipolarPrior",
    "EpipolarComparison",
    "load_frame_telemetry_json",
    "save_frame_telemetry_json",
    "load_camera_intrinsics",
    "ocr_extract_frame_telemetry",
    "PYTESSERACT_AVAILABLE",
    "wgs84_to_ecef",
    "ecef_to_enu",
    "wgs84_to_enu",
    "enu_basis_at",
    "build_camera_pose",
    "relative_pose",
    "pose_completeness",
    "essential_from_relative_pose",
    "fundamental_from_relative_pose",
    "epipolar_line_from_pose",
    "directional_prior_from_translation",
    "compare_epipolar_lines",
    "draw_epipolar_comparison",
]
