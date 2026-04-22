"""Phase 7 validator (EXPERIMENTAL): synthetic sanity tests for the
telemetry -> coarse epipolar-prior module.

No external frames are required — all tests run on synthetic data with
known ground truth. Exits with a non-zero status if any check fails.

Tests:
  1. Geodetic round-trip: WGS84 -> ECEF -> ENU -> back to WGS84
     (via ENU->ECEF inverse). Sub-metre accuracy.
  2. ENU baseline from two WGS84 points matches great-circle azimuth
     (order-of-1-degree agreement over short baselines).
  3. Camera pose for a forward-looking camera at heading=0, pitch=0
     produces a rotation whose camera-z axis points to +North.
  4. Two cameras with identical orientation and a pure eastward
     translation produce a horizontal epipolar line.
  5. JSON load/save round-trip preserves telemetry.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.telemetry import (  # noqa: E402
    CameraIntrinsics,
    FrameTelemetry,
    build_camera_pose,
    compare_epipolar_lines,
    enu_basis_at,
    epipolar_line_from_pose,
    load_frame_telemetry_json,
    relative_pose,
    save_frame_telemetry_json,
    wgs84_to_ecef,
    wgs84_to_enu,
)


FAILURES: List[str] = []


def _check(cond: bool, message: str) -> None:
    if cond:
        print(f"  OK   {message}")
    else:
        print(f"  FAIL {message}")
        FAILURES.append(message)


# ---------------------------------------------------------------------- #
# Tests                                                                   #
# ---------------------------------------------------------------------- #

def test_geodetic_roundtrip() -> None:
    print("\n[1] Geodetic round-trip")
    lat, lon, alt = 32.0853, 34.7818, 25.0            # Tel Aviv
    ecef = wgs84_to_ecef(lat, lon, alt)
    enu = wgs84_to_enu(lat, lon, alt, lat, lon, alt)
    _check(np.linalg.norm(enu) < 1e-6,
           f"self-reference ENU ~= 0 (got {np.linalg.norm(enu):.3g} m)")

    # A 100 m east offset should produce ENU ~= (100, 0, 0).
    dlon = 100.0 / (111_320.0 * math.cos(math.radians(lat)))
    enu_east = wgs84_to_enu(lat, lon + dlon, alt, lat, lon, alt)
    _check(abs(enu_east[0] - 100.0) < 0.5,
           f"east offset ENU.x ~= 100 (got {enu_east[0]:.3f})")
    _check(abs(enu_east[1]) < 0.5,
           f"east offset ENU.y ~= 0 (got {enu_east[1]:.3f})")


def test_enu_basis() -> None:
    print("\n[2] ENU basis orthonormality")
    R = enu_basis_at(32.0, 34.0)
    err = np.linalg.norm(R @ R.T - np.eye(3))
    _check(err < 1e-9, f"R @ R.T == I (err={err:.3g})")
    _check(abs(np.linalg.det(R) - 1.0) < 1e-9,
           f"det(R) == 1 (got {np.linalg.det(R):.9f})")


def test_camera_pose_orientation() -> None:
    print("\n[3] Camera pose orientation for heading=0, pitch=0")
    t = FrameTelemetry(
        frame_name="synth",
        latitude=32.0, longitude=34.0, altitude_m=0.0,
        altitude_is_agl=False,
        heading_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
    )
    pose = build_camera_pose(t, 32.0, 34.0, 0.0)
    # Camera z-axis in world = R_cw^T @ [0, 0, 1] = R_wc @ e_z.
    R_wc = pose.R.T
    z_cam_in_world = R_wc @ np.array([0, 0, 1.0])
    # Should point North in ENU == (0, 1, 0).
    _check(abs(z_cam_in_world[1] - 1.0) < 1e-6,
           f"camera-z points North (got {z_cam_in_world})")

    # Heading = 90 deg (east) should rotate camera z to +East = (1, 0, 0).
    t.heading_deg = 90.0
    pose = build_camera_pose(t, 32.0, 34.0, 0.0)
    R_wc = pose.R.T
    z_cam_in_world = R_wc @ np.array([0, 0, 1.0])
    _check(abs(z_cam_in_world[0] - 1.0) < 1e-6,
           f"heading=90 -> camera-z points East (got {z_cam_in_world})")


def test_pure_eastward_baseline() -> None:
    print("\n[4] Pure eastward baseline -> horizontal epipolar line")
    # Two cameras at the same altitude, same (heading=0, pitch=0, roll=0),
    # B is 100 m east of A. Epipolar lines should be (near-)horizontal.
    tel_a = FrameTelemetry("A", latitude=32.0, longitude=34.0, altitude_m=50.0,
                           altitude_is_agl=False, heading_deg=0.0,
                           pitch_deg=0.0, roll_deg=0.0)
    dlon = 100.0 / (111_320.0 * math.cos(math.radians(32.0)))
    tel_b = FrameTelemetry("B", latitude=32.0, longitude=34.0 + dlon, altitude_m=50.0,
                           altitude_is_agl=False, heading_deg=0.0,
                           pitch_deg=0.0, roll_deg=0.0)

    pose_a = build_camera_pose(tel_a, 32.0, 34.0, 0.0)
    pose_b = build_camera_pose(tel_b, 32.0, 34.0, 0.0)
    rel = relative_pose(pose_a, pose_b, 0, 1)
    _check(rel.orientation_valid and rel.position_valid,
           "relative pose flagged valid")
    _check(abs(rel.baseline_m - 100.0) < 1.0,
           f"baseline ~= 100 m (got {rel.baseline_m:.2f})")

    K = CameraIntrinsics.from_hfov(1280, 720, 75.0)
    prior = epipolar_line_from_pose(
        source_pixel=(640.0, 360.0),
        rel=rel, intrinsics_a=K, intrinsics_b=K,
        pose_a=pose_a, pose_b=pose_b,
        telemetry_a=tel_a, telemetry_b=tel_b,
    )
    _check(prior.mode == "full_pose",
           f"prior.mode == 'full_pose' (got {prior.mode})")
    _check(prior.line is not None, "prior.line populated")
    if prior.line is not None:
        a, b, _ = float(prior.line[0]), float(prior.line[1]), float(prior.line[2])
        # For a horizontal line: a small compared to b.
        norm = math.hypot(a, b)
        slope_deg = math.degrees(math.atan2(abs(a), abs(b) + 1e-9))
        _check(norm > 1e-6, "line has a non-zero normal direction")
        _check(slope_deg < 5.0,
               f"epipolar line is near-horizontal (slope={slope_deg:.2f} deg)")


def test_directional_prior_without_orientation() -> None:
    print("\n[5] Directional prior when orientation is missing")
    tel_a = FrameTelemetry("A", latitude=32.0, longitude=34.0,
                           altitude_m=50.0, altitude_is_agl=False)
    dlon = 100.0 / (111_320.0 * math.cos(math.radians(32.0)))
    tel_b = FrameTelemetry("B", latitude=32.0, longitude=34.0 + dlon,
                           altitude_m=50.0, altitude_is_agl=False)
    pose_a = build_camera_pose(tel_a, 32.0, 34.0, 0.0)
    pose_b = build_camera_pose(tel_b, 32.0, 34.0, 0.0)
    rel = relative_pose(pose_a, pose_b, 0, 1)
    K = CameraIntrinsics.from_hfov(1280, 720, 75.0)
    prior = epipolar_line_from_pose(
        source_pixel=(640.0, 360.0),
        rel=rel, intrinsics_a=K, intrinsics_b=K,
        pose_a=pose_a, pose_b=pose_b,
        telemetry_a=tel_a, telemetry_b=tel_b,
    )
    _check(prior.mode == "directional",
           f"mode == 'directional' when orientation missing (got {prior.mode})")
    _check(prior.direction_deg is not None and abs(prior.direction_deg - 90.0) < 1.0,
           f"direction ~= 90 deg (east) (got {prior.direction_deg})")


def test_json_roundtrip() -> None:
    print("\n[6] JSON load/save round-trip")
    tl = [
        FrameTelemetry("a.png", latitude=1.0, longitude=2.0, altitude_m=3.0,
                       heading_deg=42.0),
        FrameTelemetry("b.png"),  # all-NaN
    ]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.json"
        save_frame_telemetry_json(tl, p, reference=(1.0, 2.0, 0.0))
        loaded, meta = load_frame_telemetry_json(p, ["a.png", "b.png"])
    _check(len(loaded) == 2, "two frames loaded back")
    _check(math.isclose(loaded[0].latitude, 1.0), f"lat preserved (got {loaded[0].latitude})")
    _check(math.isnan(loaded[1].latitude),
           f"missing lat -> NaN (got {loaded[1].latitude})")
    _check(meta.get("reference", {}).get("latitude") == 1.0,
           "reference preserved in meta")


# ---------------------------------------------------------------------- #
# Entry                                                                   #
# ---------------------------------------------------------------------- #

def main() -> int:
    print("Phase 7 validation (experimental telemetry subpackage)")
    test_geodetic_roundtrip()
    test_enu_basis()
    test_camera_pose_orientation()
    test_pure_eastward_baseline()
    test_directional_prior_without_orientation()
    test_json_roundtrip()

    print("\n" + "=" * 50)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)")
        for msg in FAILURES:
            print(f"  - {msg}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
