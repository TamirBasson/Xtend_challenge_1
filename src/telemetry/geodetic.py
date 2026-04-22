"""WGS84 <-> ECEF <-> local ENU conversions.

Pure NumPy. No external geodesy dependency.

References
----------
* WGS84 ellipsoid constants: a = 6378137.0, f = 1/298.257223563.
* ECEF/ENU math: standard textbook derivation (see e.g. Misra & Enge,
  "Global Positioning System", App. B).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


# WGS84 ellipsoid
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


def wgs84_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Convert geodetic (lat, lon, alt) to Earth-Centred Earth-Fixed XYZ.

    Parameters
    ----------
    lat_deg, lon_deg : decimal degrees, WGS84.
    alt_m : ellipsoidal altitude in metres.

    Returns
    -------
    (3,) float64 array in metres.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    x = (n + alt_m) * cos_lat * cos_lon
    y = (n + alt_m) * cos_lat * sin_lon
    z = (n * (1.0 - _WGS84_E2) + alt_m) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def enu_basis_at(ref_lat_deg: float, ref_lon_deg: float) -> np.ndarray:
    """Return the 3x3 rotation ECEF -> ENU at a reference lat/lon.

    ENU basis at (lat0, lon0):
        e = [-sin(lon),               cos(lon),              0      ]
        n = [-sin(lat) cos(lon), -sin(lat) sin(lon), cos(lat)]
        u = [ cos(lat) cos(lon),  cos(lat) sin(lon), sin(lat)]
    Each row of the returned matrix expresses one ENU axis in ECEF.
    """
    lat = math.radians(ref_lat_deg)
    lon = math.radians(ref_lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    return np.array([
        [-so,         co,       0.0],
        [-sl * co, -sl * so,     cl],
        [ cl * co,  cl * so,     sl],
    ], dtype=np.float64)


def ecef_to_enu(
    ecef_xyz: np.ndarray,
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_alt_m: float,
) -> np.ndarray:
    """Convert an ECEF point to local ENU (metres) around a reference."""
    ref_ecef = wgs84_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)
    R = enu_basis_at(ref_lat_deg, ref_lon_deg)
    return R @ (np.asarray(ecef_xyz, dtype=np.float64) - ref_ecef)


def wgs84_to_enu(
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_alt_m: float,
) -> np.ndarray:
    """One-shot WGS84 -> local ENU (metres)."""
    return ecef_to_enu(
        wgs84_to_ecef(lat_deg, lon_deg, alt_m),
        ref_lat_deg, ref_lon_deg, ref_alt_m,
    )


def azimuth_between_wgs84(
    lat1_deg: float, lon1_deg: float,
    lat2_deg: float, lon2_deg: float,
) -> float:
    """Initial bearing (degrees, compass 0=N cw) from point 1 to point 2.

    Uses the standard spherical approximation. Adequate for coarse drone
    telemetry at short baselines.
    """
    l1 = math.radians(lat1_deg)
    l2 = math.radians(lat2_deg)
    dlon = math.radians(lon2_deg - lon1_deg)
    x = math.sin(dlon) * math.cos(l2)
    y = math.cos(l1) * math.sin(l2) - math.sin(l1) * math.cos(l2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def horizontal_distance_m(enu_a: np.ndarray, enu_b: np.ndarray) -> float:
    """Horizontal (E/N only) distance between two ENU points, in metres."""
    d = np.asarray(enu_b, dtype=np.float64)[:2] - np.asarray(enu_a, dtype=np.float64)[:2]
    return float(math.hypot(d[0], d[1]))
