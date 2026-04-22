"""Load telemetry from JSON, and (optionally) OCR it from frame overlays.

JSON format (one entry per frame, keyed by frame filename):

    {
      "reference": {
        "latitude":  32.0000,
        "longitude": 34.0000,
        "altitude_m": 0.0
      },
      "default_pitch_deg": 0.0,
      "default_roll_deg":  0.0,
      "frames": {
        "2026-02-15_16-25-03_04569.png": {
          "latitude":  32.0001,
          "longitude": 34.0002,
          "altitude_m": 42.5,
          "altitude_is_agl": true,
          "heading_deg": 137.0,
          "pitch_deg":   null,
          "roll_deg":    null,
          "gimbal_pitch_deg": -45.0,
          "gimbal_yaw_deg":   null,
          "gimbal_roll_deg":  null,
          "timestamp": "16:25:03",
          "notes": "manual"
        },
        ...
      }
    }

Any missing field is treated as unknown (NaN / None). `null` and
missing keys behave identically.

OCR notes
---------
The OCR path is OPTIONAL: it requires `pytesseract` AND a working
Tesseract binary installed system-wide. It is a best-effort extractor
designed for the regions named `gps`, `altitude`, `home` and `server`
in `config/overlay_regions.json`. It is expected to fail on some
frames — the caller should still get a `FrameTelemetry` object where
the unparseable fields are NaN, so the downstream pose builder can
fall back to the directional prior.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .types import CameraIntrinsics, FrameTelemetry


# ---------------------------------------------------------------------- #
# Optional pytesseract                                                    #
# ---------------------------------------------------------------------- #

try:  # best-effort: the experimental module must not hard-depend on OCR.
    import pytesseract  # type: ignore

    PYTESSERACT_AVAILABLE = True
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore
    PYTESSERACT_AVAILABLE = False


# ---------------------------------------------------------------------- #
# JSON I/O                                                                #
# ---------------------------------------------------------------------- #

def _coerce_float(x) -> float:
    if x is None:
        return float("nan")
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(v):
        return float("nan")
    return v


def load_frame_telemetry_json(
    path: str | Path,
    frame_names: Optional[Sequence[str]] = None,
) -> Tuple[List[FrameTelemetry], Dict]:
    """Load a telemetry JSON file.

    Returns
    -------
    telemetry_list : one FrameTelemetry per frame name. When `frame_names`
                     is provided, entries are produced in the same order
                     (unknown frames yield all-NaN telemetry).
    meta : the top-level JSON dict minus the per-frame payload (contains
           the reference origin, default pitch/roll, etc.).
    """
    p = Path(path)
    data = json.loads(p.read_text())
    frames = data.get("frames", {}) or {}

    if frame_names is None:
        frame_names = list(frames.keys())

    out: List[FrameTelemetry] = []
    for name in frame_names:
        entry = frames.get(name, {}) or {}
        out.append(FrameTelemetry(
            frame_name=name,
            latitude=_coerce_float(entry.get("latitude")),
            longitude=_coerce_float(entry.get("longitude")),
            altitude_m=_coerce_float(entry.get("altitude_m")),
            altitude_is_agl=bool(entry.get("altitude_is_agl", True)),
            heading_deg=_coerce_float(entry.get("heading_deg")),
            pitch_deg=_coerce_float(entry.get("pitch_deg")),
            roll_deg=_coerce_float(entry.get("roll_deg")),
            gimbal_pitch_deg=_coerce_float(entry.get("gimbal_pitch_deg")),
            gimbal_yaw_deg=_coerce_float(entry.get("gimbal_yaw_deg")),
            gimbal_roll_deg=_coerce_float(entry.get("gimbal_roll_deg")),
            timestamp=entry.get("timestamp"),
            raw_ocr=dict(entry.get("raw_ocr") or {}),
            notes=str(entry.get("notes") or ""),
        ))

    meta = {k: v for k, v in data.items() if k != "frames"}
    return out, meta


def save_frame_telemetry_json(
    telemetry: Sequence[FrameTelemetry],
    path: str | Path,
    reference: Optional[Tuple[float, float, float]] = None,
    default_pitch_deg: float = 0.0,
    default_roll_deg: float = 0.0,
) -> Path:
    """Serialize a list of FrameTelemetry to the JSON format above."""
    def _val(x: float):
        return None if (isinstance(x, float) and math.isnan(x)) else x

    payload: Dict = {}
    if reference is not None:
        payload["reference"] = {
            "latitude":   reference[0],
            "longitude":  reference[1],
            "altitude_m": reference[2],
        }
    payload["default_pitch_deg"] = default_pitch_deg
    payload["default_roll_deg"] = default_roll_deg
    payload["frames"] = {
        t.frame_name: {
            "latitude":         _val(t.latitude),
            "longitude":        _val(t.longitude),
            "altitude_m":       _val(t.altitude_m),
            "altitude_is_agl":  bool(t.altitude_is_agl),
            "heading_deg":      _val(t.heading_deg),
            "pitch_deg":        _val(t.pitch_deg),
            "roll_deg":         _val(t.roll_deg),
            "gimbal_pitch_deg": _val(t.gimbal_pitch_deg),
            "gimbal_yaw_deg":   _val(t.gimbal_yaw_deg),
            "gimbal_roll_deg":  _val(t.gimbal_roll_deg),
            "timestamp":        t.timestamp,
            "raw_ocr":          dict(t.raw_ocr),
            "notes":            t.notes,
        } for t in telemetry
    }

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return out


def load_camera_intrinsics(path: str | Path) -> CameraIntrinsics:
    """Load intrinsics from a small JSON descriptor.

    Supports two forms:
      * Explicit pinhole : {"fx", "fy", "cx", "cy", "width", "height"}
      * FOV-based        : {"hfov_deg", "width", "height"}
    """
    p = Path(path)
    data = json.loads(p.read_text())

    width = int(data["width"])
    height = int(data["height"])

    if "fx" in data and "fy" in data:
        return CameraIntrinsics(
            fx=float(data["fx"]), fy=float(data["fy"]),
            cx=float(data.get("cx", 0.5 * width)),
            cy=float(data.get("cy", 0.5 * height)),
            width=width, height=height,
        )
    if "hfov_deg" in data:
        return CameraIntrinsics.from_hfov(width, height, float(data["hfov_deg"]))
    raise ValueError(
        f"{p}: intrinsics JSON must declare either (fx, fy[, cx, cy]) or hfov_deg."
    )


# ---------------------------------------------------------------------- #
# OCR extraction (optional)                                              #
# ---------------------------------------------------------------------- #

_NUMBER_RE = re.compile(r"[-+]?\d+\.\d+|[-+]?\d+")


def _parse_numbers(text: str) -> List[float]:
    return [float(m) for m in _NUMBER_RE.findall(text)]


def _ocr_crop(image: np.ndarray, rect: Tuple[int, int, int, int]) -> str:
    """Run Tesseract on a single cropped region. Returns "" on failure."""
    if not PYTESSERACT_AVAILABLE:
        return ""
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return ""
    crop = image[y:y + h, x:x + w]
    if crop.size == 0:
        return ""
    # Simple upscale + grayscale; good enough for large HUD text.
    try:
        import cv2  # local import keeps the module importable without cv2
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        big = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        _, bw = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return pytesseract.image_to_string(bw, config="--psm 6")  # type: ignore
    except Exception:
        return ""


def _scale_rect(
    rect_xywh: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    calibration_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    img_w, img_h = image_size
    cal_w, cal_h = calibration_size
    if (img_w, img_h) == (cal_w, cal_h):
        return rect_xywh
    sx = img_w / cal_w
    sy = img_h / cal_h
    x, y, w, h = rect_xywh
    return (int(round(x * sx)), int(round(y * sy)),
            int(round(w * sx)), int(round(h * sy)))


def ocr_extract_frame_telemetry(
    frame_name: str,
    image: np.ndarray,
    regions: Dict[str, Tuple[int, int, int, int]],
    calibration_size: Tuple[int, int] = (1280, 720),
) -> FrameTelemetry:
    """Best-effort OCR extractor for a single frame.

    `regions` is a dict mapping region names to (x, y, w, h) in the
    calibration-size pixel space. It is expected to contain some subset
    of `{"gps", "altitude", "home", "server"}`. Any missing region or
    OCR failure leaves the corresponding field as NaN.

    The parsing rules are intentionally conservative:
      * `gps`       : two floating-point numbers -> (lat, lon)
      * `altitude`  : first numeric token is metres of altitude
      * `home`      : first numeric token that looks like a compass
                      heading (0..360) is interpreted as heading_deg
      * `server`    : currently only a pass-through into `raw_ocr`

    This is a baseline. Real HUDs have unit suffixes, degrees symbols
    and layout quirks that will require per-field tuning — leave that
    tuning to the validation script.
    """
    h_img, w_img = image.shape[:2]
    lat = lon = alt = heading = float("nan")
    raw: Dict[str, str] = {}

    gps_rect = regions.get("gps")
    if gps_rect is not None:
        text = _ocr_crop(image, _scale_rect(gps_rect, (w_img, h_img), calibration_size))
        raw["gps"] = text
        nums = _parse_numbers(text)
        if len(nums) >= 2:
            lat, lon = nums[0], nums[1]

    alt_rect = regions.get("altitude")
    if alt_rect is not None:
        text = _ocr_crop(image, _scale_rect(alt_rect, (w_img, h_img), calibration_size))
        raw["altitude"] = text
        nums = _parse_numbers(text)
        if nums:
            alt = nums[0]

    home_rect = regions.get("home")
    if home_rect is not None:
        text = _ocr_crop(image, _scale_rect(home_rect, (w_img, h_img), calibration_size))
        raw["home"] = text
        for n in _parse_numbers(text):
            if 0.0 <= n <= 360.0:
                heading = n
                break

    server_rect = regions.get("server")
    if server_rect is not None:
        text = _ocr_crop(image, _scale_rect(server_rect, (w_img, h_img), calibration_size))
        raw["server"] = text

    return FrameTelemetry(
        frame_name=frame_name,
        latitude=lat,
        longitude=lon,
        altitude_m=alt,
        altitude_is_agl=True,
        heading_deg=heading,
        raw_ocr=raw,
        notes="ocr" if PYTESSERACT_AVAILABLE else "ocr_unavailable",
    )
