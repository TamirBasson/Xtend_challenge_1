"""OCR-based telemetry extractor (EXPERIMENTAL).

Reads the named overlay regions from `config/overlay_regions.json`,
runs EasyOCR on each region for every frame, and writes the parsed
telemetry into `config/frame_telemetry.json` (or another path).

Region -> field mapping (tuned to the actual HUD layout of this dataset):

    gps      -> latitude, longitude       (LAT: xx.xxxxxx / LON: yy.yyyyyy)
    altitude -> altitude_m                (first number, in metres AGL)
                speed_kmh (raw_ocr only)  (third number, not used for pose)
    home     -> heading_deg               (large centred number on the
                                           compass ribbon, 0..360)
    server   -> timestamp                 (HH:MM:SS)

Anything OCR couldn't read is left as NaN / None, exactly as expected
by the phase-7 pose builder. A raw dump of every region's OCR output is
stored in the `raw_ocr` field for debugging.

Usage (from repository root):
    python scripts/ocr_extract_telemetry.py
    python scripts/ocr_extract_telemetry.py --output my_telemetry.json
    python scripts/ocr_extract_telemetry.py --frames 0 1 2 --debug-crops
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from src import load_frames, load_regions_from_json  # noqa: E402
from src.telemetry import (  # noqa: E402
    FrameTelemetry,
    save_frame_telemetry_json,
)
from src.telemetry.pose import pick_reference_origin  # noqa: E402


INPUT_FOLDER = REPO_ROOT / "drones_images_input"
DEFAULT_REGIONS_JSON = REPO_ROOT / "config" / "overlay_regions.json"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "config" / "frame_telemetry.json"
DEBUG_DIR = REPO_ROOT / "outputs" / "phase7" / "ocr_crops"


# ---------------------------------------------------------------------- #
# Preprocessing + OCR                                                     #
# ---------------------------------------------------------------------- #

def _scale_rect(r: dict, img_size: Tuple[int, int], cal_size: Tuple[int, int]):
    iw, ih = img_size
    cw, ch = cal_size
    sx, sy = iw / cw, ih / ch
    return (int(round(r["x"] * sx)), int(round(r["y"] * sy)),
            int(round(r["w"] * sx)), int(round(r["h"] * sy)))


def _preprocess(crop: np.ndarray, upscale: float = 2.0) -> np.ndarray:
    """Enhance HUD text for OCR: grayscale, upscale, adaptive threshold.

    The drone HUD renders white text with a thin dark shadow on varying
    backgrounds. A simple Otsu threshold after mild upscaling gives
    EasyOCR a much cleaner input than the raw RGB.
    """
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    if upscale != 1.0:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale,
                          interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # HUD text is light on dark; after Otsu the text may end up either
    # polarity. Pick the polarity whose white-pixel count is smaller
    # (text is usually the minority class).
    white_ratio = float((th == 255).mean())
    if white_ratio > 0.5:
        th = cv2.bitwise_not(th)
    return th


def _ocr_texts(reader, crop: np.ndarray) -> List[str]:
    """Return a list of recognised text strings for `crop`.

    EasyOCR tends to split "LAT: 32.055649" into two tokens on the
    HUD font, so we also run a second pass with the preprocessed
    image (binarised + upscaled) and merge the results.
    """
    outs: List[str] = []
    if crop.size == 0:
        return outs
    try:
        res = reader.readtext(crop, detail=0, paragraph=False)
        outs.extend(res)
    except Exception:
        pass
    try:
        pp = _preprocess(crop, upscale=2.0)
        res = reader.readtext(pp, detail=0, paragraph=False)
        outs.extend(res)
    except Exception:
        pass
    return outs


# ---------------------------------------------------------------------- #
# Per-region parsers                                                      #
# ---------------------------------------------------------------------- #

_FLOAT_RE = re.compile(r"[-+]?\d+\.\d+|[-+]?\d+")


def _reinsert_decimal(digits: str, int_part_len: int) -> Optional[float]:
    """Insert a decimal point after `int_part_len` digits in an all-digit
    token. Used to recover lat/lon when OCR drops the decimal dot.
    Returns None if the reconstructed value is out of plausible range.
    """
    digits = digits.lstrip("+").lstrip("-")
    if not digits.isdigit() or len(digits) <= int_part_len:
        return None
    reconstructed = f"{digits[:int_part_len]}.{digits[int_part_len:]}"
    try:
        return float(reconstructed)
    except ValueError:
        return None


def _find_lat_lon(texts: Sequence[str]) -> Tuple[float, float]:
    """Pull LAT/LON values from the GPS region's OCR output.

    Handles three common OCR failure modes:
      1. Clean "LAT: 32.055649" / "LON: 34.888298" text.
      2. Dropped decimal: "LAT: 32055649" / "LON: 34888298" -> reconstruct.
      3. Noise-prefixed "LA1832055649" where the first few chars are
         garbled -> digit-suffix reconstruction.
    """
    joined = " ".join(texts).replace(",", ".")
    lat = lon = float("nan")

    # ---- 1. explicit labelled floats ---------------------------------
    m_lat = re.search(r"(?:LAT|L[A4]T)\D*([-+]?\d+\.\d+)", joined, re.IGNORECASE)
    m_lon = re.search(r"(?:LON|L[O0NG]N?|LNG)\D*([-+]?\d+\.\d+)", joined, re.IGNORECASE)
    if m_lat:
        lat = float(m_lat.group(1))
    if m_lon:
        lon = float(m_lon.group(1))

    # ---- 2. labelled but decimal lost --------------------------------
    if math.isnan(lat):
        m = re.search(r"(?:LAT|L[A4]T)\D*(\d{6,12})", joined, re.IGNORECASE)
        if m:
            for int_len in (2, 1, 3):
                v = _reinsert_decimal(m.group(1), int_len)
                if v is not None and -90.0 <= v <= 90.0:
                    lat = v
                    break
    if math.isnan(lon):
        m = re.search(r"(?:LON|L[O0]N|LNG|L[O0]NG)\D*(\d{6,12})", joined, re.IGNORECASE)
        if m:
            for int_len in (2, 3, 1):
                v = _reinsert_decimal(m.group(1), int_len)
                if v is not None and -180.0 <= v <= 180.0:
                    lon = v
                    break

    # ---- 3. unlabelled fallback (ordered first-two-plausible-floats) --
    if math.isnan(lat) or math.isnan(lon):
        floats = [float(x) for x in _FLOAT_RE.findall(joined) if "." in x]
        for i in range(len(floats) - 1):
            a, b = floats[i], floats[i + 1]
            if -90 <= a <= 90 and -180 <= b <= 180:
                if math.isnan(lat):
                    lat = a
                if math.isnan(lon):
                    lon = b
                break

    return lat, lon


_MAX_PLAUSIBLE_ALT_M = 500.0


def _find_altitude(texts: Sequence[str]) -> Tuple[float, float]:
    """First number in the altitude region is metres-AGL. Third is speed (km/h).

    HUD altitude is rendered as e.g. "21.0 M"; when OCR drops the decimal
    we recover "210" -> we retry with a decimal re-inserted after the
    first 1-3 digits and keep the first value below `_MAX_PLAUSIBLE_ALT_M`.
    Implausible values (>500 m) are reported as NaN.
    """
    joined = " ".join(texts).replace(",", ".")
    floats = [float(x) for x in _FLOAT_RE.findall(joined)]
    alt = floats[0] if len(floats) >= 1 else float("nan")
    speed = floats[2] if len(floats) >= 3 else float("nan")

    if not math.isnan(alt) and alt > _MAX_PLAUSIBLE_ALT_M:
        # Recover from dropped decimal: try "XY.Z...", "X.YZ..." etc.
        digits = str(int(alt))
        recovered = float("nan")
        for int_len in (2, 1, 3):
            v = _reinsert_decimal(digits, int_len)
            if v is not None and 0.0 <= v <= _MAX_PLAUSIBLE_ALT_M:
                recovered = v
                break
        alt = recovered

    return alt, speed


def _find_heading(texts: Sequence[str]) -> float:
    """Compass ribbon: the largest central integer 0..360 is the heading.

    The ribbon renders tick-label digits on the edges (e.g. `120`, `090`);
    to isolate the *current* heading we pick the token whose bounding
    box is closest to the horizontal centre. Since we only get text
    strings here (detail=0), we fall back to "first integer in [0, 359]
    that is not one of the common 30-deg ticks".
    """
    tick_values = {0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360}
    joined = " ".join(texts).replace(",", ".")
    ints: List[int] = []
    for tok in _FLOAT_RE.findall(joined):
        try:
            v = int(float(tok))
        except ValueError:
            continue
        if 0 <= v <= 360:
            ints.append(v)
    # Prefer integers that are NOT standard tick marks.
    non_tick = [v for v in ints if v not in tick_values]
    if non_tick:
        return float(non_tick[0])
    if ints:
        return float(ints[0])
    return float("nan")


def _find_heading_with_detail(reader, crop: np.ndarray) -> float:
    """Detail-mode EasyOCR pass to prefer the *centred* integer on the ribbon."""
    if crop.size == 0:
        return float("nan")
    try:
        results = reader.readtext(crop, detail=1, paragraph=False)
    except Exception:
        return float("nan")
    h, w = crop.shape[:2]
    best_val = float("nan")
    best_dx = float("inf")
    tick_values = {0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360}
    for bbox, text, _conf in results:
        m = re.search(r"\d+", text)
        if not m:
            continue
        try:
            v = int(m.group(0))
        except ValueError:
            continue
        if not (0 <= v <= 360):
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        cx = sum(xs) / 4.0
        cy = sum(ys) / 4.0
        box_h = max(ys) - min(ys)
        dx = abs(cx - w / 2.0)
        # The centred heading is drawn in a larger font than the ticks;
        # penalise short boxes and exact tick values.
        if box_h < 0.35 * h:
            continue
        if v in tick_values:
            dx += w  # heavy penalty, but keep as fallback
        if dx < best_dx:
            best_dx = dx
            best_val = float(v)
    return best_val


def _find_timestamp(texts: Sequence[str]) -> Optional[str]:
    """Pull HH:MM:SS out of the server region. OCR often misreads colons
    as dots or commas, so we accept any non-digit separator and
    normalise to canonical colon form."""
    joined = " ".join(texts)
    m = re.search(r"\b(\d{2})[^\d](\d{2})[^\d](\d{2})\b", joined)
    if m:
        return f"{m.group(1)}:{m.group(2)}:{m.group(3)}"
    return None


# ---------------------------------------------------------------------- #
# Main                                                                    #
# ---------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 7 OCR telemetry extractor.")
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGIONS_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--frames", type=int, nargs="+", default=None,
                        help="Optional subset of frame indices to process.")
    parser.add_argument("--debug-crops", action="store_true",
                        help="Dump per-region crops alongside the OCR text.")
    parser.add_argument("--default-pitch", type=float, default=-45.0,
                        help="default_pitch_deg written into the JSON meta "
                             "(no pitch info in HUD -> this becomes the fallback "
                             "used by pose building). -45 is typical for an "
                             "oblique drone gimbal.")
    parser.add_argument("--default-roll", type=float, default=0.0)
    args = parser.parse_args()

    import easyocr  # heavy; import lazily
    print("Loading EasyOCR (english, CPU) ... first run downloads models.")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    print("EasyOCR ready.")

    regions_list, cal = load_regions_from_json(args.regions)
    region_map = {r.name: {"x": r.x, "y": r.y, "w": r.w, "h": r.h} for r in regions_list}
    for name in ("gps", "altitude", "home", "server"):
        if name not in region_map:
            print(f"WARNING: region '{name}' missing from {args.regions.name}.")

    frames = load_frames(INPUT_FOLDER)
    if args.frames:
        frames = [frames[i] for i in args.frames if 0 <= i < len(frames)]

    if args.debug_crops:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    telemetry: List[FrameTelemetry] = []
    print(f"\nExtracting telemetry from {len(frames)} frames ...\n")
    for idx, f in enumerate(frames):
        img = f.load_image()
        h_img, w_img = img.shape[:2]
        raw: Dict[str, str] = {}
        crops: Dict[str, np.ndarray] = {}
        for name in ("gps", "altitude", "home", "server"):
            rect = region_map.get(name)
            if rect is None:
                continue
            x, y, w, hh = _scale_rect(rect, (w_img, h_img), cal)
            crop = img[y:y + hh, x:x + w]
            crops[name] = crop
            texts = _ocr_texts(reader, crop)
            raw[name] = " | ".join(texts)
            if args.debug_crops:
                cv2.imwrite(str(DEBUG_DIR / f"{f.name}_{name}.png"), crop)

        lat, lon = _find_lat_lon(raw.get("gps", "").split(" | "))
        alt, speed_kmh = _find_altitude(raw.get("altitude", "").split(" | "))
        hd = _find_heading_with_detail(reader, crops.get("home", np.zeros((0, 0, 3), np.uint8)))
        if math.isnan(hd):
            hd = _find_heading(raw.get("home", "").split(" | "))
        ts = _find_timestamp(raw.get("server", "").split(" | "))

        if not math.isnan(speed_kmh):
            raw["_parsed_speed_kmh"] = f"{speed_kmh:.1f}"

        t = FrameTelemetry(
            frame_name=f.name,
            latitude=lat, longitude=lon,
            altitude_m=alt, altitude_is_agl=True,
            heading_deg=hd,
            timestamp=ts,
            raw_ocr=raw,
            notes="ocr:easyocr",
        )
        telemetry.append(t)

        def _fmt(v):
            if isinstance(v, float) and math.isnan(v):
                return "  --  "
            if isinstance(v, float):
                return f"{v:8.4f}"
            return str(v)

        print(f"  [{idx:2d}] {f.name}  "
              f"lat={_fmt(lat)}  lon={_fmt(lon)}  "
              f"alt={_fmt(alt)} m  hdg={_fmt(hd)} deg  "
              f"ts={ts or '-'}")

    ref = pick_reference_origin(telemetry)
    save_frame_telemetry_json(
        telemetry, args.output,
        reference=ref,
        default_pitch_deg=args.default_pitch,
        default_roll_deg=args.default_roll,
    )
    print(f"\nWrote telemetry -> {args.output}")
    if ref is not None:
        print(f"ENU reference   : lat={ref[0]:.6f}  lon={ref[1]:.6f}  alt={ref[2]:.2f}")
    else:
        print("ENU reference   : none (no frame had lat/lon)")

    # Quick coverage summary.
    n = len(telemetry)
    n_gps = sum(1 for t in telemetry if t.has_gps())
    n_alt = sum(1 for t in telemetry if t.has_altitude())
    n_hd  = sum(1 for t in telemetry if t.has_heading())
    print(f"Coverage        : gps {n_gps}/{n}  alt {n_alt}/{n}  heading {n_hd}/{n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
