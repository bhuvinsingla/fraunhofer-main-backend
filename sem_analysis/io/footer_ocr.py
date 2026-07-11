"""OCR SEM info-bar / footer and extract all visible instrument values.

Uses RapidOCR (ONNX, no system Tesseract required). Falls back to pytesseract
when available. Also flattens Zeiss SmartSEM tag 34118 into key/value pairs.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import cv2
import numpy as np

log = logging.getLogger("sem-api.stages")

# Common Zeiss / SEM footer labels → normalized keys
_FIELD_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("magnification", re.compile(r"(?:Mag(?:nification)?)\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*([kKmM])?\s*[xX]", re.I)),
    ("magnification", re.compile(r"([0-9]*\.?[0-9]+)\s*([kKmM])\s*[xX]\b")),
    ("scale_bar_nm", re.compile(r"(?:Scale|Signal\s*A)?\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*nm\b", re.I)),
    ("scale_bar_um", re.compile(r"(?:Scale)?\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*(?:um|µm|μm)\b", re.I)),
    ("working_distance_mm", re.compile(r"(?:WD|Working\s*Distance)\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*mm", re.I)),
    ("eht_kv", re.compile(r"(?:EHT|HV|High\s*Voltage)\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*kV", re.I)),
    ("detector", re.compile(r"(?:Detector|Signal\s*A)\s*[:=]?\s*([A-Za-z0-9_\-\+\.\/ ]+?)(?:\s{2,}|\s+(?:Mag|WD|EHT|Date|Time|File)|$)", re.I)),
    ("date", re.compile(r"(?:Date)\s*[:=]?\s*([0-9]{1,4}[-./][0-9]{1,2}[-./][0-9]{1,4})", re.I)),
    ("time", re.compile(r"(?:Time)\s*[:=]?\s*([0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?)", re.I)),
    ("pixel_size_nm", re.compile(r"(?:Image\s+)?Pixel\s*Size\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*nm", re.I)),
    ("pixel_size_um", re.compile(r"(?:Image\s+)?Pixel\s*Size\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*(?:um|µm|μm)", re.I)),
    ("stage_tilt_deg", re.compile(r"(?:Stage\s*)?Tilt\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*°?", re.I)),
    ("stage_x_mm", re.compile(r"Stage\s*X\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*mm", re.I)),
    ("stage_y_mm", re.compile(r"Stage\s*Y\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*mm", re.I)),
    ("stage_z_mm", re.compile(r"Stage\s*Z\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*mm", re.I)),
    ("aperture_um", re.compile(r"(?:Aperture|Apert\.?)\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*(?:um|µm|μm)", re.I)),
    ("spot_size", re.compile(r"(?:Spot|I\s*Probe)\s*[:=]?\s*([0-9]*\.?[0-9]+)", re.I)),
    ("dwell_time_us", re.compile(r"(?:Dwell|Noise\s*Reduction)\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*(?:us|µs|μs)", re.I)),
    ("file_name", re.compile(r"(?:File\s*Name|Filename)\s*[:=]?\s*([^\s]+)", re.I)),
    ("system", re.compile(r"\b(SUPRA|SIGMA|Merlin|Gemini|Ultra|Crossbeam|FIB)\b", re.I)),
]

_MAG_MULT = {"k": 1_000.0, "K": 1_000.0, "m": 1_000_000.0, "M": 1_000_000.0}

_ZEISS_KEY_MAP = {
    "ap_mag": "magnification",
    "ap_wd": "working_distance_mm",
    "ap_eht": "eht_kv",
    "ap_detector": "detector",
    "ap_date": "date",
    "ap_time": "time",
    "ap_image_pixel_size": "pixel_size_nm",
    "ap_pixel_size": "pixel_size_nm",
    "dp_pixel_size": "pixel_size_nm",
    "ap_stage_at_x": "stage_x_mm",
    "ap_stage_at_y": "stage_y_mm",
    "ap_stage_at_z": "stage_z_mm",
    "ap_stage_at_t": "stage_tilt_deg",
    "ap_stage_at_r": "stage_rotation_deg",
    "ap_filename": "file_name",
    "ap_system": "system",
    "dp_detector_channel": "detector",
    "ap_free_text": "free_text",
}


def flatten_zeiss_tag(raw_tag: Any) -> dict[str, Any]:
    """Flatten Zeiss tag 34118 (dict / ASCII / bytes) into {key: value} pairs."""
    out: dict[str, Any] = {}
    if raw_tag is None:
        return out

    if isinstance(raw_tag, dict):
        for key, val in raw_tag.items():
            k = str(key)
            if isinstance(val, (tuple, list)):
                label = None
                num = None
                unit = None
                text_parts: list[str] = []
                for i, item in enumerate(val):
                    if i == 0 and isinstance(item, str) and not item.replace(".", "", 1).isdigit():
                        label = item
                        continue
                    if isinstance(item, (int, float)):
                        num = float(item)
                    elif isinstance(item, str):
                        if item.lower() in ("nm", "um", "µm", "μm", "pm", "mm", "kv", "deg", "°", "kx", "x"):
                            unit = item
                        else:
                            text_parts.append(item)
                entry: dict[str, Any] = {}
                if label:
                    entry["label"] = label
                if num is not None:
                    entry["value"] = num
                if unit:
                    entry["unit"] = unit
                if text_parts:
                    entry["text"] = " ".join(text_parts)
                if not entry and val:
                    entry["raw"] = list(val)
                out[k] = entry if entry else val
            else:
                out[k] = val
        return out

    from sem_analysis.io.scale_calibration import _decode_zeiss_tag_bytes

    ascii_blob = _decode_zeiss_tag_bytes(raw_tag)
    for line in ascii_blob.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        left, right = line.split("=", 1)
        key = left.strip()
        val = right.strip()
        if key:
            out[key] = val
    return out


def zeiss_values_normalized(flat: dict[str, Any]) -> dict[str, Any]:
    """Map known Zeiss keys to normalized calibration field names."""
    values: dict[str, Any] = {}
    for src, dst in _ZEISS_KEY_MAP.items():
        if src not in flat:
            continue
        entry = flat[src]
        if isinstance(entry, dict):
            if "value" in entry:
                val = entry["value"]
                unit = (entry.get("unit") or "").strip()
                # Mag often stored as 50.0 with unit "K X"
                if dst == "magnification" and unit:
                    u = unit.lower().replace(" ", "")
                    if u.startswith("k"):
                        val = float(val) * 1000.0
                        values["magnification_display"] = f"{entry['value']} K X"
                    elif u.startswith("m") and "mm" not in u:
                        val = float(val) * 1_000_000.0
                values[dst] = val
                if unit:
                    values[f"{dst}_unit"] = unit
            elif "text" in entry:
                values[dst] = entry["text"]
            elif "label" in entry and len(entry) == 1:
                values[dst] = entry["label"]
            else:
                values[dst] = entry
        else:
            values[dst] = entry
    return values


def _prepare_footer_for_ocr(footer: np.ndarray) -> np.ndarray:
    """Upscale + contrast boost for small SEM footer text."""
    if footer.ndim == 3:
        gray = cv2.cvtColor(footer, cv2.COLOR_BGR2GRAY)
    else:
        gray = footer.copy()
    h, w = gray.shape[:2]
    scale = 3 if max(h, w) < 1200 else 2
    up = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    if float(np.mean(up)) < 80:
        up = cv2.normalize(up, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        up = clahe.apply(up)
    else:
        up = cv2.bitwise_not(up)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        up = clahe.apply(up)
    return up


def _run_rapidocr(image: np.ndarray) -> tuple[str, list[dict[str, Any]]]:
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    result, _ = engine(image)
    lines: list[dict[str, Any]] = []
    texts: list[str] = []
    if not result:
        return "", lines
    for item in result:
        if len(item) < 2:
            continue
        text = str(item[1]).strip()
        score = float(item[2]) if len(item) > 2 else None
        if not text:
            continue
        texts.append(text)
        lines.append({"text": text, "confidence": score, "box": item[0]})
    return " ".join(texts), lines


def _run_pytesseract(image: np.ndarray) -> tuple[str, list[dict[str, Any]]]:
    import pytesseract

    config = "--psm 6"
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=config)
    lines: list[dict[str, Any]] = []
    texts: list[str] = []
    n = len(data.get("text") or [])
    for i in range(n):
        text = str(data["text"][i]).strip()
        if not text:
            continue
        conf = float(data["conf"][i]) if data["conf"][i] not in ("-1", -1) else None
        texts.append(text)
        lines.append(
            {
                "text": text,
                "confidence": conf,
                "box": [
                    int(data["left"][i]),
                    int(data["top"][i]),
                    int(data["width"][i]),
                    int(data["height"][i]),
                ],
            }
        )
    return " ".join(texts), lines


def parse_ocr_text(text: str) -> dict[str, Any]:
    """Extract ALL values from OCR text — known SEM fields + every Key=Value pair."""
    values: dict[str, Any] = {}
    if not text:
        return values

    cleaned = (
        text.replace("µ", "u")
        .replace("μ", "u")
        .replace("×", "x")
        .replace("—", "-")
        .replace("|", " ")
    )
    # Keep newlines for KV parsing; also a collapsed form for regex fields
    collapsed = re.sub(r"\s+", " ", cleaned)

    # Unlimited: every Key: Value / Key = Value on its own line or inline
    for m in re.finditer(
        r"(?m)^\s*([A-Za-z][A-Za-z0-9_\-\s\.]{0,60}?)\s*[:=]\s*(.+?)\s*$",
        cleaned,
    ):
        key = re.sub(r"\s+", "_", m.group(1).strip().lower())
        key = re.sub(r"[^\w\-]+", "", key)
        val = m.group(2).strip()
        if key and val and key not in values:
            values[key] = val

    for m in re.finditer(
        r"([A-Za-z][A-Za-z0-9_\-]{1,40})\s*[:=]\s*([0-9]*\.?[0-9]+(?:\s*[A-Za-zµμ°/%]+)?)",
        collapsed,
    ):
        key = m.group(1).strip().lower()
        if key not in values:
            values[key] = m.group(2).strip()

    for key, pat in _FIELD_PATTERNS:
        m = pat.search(collapsed)
        if not m:
            continue
        if key == "magnification":
            num = float(m.group(1))
            mult = m.group(2) if m.lastindex and m.lastindex >= 2 else None
            if mult:
                values["magnification"] = num * _MAG_MULT.get(mult, 1.0)
                values["magnification_display"] = f"{num} {mult.upper()} X"
            else:
                values["magnification"] = num
                values["magnification_display"] = f"{num} X"
        elif key in ("detector", "date", "time", "file_name", "system"):
            values[key] = m.group(1).strip()
        elif key == "scale_bar_um":
            values["scale_bar_nm"] = float(m.group(1)) * 1000.0
        elif key == "pixel_size_um":
            values["pixel_size_nm"] = float(m.group(1)) * 1000.0
        else:
            try:
                values[key] = float(m.group(1))
            except (TypeError, ValueError):
                values[key] = m.group(1).strip()

    if "scale_bar_nm" not in values:
        m = re.search(r"(?<![0-9.])(10|20|50|100|200|500)\s*nm\b", collapsed, re.I)
        if m:
            values["scale_bar_nm"] = float(m.group(1))

    # Preserve full OCR dump (unlimited)
    values["_ocr_raw_text"] = text
    values["_ocr_line_count"] = len([ln for ln in text.splitlines() if ln.strip()])
    return values


def ocr_image_region(image: np.ndarray, prepare: bool = True) -> dict[str, Any]:
    """Run OCR on a region. Prefer RapidOCR, then pytesseract. No line/char caps."""
    prepared = _prepare_footer_for_ocr(image) if prepare else (
        image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    )
    errors: list[str] = []

    try:
        text, lines = _run_rapidocr(prepared)
        if text.strip():
            return {
                "engine": "rapidocr",
                "raw_text": text,
                "lines": lines,  # all lines, unlimited
                "ok": True,
            }
        errors.append("rapidocr_empty")
    except Exception as exc:
        errors.append(f"rapidocr: {exc}")
        log.warning("[OCR] RapidOCR failed: %s", exc)

    try:
        text, lines = _run_pytesseract(prepared)
        if text.strip():
            return {
                "engine": "pytesseract",
                "raw_text": text,
                "lines": lines,
                "ok": True,
            }
        errors.append("pytesseract_empty")
    except Exception as exc:
        errors.append(f"pytesseract: {exc}")
        log.warning("[OCR] pytesseract failed: %s", exc)

    return {
        "engine": None,
        "raw_text": "",
        "lines": [],
        "ok": False,
        "errors": errors,
    }


def extract_unlimited_ocr(
    image: np.ndarray,
    config: dict | None = None,
    footer_row: int | None = None,
) -> dict[str, Any]:
    """
    Unlimited OCR: OpenAI Vision (footer only, if OPENAI_API_KEY set) →
    Baidu Unlimited-OCR → RapidOCR full/footer.

    Returns every detected line and every parsed key/value with no caps.
    OpenAI is used ONLY for SEM info-bar text extraction.
    """
    from sem_analysis.roi import detect_footer_row

    cfg = (config or {}).get("ocr", {}) if config else {}
    engine_pref = str(cfg.get("engine", "unlimited")).lower()  # unlimited | rapid | auto | openai
    region = str(cfg.get("region", "full")).lower()  # full | footer | both

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = np.asarray(image)
    h = gray.shape[0]
    if footer_row is None:
        footer_row = detect_footer_row(gray)
    footer_row = int(np.clip(footer_row, 0, h - 1))
    y0 = max(0, footer_row - 4)
    footer = gray[y0:h, :]

    errors: list[str] = []
    primary: dict[str, Any] | None = None
    openai_values: dict[str, Any] = {}

    # 0) OpenAI Vision — ONLY SEM footer/info-bar (requires OPENAI_API_KEY in .env)
    use_openai = engine_pref in ("openai", "unlimited", "auto") and bool(cfg.get("use_openai", True))
    if use_openai:
        try:
            from sem_analysis.io.openai_ocr import openai_configured, run_openai_sem_ocr

            if openai_configured():
                oai = run_openai_sem_ocr(
                    footer,
                    model=cfg.get("openai_model") or None,
                    timeout_s=float(cfg.get("openai_timeout_s", 120)),
                )
                if oai.get("ok"):
                    primary = oai
                    openai_values = dict(oai.get("values") or {})
                else:
                    errors.extend(oai.get("errors") or ["openai_empty"])
            elif engine_pref == "openai":
                errors.append("OPENAI_API_KEY not set in .env")
        except Exception as exc:
            errors.append(f"openai: {exc}")
            log.warning("[OCR] OpenAI footer OCR failed: %s", exc)

    # 1) Baidu Unlimited-OCR (full image) — optional local/server path
    if primary is None and engine_pref in ("unlimited", "auto"):
        try:
            from sem_analysis.io.unlimited_ocr import run_unlimited_ocr

            u = run_unlimited_ocr(gray, config=config)
            if u.get("ok"):
                primary = u
            else:
                errors.extend(u.get("errors") or ["unlimited_failed"])
        except Exception as exc:
            errors.append(f"unlimited: {exc}")
            log.warning("[OCR] Unlimited-OCR path failed: %s", exc)

    # 2) RapidOCR / tesseract — full image and/or footer (unlimited lines)
    rapid_parts: list[dict[str, Any]] = []
    if primary is None or engine_pref in ("rapid", "auto", "unlimited", "openai"):
        regions: list[tuple[str, np.ndarray]] = []
        if region in ("full", "both") or primary is None:
            regions.append(("full", gray))
        if region in ("footer", "both") or primary is None:
            regions.append(("footer", footer))

        seen_text: set[str] = set()
        all_lines: list[dict[str, Any]] = []
        text_chunks: list[str] = []
        eng = None
        for name, roi in regions:
            if roi.size == 0:
                continue
            o = ocr_image_region(roi, prepare=True)
            if not o.get("ok"):
                errors.extend(o.get("errors") or [])
                continue
            eng = o.get("engine") or eng
            chunk = (o.get("raw_text") or "").strip()
            if chunk:
                text_chunks.append(f"[{name}] {chunk}")
            for line in o.get("lines") or []:
                t = (line.get("text") or "").strip()
                if not t:
                    continue
                dedup_key = f"{name}:{t}"
                if dedup_key in seen_text:
                    continue
                seen_text.add(dedup_key)
                all_lines.append({**line, "region": name})

        if text_chunks:
            rapid = {
                "engine": eng or "rapidocr",
                "ok": True,
                "raw_text": "\n".join(text_chunks),
                "lines": all_lines,
                "regions": [n for n, _ in regions],
            }
            rapid_parts.append(rapid)
            if primary is None:
                primary = rapid
            else:
                # Merge RapidOCR lines into primary (no truncation)
                primary = dict(primary)
                primary["rapidocr_supplement"] = rapid
                merged_lines = list(primary.get("lines") or []) + all_lines
                primary["lines"] = merged_lines
                extra = "\n".join(text_chunks)
                if extra and extra not in (primary.get("raw_text") or ""):
                    primary["raw_text"] = ((primary.get("raw_text") or "") + "\n" + extra).strip()

    if primary is None:
        primary = {
            "engine": None,
            "ok": False,
            "raw_text": "",
            "lines": [],
            "errors": errors,
        }

    values = parse_ocr_text(primary.get("raw_text") or "")
    # OpenAI structured values win over regex parse for the same keys
    if openai_values:
        values = {**values, **openai_values}
    elif primary.get("engine") == "openai" and primary.get("values"):
        values = {**values, **(primary.get("values") or {})}

    # Also index every OCR line as line_001, line_002, … (unlimited)
    for i, line in enumerate(primary.get("lines") or [], start=1):
        t = (line.get("text") if isinstance(line, dict) else str(line) or "").strip()
        if t:
            values[f"line_{i:03d}"] = t

    result = {
        "footer_row": footer_row,
        "footer_h": int(h - footer_row),
        "engine": primary.get("engine"),
        "ok": bool(primary.get("ok")),
        "raw_text": primary.get("raw_text") or "",
        "raw_model_output": primary.get("raw_model_output"),
        "lines": primary.get("lines") or [],
        "values": values,
        "errors": errors or primary.get("errors"),
        "mode": "unlimited",
        "rapidocr_supplement": primary.get("rapidocr_supplement"),
    }
    log.info(
        "[OCR] unlimited engine=%s ok=%s lines=%d values=%d text_len=%d",
        result["engine"],
        result["ok"],
        len(result["lines"]),
        len(values),
        len(result["raw_text"]),
    )
    return result


def extract_footer_ocr(
    image: np.ndarray,
    footer_row: int | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    """Backward-compatible entry — runs unlimited OCR (full image + footer)."""
    return extract_unlimited_ocr(image, config=config, footer_row=footer_row)


def extract_all_sem_values(
    path=None,
    image: np.ndarray | None = None,
    raw_zeiss_tag: Any = None,
    existing_ocr: dict[str, Any] | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    """
    Extract all available SEM values from Zeiss TIFF tag + unlimited OCR.

    Priority for overlapping keys: Zeiss tag > OCR (tag is authoritative).
    """
    from pathlib import Path

    from sem_analysis.io.scale_calibration import extract_zeiss_sem_tag

    result: dict[str, Any] = {
        "zeiss_all": {},
        "zeiss_normalized": {},
        "ocr": None,
        "combined": {},
    }

    if raw_zeiss_tag is None and path is not None:
        p = Path(path)
        if p.suffix.lower() in (".tif", ".tiff"):
            raw_zeiss_tag = extract_zeiss_sem_tag(p)

    if raw_zeiss_tag is not None:
        flat = flatten_zeiss_tag(raw_zeiss_tag)
        result["zeiss_all"] = flat
        result["zeiss_normalized"] = zeiss_values_normalized(flat)

    if existing_ocr is not None:
        result["ocr"] = existing_ocr
    elif image is not None:
        result["ocr"] = extract_unlimited_ocr(np.asarray(image), config=config)

    combined: dict[str, Any] = {}
    if result.get("ocr") and result["ocr"].get("values"):
        # Skip internal dump keys from polluting combined headline fields
        for k, v in (result["ocr"]["values"] or {}).items():
            if k.startswith("_"):
                continue
            combined[k] = v
    combined.update(result.get("zeiss_normalized") or {})
    result["combined"] = combined
    return result
