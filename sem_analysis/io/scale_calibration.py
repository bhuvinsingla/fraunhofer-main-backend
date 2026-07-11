"""Automatic SEM scale calibration from Zeiss SmartSEM TIFF tags / footer bar."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile

log = logging.getLogger("sem-api.stages")

# Zeiss SmartSEM private IFD tag with ASCII instrument parameters
ZEISS_SEM_TAG = 34118

_PIXEL_SIZE_PATTERNS = [
    re.compile(r"Image\s+Pixel\s+Size\s*=\s*([0-9]*\.?[0-9]+)\s*(nm|um|µm|μm|pm)", re.I),
    re.compile(r"AP_IMAGE_PIXEL_SIZE[^\n]*?([0-9]*\.?[0-9]+)\s*(nm|um|µm|μm|pm)", re.I),
    re.compile(r"Pixel\s+Size\s*=\s*([0-9]*\.?[0-9]+)\s*(nm|um|µm|μm|pm)", re.I),
    re.compile(r"DP_PIXEL_SIZE[^\n]*?([0-9]*\.?[0-9]+)\s*(nm|um|µm|μm|pm)", re.I),
]

_SCALE_BAR_LABEL_PATTERNS = [
    re.compile(r"Scale\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*nm", re.I),
    re.compile(r"(?<![0-9.])([0-9]*\.?[0-9]+)\s*nm(?![0-9])", re.I),
]


def _unit_to_nm(value: float, unit: str) -> float:
    u = unit.lower().replace("μ", "u").replace("µ", "u")
    if u == "nm":
        return float(value)
    if u in ("um", "µm", "μm"):
        return float(value) * 1000.0
    if u == "pm":
        return float(value) / 1000.0
    return float(value)


def _decode_zeiss_tag_bytes(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        for enc in ("latin-1", "utf-8", "cp1252"):
            try:
                return raw.decode(enc, errors="ignore")
            except Exception:
                continue
        return raw.decode("latin-1", errors="ignore")
    if isinstance(raw, dict):
        # Newer SmartSEM: tag 34118 is a dict of (label, value[, unit]) tuples
        lines = []
        for key, val in raw.items():
            if isinstance(val, (tuple, list)):
                parts = [str(x) for x in val if x is not None and str(x) != ""]
                lines.append(f"{key} = " + " ".join(parts))
            else:
                lines.append(f"{key} = {val}")
        return "\n".join(lines)
    if isinstance(raw, (list, tuple)):
        # Sometimes stored as sequence of ints
        try:
            return bytes(int(x) & 0xFF for x in raw).decode("latin-1", errors="ignore")
        except Exception:
            return " ".join(str(x) for x in raw)
    return str(raw)


def parse_nm_per_pixel_from_zeiss_dict(tag_dict: dict) -> tuple[float | None, dict[str, Any]]:
    """Prefer structured Zeiss dict keys for Image Pixel Size."""
    info: dict[str, Any] = {"format": "dict", "n_keys": len(tag_dict)}
    for key in ("ap_image_pixel_size", "ap_pixel_size", "dp_pixel_size"):
        if key not in tag_dict:
            continue
        val = tag_dict[key]
        # ('Image Pixel Size', 2.233, 'nm') or similar
        if isinstance(val, (tuple, list)) and len(val) >= 2:
            try:
                # find first float-like and optional unit
                num = None
                unit = "nm"
                for item in val[1:]:
                    if isinstance(item, (int, float)):
                        num = float(item)
                    elif isinstance(item, str) and item.lower() in ("nm", "um", "µm", "μm", "pm", "mm"):
                        unit = item
                if num is None and len(val) >= 2:
                    num = float(val[1])
                if num is not None and num > 0:
                    nm = _unit_to_nm(num, unit)
                    info.update({"key": key, "raw": val, "nm_per_pixel": nm})
                    return nm, info
            except (TypeError, ValueError):
                continue
    return None, info


def extract_zeiss_sem_tag(path: Path) -> Any:
    """Return raw Zeiss SmartSEM tag 34118 value (dict or bytes/str)."""
    try:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            tag = page.tags.get(ZEISS_SEM_TAG)
            if tag is None:
                for t in page.tags.values():
                    if getattr(t, "code", None) == ZEISS_SEM_TAG:
                        tag = t
                        break
            if tag is None:
                return None
            return tag.value
    except Exception as exc:
        log.warning("[STAGE 1/4 ANALYZE IMAGE] Zeiss tag read failed: %s", exc)
        return None


def extract_zeiss_sem_ascii(path: Path) -> str:
    """Return Zeiss SmartSEM ASCII metadata blob from TIFF tag 34118 (if present)."""
    return _decode_zeiss_tag_bytes(extract_zeiss_sem_tag(path))


def parse_nm_per_pixel_from_ascii(ascii_blob: str) -> tuple[float | None, dict[str, Any]]:
    """Parse Image Pixel Size from Zeiss ASCII metadata → nm/px."""
    info: dict[str, Any] = {"ascii_chars": len(ascii_blob or "")}
    if not ascii_blob:
        return None, info

    for pat in _PIXEL_SIZE_PATTERNS:
        m = pat.search(ascii_blob)
        if not m:
            continue
        value = float(m.group(1))
        unit = m.group(2)
        nm = _unit_to_nm(value, unit)
        if nm > 0:
            info.update(
                {
                    "matched": m.group(0).strip(),
                    "value": value,
                    "unit": unit,
                    "nm_per_pixel": nm,
                }
            )
            return nm, info
    return None, info


def parse_scale_bar_nm_from_ascii(ascii_blob: str) -> float | None:
    """Best-effort scale-bar label in nm from ASCII (prefer explicit Scale:)."""
    if not ascii_blob:
        return None
    m = _SCALE_BAR_LABEL_PATTERNS[0].search(ascii_blob)
    if m:
        return float(m.group(1))
    # Prefer common SEM bar lengths
    candidates = [float(x) for x in re.findall(r"(?<![0-9.])(50|100|200|500|20|10)\s*nm", ascii_blob, re.I)]
    if candidates:
        return float(candidates[0])
    return None


def measure_footer_scale_bar_pixels(image: np.ndarray, footer_row: int | None = None) -> dict[str, Any]:
    """
    Measure the longest bright horizontal segment in the SEM footer (scale bar).

    Returns dict with scale_bar_pixels and diagnostics.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    h, w = gray.shape[:2]
    if footer_row is None:
        from sem_analysis.roi import detect_footer_row

        footer_row = detect_footer_row(gray)
    footer_row = int(np.clip(footer_row, 0, h - 1))
    footer = gray[footer_row:h, :]
    out: dict[str, Any] = {
        "footer_row": footer_row,
        "footer_h": int(footer.shape[0]),
        "scale_bar_pixels": None,
    }
    if footer.size == 0 or footer.shape[0] < 4:
        return out

    # Bright bar on dark footer
    thr = max(180, int(np.percentile(footer, 92)))
    mask = (footer >= thr).astype(np.uint8) * 255
    # Keep horizontal structures
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 80), 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best_w = 0
    best = None
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if bw < 20 or bh > max(12, footer.shape[0] // 3):
            continue
        aspect = bw / max(bh, 1)
        if aspect < 4:
            continue
        if bw > best_w:
            best_w = int(bw)
            best = {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh), "area": int(area)}

    if best is not None:
        out["scale_bar_pixels"] = float(best["w"])
        out["bar_bbox"] = best
    return out


def _attach_extracted_values(
    result: dict[str, Any],
    path: Path,
    image: np.ndarray | None,
    raw_tag: Any,
    config: dict | None = None,
) -> None:
    """Attach Zeiss-all + unlimited OCR values (does not override nm_per_pixel)."""
    try:
        from sem_analysis.io.footer_ocr import extract_all_sem_values

        extracted = extract_all_sem_values(
            path=path,
            image=image,
            raw_zeiss_tag=raw_tag,
            existing_ocr=result.get("ocr"),
            config=config,
        )
        result["extracted_values"] = extracted.get("combined") or {}
        result["zeiss_all"] = extracted.get("zeiss_all") or {}
        if extracted.get("ocr"):
            result["ocr"] = extracted["ocr"]
        combined = result["extracted_values"]
        if result.get("magnification") is None and combined.get("magnification") is not None:
            result["magnification"] = combined["magnification"]
        if result.get("scale_bar_nm") is None and combined.get("scale_bar_nm") is not None:
            result["scale_bar_nm"] = combined["scale_bar_nm"]
        n_zeiss = len(result["zeiss_all"])
        n_ocr = len((result.get("ocr") or {}).get("values") or {})
        log.info(
            "[STAGE 1/4 ANALYZE IMAGE] extracted values: zeiss_keys=%d ocr_fields=%d combined=%s",
            n_zeiss,
            n_ocr,
            list(combined.keys())[:40],
        )
    except Exception as exc:
        log.warning("[STAGE 1/4 ANALYZE IMAGE] value extraction failed: %s", exc)
        result["extraction_error"] = str(exc)


def calibrate_from_image(
    path: Path,
    image: np.ndarray | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    """
    Auto-calibrate nm/px for a SEM TIFF.

    Priority:
      1) Zeiss tag 34118 Image Pixel Size (dict or ASCII)
      2) Footer OCR scale label + measured scale-bar pixels
      3) Footer scale-bar pixels + scale label from ASCII
    Always attaches full Zeiss + unlimited OCR extracted values.
    """
    path = Path(path)
    result: dict[str, Any] = {
        "nm_per_pixel": None,
        "calibration_source": None,
        "scale_bar_nm": None,
        "scale_bar_pixels": None,
        "zeiss_ascii_ok": False,
    }

    raw_tag = None
    ascii_blob = ""
    if image is None and path.exists():
        try:
            image = tifffile.imread(path)
            if image.ndim == 3:
                image = image[:, :, 0]
        except Exception:
            image = None

    if path.suffix.lower() in (".tif", ".tiff"):
        raw_tag = extract_zeiss_sem_tag(path)
        if isinstance(raw_tag, dict):
            nm, info = parse_nm_per_pixel_from_zeiss_dict(raw_tag)
            result["zeiss_parse"] = info
            result["zeiss_ascii_ok"] = True
            if nm is not None and nm > 0:
                result["nm_per_pixel"] = float(nm)
                result["calibration_source"] = "zeiss_tag_34118"
                mag = raw_tag.get("ap_mag")
                if isinstance(mag, (tuple, list)) and len(mag) >= 2:
                    result["magnification"] = mag[1]
                log.info(
                    "[STAGE 1/4 ANALYZE IMAGE] auto-cal from Zeiss tag: %.6f nm/px (key=%s)",
                    nm,
                    info.get("key"),
                )
                _attach_extracted_values(result, path, image, raw_tag, config=config)
                return result
        ascii_blob = _decode_zeiss_tag_bytes(raw_tag)
        result["zeiss_ascii_ok"] = bool(ascii_blob)
        nm, info = parse_nm_per_pixel_from_ascii(ascii_blob)
        result["zeiss_parse"] = info
        if nm is not None and nm > 0:
            result["nm_per_pixel"] = float(nm)
            result["calibration_source"] = "zeiss_tag_34118"
            log.info(
                "[STAGE 1/4 ANALYZE IMAGE] auto-cal from Zeiss ASCII: %.6f nm/px (%s)",
                nm,
                info.get("matched"),
            )
            _attach_extracted_values(result, path, image, raw_tag, config=config)
            return result

    # Unlimited OCR early so scale_bar_nm can come from visible text
    ocr_scale_nm = None
    if image is not None:
        try:
            from sem_analysis.io.footer_ocr import extract_unlimited_ocr

            ocr = extract_unlimited_ocr(np.asarray(image), config=config)
            result["ocr"] = ocr
            ocr_scale_nm = (ocr.get("values") or {}).get("scale_bar_nm")
            if ocr_scale_nm is not None:
                ocr_scale_nm = float(ocr_scale_nm)
        except Exception as exc:
            log.warning("[STAGE 1/4 ANALYZE IMAGE] unlimited OCR failed: %s", exc)

    scale_nm = ocr_scale_nm if ocr_scale_nm is not None else parse_scale_bar_nm_from_ascii(ascii_blob)
    if scale_nm is None and ascii_blob and re.search(r"50\.00\s*K\s*X|Mag\s*=\s*50", ascii_blob, re.I):
        scale_nm = 50.0
    if scale_nm is None:
        scale_nm = 50.0
        result["scale_bar_nm_assumed"] = True

    result["scale_bar_nm"] = float(scale_nm)

    if image is not None:
        bar = measure_footer_scale_bar_pixels(np.asarray(image))
        result.update({k: v for k, v in bar.items() if k != "scale_bar_pixels" or v is not None})
        px = bar.get("scale_bar_pixels")
        result["scale_bar_pixels"] = px
        if px and px > 0:
            nm_px = float(scale_nm) / float(px)
            result["nm_per_pixel"] = nm_px
            result["calibration_source"] = (
                "footer_ocr_scale_bar" if ocr_scale_nm is not None else "footer_scale_bar"
            )
            log.info(
                "[STAGE 1/4 ANALYZE IMAGE] auto-cal from footer bar: %.6f nm/px "
                "(bar=%.1f nm / %.1f px, ocr_scale=%s)",
                nm_px,
                scale_nm,
                px,
                ocr_scale_nm,
            )
            _attach_extracted_values(result, path, image, raw_tag, config=config)
            return result

    log.warning(
        "[STAGE 1/4 ANALYZE IMAGE] auto-cal failed (no Zeiss pixel size, no footer bar). "
        "ASCII_len=%d scale_nm=%s",
        len(ascii_blob),
        scale_nm,
    )
    _attach_extracted_values(result, path, image, raw_tag, config=config)
    return result
