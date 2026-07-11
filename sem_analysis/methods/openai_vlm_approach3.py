"""Approach 3 — OpenAI Vision peaks → PDF Methods 1–3 geometry.

  SEM image
    → OpenCV preprocess (done upstream in pipeline)
    → OpenAI: detect all tip peaks (yellow; ignore noise)
    → For each peak (PDF Tip Radius Measurement Brainstorming):
        Method 1 — fixed-distance inscribed circle → R
        Method 2 — projected tip distance → l
        Method 3 — inscribed angle → θ
    → Statistics (mean R, std, peak count)

Contour / free circle-fit remains available as an optional diagnostic overlay
when ``openai_vlm_approach.also_contour_fit`` is true.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import cv2
import numpy as np

from sem_analysis.edge_detection import detect_serration_peaks_global
from sem_analysis.io.openai_ocr import _encode_png_b64, openai_api_key, openai_configured
from sem_analysis.methods.fixed_distance_circle import (
    measure_method1_at_l,
    method1_to_dict,
)
from sem_analysis.methods.inscribed_angle import (
    inscribed_angle_to_dict,
    measure_inscribed_angle,
)
from sem_analysis.methods.projected_tip_distance import (
    measure_projected_tip_distance,
    projected_tip_distance_to_dict,
)
from sem_analysis.protocol import get_protocol
from sem_analysis.qwen_peaks import refine_with_cv
from sem_analysis.radius_computation import fit_circle
from sem_analysis.stats_summary import summarize_values

log = logging.getLogger("sem-api.stages")

PEAK_PROMPT = (
    "This is a SEM micrograph of stacked V-shaped blade / lamella tips.\n"
    "Task: list EVERY visible sharp V-apex (tip of each asperity) in the image.\n"
    "Ignore noise, scratches, and incomplete border fragments.\n"
    "Image size is {width}x{height} pixels. Origin is top-left; x right, y down.\n"
    "Return ONLY JSON of the form:\n"
    '{{"peaks":[{{"id":1,"x":123,"y":456}}, ...]}}\n'
    "Use absolute integer pixel coordinates inside the image bounds.\n"
    "If you see many tips, return up to {max_peaks} of the clearest ones."
)

CONTOUR_PROMPT = (
    "This is a cropped SEM ROI centered on one blade tip apex.\n"
    "Outline the curved apex edge of the tip (the bright/dark boundary of the rounded tip).\n"
    "ROI size is {width}x{height} pixels (origin top-left of THIS crop).\n"
    "Return ONLY JSON:\n"
    '{{"contour":[{{"x":10,"y":20}}, ...], "confidence": 0.0}}\n'
    "Provide at least 8 points along the curved apex. confidence in [0,1]."
)


def _model_name(cfg: dict) -> str:
    return str(
        cfg.get("model")
        or os.environ.get("OPENAI_VLM_MODEL")
        or os.environ.get("OPENAI_OCR_MODEL")
        or "gpt-4o-mini"
    ).strip()


def _parse_json(content: str) -> Any:
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start_obj, end_obj = text.find("{"), text.rfind("}")
        start_arr, end_arr = text.find("["), text.rfind("]")
        if start_obj >= 0 and end_obj > start_obj:
            try:
                return json.loads(text[start_obj : end_obj + 1])
            except json.JSONDecodeError:
                pass
        if start_arr >= 0 and end_arr > start_arr:
            try:
                return json.loads(text[start_arr : end_arr + 1])
            except json.JSONDecodeError:
                pass
    return None


def _openai_vision_json(
    image: np.ndarray,
    prompt: str,
    *,
    model: str,
    timeout_s: float,
    max_tokens: int,
) -> tuple[Any, str | None]:
    """Call OpenAI Vision; return (parsed_json_or_None, error_or_None)."""
    if not openai_configured():
        return None, "OPENAI_API_KEY not set"
    try:
        from openai import OpenAI

        b64 = _encode_png_b64(np.asarray(image))
        client = OpenAI(api_key=openai_api_key(), timeout=timeout_s)
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=max_tokens,
        )
        raw = (resp.choices[0].message.content or "") if resp.choices else ""
        return _parse_json(raw), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _parse_peaks(data: Any, w: int, h: int, coord_space: str) -> list[dict]:
    if data is None:
        return []
    items = None
    if isinstance(data, dict):
        for key in ("peaks", "tips", "apexes", "points", "detections"):
            if isinstance(data.get(key), list):
                items = data[key]
                break
        if items is None:
            # single point object
            if "x" in data and "y" in data:
                items = [data]
    elif isinstance(data, list):
        items = data
    if not isinstance(items, list):
        return []

    out: list[dict] = []
    norm = str(coord_space).lower() in ("normalized_1000", "norm1000", "1000")
    for i, item in enumerate(items):
        x = y = None
        if isinstance(item, dict):
            if "x" in item and "y" in item:
                x, y = item.get("x"), item.get("y")
            elif "cx" in item and "cy" in item:
                x, y = item.get("cx"), item.get("cy")
            elif isinstance(item.get("xy"), (list, tuple)) and len(item["xy"]) >= 2:
                x, y = item["xy"][0], item["xy"][1]
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            x, y = item[0], item[1]
        if x is None or y is None:
            continue
        try:
            xf, yf = float(x), float(y)
        except (TypeError, ValueError):
            continue
        # Auto-detect normalized 0..1 coords
        if (not norm) and 0 <= xf <= 1.0 and 0 <= yf <= 1.0 and max(w, h) > 2:
            xf, yf = xf * w, yf * h
        elif norm:
            xf, yf = xf * w / 1000.0, yf * h / 1000.0
        out.append(
            {
                "id": int(item.get("id", i + 1)) if isinstance(item, dict) else i + 1,
                "x": float(np.clip(xf, 0, w - 1)),
                "y": float(np.clip(yf, 0, h - 1)),
            }
        )
    return out


def _cv_peak_seeds(image: np.ndarray, config: dict | None, max_peaks: int) -> list[dict]:
    """Classical ridge peaks used when OpenAI returns no apexes."""
    try:
        edge = detect_serration_peaks_global(image, config or {})
        locs = np.asarray(edge.peak_locations, dtype=float).reshape(-1, 2)
    except Exception as exc:  # noqa: BLE001
        log.warning("[APPROACH 3] CV peak fallback failed: %s", exc)
        return []
    out: list[dict] = []
    for i, (x, y) in enumerate(locs):
        if max_peaks > 0 and i >= max_peaks:
            break
        out.append({"id": i + 1, "x": float(x), "y": float(y)})
    return out


def detect_apexes_openai(image: np.ndarray, config: dict | None = None) -> tuple[list[dict], dict]:
    """Pass 1 — OpenAI finds all tip peaks on the full (preprocessed) image."""
    cfg = (config or {}).get("openai_vlm_approach", {}) or {}
    meta: dict[str, Any] = {"engine": "openai_vision", "ok": False, "stage": "peaks"}
    if not openai_configured():
        meta["error"] = "OPENAI_API_KEY not set"
        return [], meta

    model = _model_name(cfg)
    timeout_s = float(cfg.get("timeout_s", 120))
    h, w = np.asarray(image).shape[:2]
    max_peaks = int(cfg.get("max_peaks", 40))
    tmpl = str(cfg.get("peak_prompt") or cfg.get("prompt") or PEAK_PROMPT)
    try:
        prompt = tmpl.format(width=w, height=h, max_peaks=max_peaks or 40)
    except (KeyError, ValueError, IndexError):
        prompt = tmpl

    data, err = _openai_vision_json(
        image,
        prompt,
        model=model,
        timeout_s=timeout_s,
        max_tokens=int(cfg.get("max_tokens", 2048)),
    )
    if err:
        meta["error"] = err
        log.warning("[APPROACH 3] OpenAI peak detection failed: %s", err)
        return [], meta

    pts = _parse_peaks(data, w, h, str(cfg.get("coord_space", "absolute")))
    # One retry if empty — models sometimes return {} on first shot
    if not pts:
        data2, err2 = _openai_vision_json(
            image,
            prompt + "\nYou MUST return at least one peak if any tip is visible.",
            model=model,
            timeout_s=timeout_s,
            max_tokens=int(cfg.get("max_tokens", 2048)),
        )
        if not err2:
            pts = _parse_peaks(data2, w, h, str(cfg.get("coord_space", "absolute")))
            meta["retried"] = True

    if max_peaks > 0 and len(pts) > max_peaks:
        pts = pts[:max_peaks]

    meta.update({"ok": True, "model": model, "n_approx": len(pts)})
    log.info("[APPROACH 3] OpenAI peaks: %d (model=%s)", len(pts), model)
    return pts, meta


def _crop_roi(
    image: np.ndarray, cx: float, cy: float, half: int
) -> tuple[np.ndarray, int, int]:
    """Return (crop, x0, y0) with crop clamped to image bounds."""
    h, w = image.shape[:2]
    x0 = int(np.clip(round(cx) - half, 0, max(0, w - 1)))
    y0 = int(np.clip(round(cy) - half, 0, max(0, h - 1)))
    x1 = int(np.clip(round(cx) + half, 1, w))
    y1 = int(np.clip(round(cy) + half, 1, h))
    if x1 <= x0 or y1 <= y0:
        return image[0:1, 0:1].copy(), 0, 0
    return image[y0:y1, x0:x1].copy(), x0, y0


def _parse_contour(data: Any, roi_w: int, roi_h: int) -> tuple[np.ndarray, float]:
    """Parse ROI-local contour points + confidence."""
    if not isinstance(data, dict):
        return np.zeros((0, 2), dtype=float), 0.0
    conf = data.get("confidence", data.get("score", 0.5))
    try:
        confidence = float(np.clip(float(conf), 0.0, 1.0))
    except (TypeError, ValueError):
        confidence = 0.5

    items = data.get("contour") or data.get("polygon") or data.get("points") or []
    if not isinstance(items, list):
        return np.zeros((0, 2), dtype=float), confidence
    pts: list[list[float]] = []
    for item in items:
        if isinstance(item, dict) and "x" in item and "y" in item:
            try:
                pts.append([float(item["x"]), float(item["y"])])
            except (TypeError, ValueError):
                continue
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                pts.append([float(item[0]), float(item[1])])
            except (TypeError, ValueError):
                continue
    if not pts:
        return np.zeros((0, 2), dtype=float), confidence
    arr = np.asarray(pts, dtype=float)
    arr[:, 0] = np.clip(arr[:, 0], 0, max(roi_w - 1, 0))
    arr[:, 1] = np.clip(arr[:, 1], 0, max(roi_h - 1, 0))
    return arr, confidence


def outline_apex_openai(
    roi: np.ndarray,
    config: dict | None = None,
) -> tuple[np.ndarray, float, dict]:
    """Pass 2 — OpenAI outlines the curved apex inside one ROI crop."""
    cfg = (config or {}).get("openai_vlm_approach", {}) or {}
    meta: dict[str, Any] = {"ok": False}
    model = _model_name(cfg)
    rh, rw = roi.shape[:2]
    tmpl = str(cfg.get("contour_prompt") or CONTOUR_PROMPT)
    try:
        prompt = tmpl.format(width=rw, height=rh)
    except (KeyError, ValueError, IndexError):
        prompt = tmpl

    data, err = _openai_vision_json(
        roi,
        prompt,
        model=model,
        timeout_s=float(cfg.get("timeout_s", 120)),
        max_tokens=int(cfg.get("contour_max_tokens", cfg.get("max_tokens", 1024))),
    )
    if err:
        meta["error"] = err
        return np.zeros((0, 2), dtype=float), 0.0, meta

    contour, confidence = _parse_contour(data, rw, rh)
    meta.update({"ok": len(contour) >= 3, "model": model, "n_points": int(len(contour))})
    return contour, confidence, meta


def _fallback_contour_roi(roi: np.ndarray, min_pts: int = 8) -> tuple[np.ndarray, float]:
    """OpenCV-only apex outline when VLM contour is empty — Canny arc near ROI center."""
    gray = _as_gray_u8(roi)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 40, 120)
    ys, xs = np.where(edges > 0)
    if len(xs) < min_pts:
        return np.zeros((0, 2), dtype=float), 0.2
    pts = np.column_stack([xs.astype(float), ys.astype(float)])
    cy, cx = gray.shape[0] / 2.0, gray.shape[1] / 2.0
    # Prefer edge pixels near the tip (upper-middle of ROI for downward-opening V)
    d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    keep = d <= max(8.0, 0.45 * min(gray.shape))
    local = pts[keep] if int(np.sum(keep)) >= min_pts else pts
    # Take the highest (smallest y) band — tip apex
    y_cut = float(np.percentile(local[:, 1], 40))
    tip = local[local[:, 1] <= y_cut]
    if len(tip) < min_pts:
        tip = local
    order = np.argsort(tip[:, 0])
    tip = tip[order]
    # Subsample evenly
    idx = np.linspace(0, len(tip) - 1, num=min(len(tip), max(min_pts, 16)), dtype=int)
    return tip[idx], 0.35


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert float/uint16 SEM arrays to contiguous CV_8U grayscale-ready array."""
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        out = arr
    elif arr.dtype == np.uint16:
        out = (arr / 256).astype(np.uint8)
    else:
        f = arr.astype(np.float32)
        mx = float(np.nanmax(f)) if f.size else 0.0
        if mx <= 1.0 + 1e-6:
            out = (np.clip(f, 0, 1) * 255).astype(np.uint8)
        elif mx <= 255.0 + 1e-3:
            out = np.clip(f, 0, 255).astype(np.uint8)
        else:
            out = (np.clip(f / max(mx, 1e-6), 0, 1) * 255).astype(np.uint8)
    return np.ascontiguousarray(out, dtype=np.uint8)


def _as_gray_u8(image: np.ndarray) -> np.ndarray:
    src = _to_uint8(image)
    if src.ndim == 2:
        return src
    if src.shape[2] == 1:
        return np.ascontiguousarray(src[:, :, 0], dtype=np.uint8)
    return np.ascontiguousarray(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), dtype=np.uint8)


def refine_contour_opencv(
    image: np.ndarray,
    contour_global: np.ndarray,
    *,
    search_px: float = 4.0,
    canny1: int = 40,
    canny2: int = 120,
) -> np.ndarray:
    """Snap VLM polygon vertices to nearby Canny edges; densify along the outline."""
    pts = np.asarray(contour_global, dtype=float).reshape(-1, 2)
    if len(pts) < 3:
        return pts

    try:
        gray = _as_gray_u8(image)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        if blur.dtype != np.uint8:
            blur = np.ascontiguousarray(blur, dtype=np.uint8)
        edges = cv2.Canny(blur, int(canny1), int(canny2))
    except Exception as exc:  # noqa: BLE001
        log.warning("[APPROACH 3] Canny refine skipped: %s", exc)
        return pts

    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        return pts
    edge_xy = np.column_stack([xs.astype(float), ys.astype(float)])

    # Densify polygon for a denser fit set
    dense: list[np.ndarray] = []
    for i in range(len(pts)):
        a = pts[i]
        b = pts[(i + 1) % len(pts)]
        nseg = max(2, int(np.linalg.norm(b - a) / 2.0))
        for t in np.linspace(0, 1, nseg, endpoint=False):
            dense.append(a + t * (b - a))
    dense_arr = np.asarray(dense, dtype=float)

    from scipy.spatial import cKDTree

    tree = cKDTree(edge_xy)
    dists, idxs = tree.query(dense_arr, k=1)
    keep = dists <= float(search_px)
    if int(np.sum(keep)) < 5:
        # Fall back: original vertices only, lightly snapped
        d2, i2 = tree.query(pts, k=1)
        snapped = edge_xy[i2]
        snapped[d2 > search_px] = pts[d2 > search_px]
        return snapped
    return edge_xy[idxs[keep]]


def run_openai_vlm_approach(
    image: np.ndarray,
    nm_per_pixel: float,
    config: dict | None = None,
    tip_seeds: list | np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Approach 3 pipeline (OpenAI instead of Gemini):
      peaks → ROI crop → apex contour → OpenCV refine → circle fit → nm + stats

    tip_seeds: optional Nx2 tip locations (e.g. Method 1 accepted tips) to anchor ROIs.
    """
    cfg = (config or {}).get("openai_vlm_approach", {}) or {}
    fit_method = str(cfg.get("circle_fit", "taubin"))
    half = int(cfg.get("roi_half_size_px", 48))
    min_pts = int(cfg.get("min_contour_points", 8))
    min_conf = float(cfg.get("min_confidence", 0.0))
    search_px = float(cfg.get("edge_search_px", 4.0))
    max_r_px = float(cfg.get("max_radius_px", 80.0))
    min_r_px = float(cfg.get("min_radius_px", 2.0))
    fit_window_px = float(cfg.get("fit_window_px", 28.0))
    max_r_nm = cfg.get("max_radius_nm")
    max_r_nm = float(max_r_nm) if max_r_nm is not None else float(200.0)

    empty: dict[str, Any] = {
        "approach": "openai_vlm_circle_fit",
        "label": "Approach 3 — OpenAI peaks + contour → circle fit",
        "headline": "mean",
        "count": 0,
        "peak_count": 0,
        "per_curve": [],
        "failed_curves": [],
        "median_radius_nm": None,
        "mean_radius_nm": None,
        "std_radius_nm": None,
        "nm_per_pixel": nm_per_pixel,
        "openai": {"ok": False},
    }

    if not bool(cfg.get("enabled", True)):
        empty["openai"] = {"ok": False, "error": "disabled in config"}
        return empty

    img = _to_uint8(image)
    max_peaks = int(cfg.get("max_peaks", 40))
    snap_px = float(cfg.get("snap_px", 80.0))
    min_seed = int(cfg.get("min_seed_peaks", 5))

    # Classical CV peaks — reliable tip locations for Results
    try:
        edge = detect_serration_peaks_global(img, config or {})
        cv_locs = np.asarray(edge.peak_locations, dtype=float).reshape(-1, 2)
    except Exception as exc:  # noqa: BLE001
        log.warning("[APPROACH 3] CV peaks unavailable: %s", exc)
        cv_locs = np.empty((0, 2), dtype=float)

    # Prefer explicit tip seeds (Method 1 tips) so Results points match the dashboard tips
    seed_from_m1 = False
    peaks: list[dict] = []
    peak_source = "cv_ridge_fallback"
    peak_meta: dict[str, Any] = {"engine": "openai_vision", "ok": False, "stage": "peaks"}
    approx: list[dict] = []
    snapped_pts: list[list[float]] = []

    if tip_seeds is not None:
        arr = np.asarray(tip_seeds, dtype=float).reshape(-1, 2)
        if len(arr) > 0:
            if max_peaks > 0:
                arr = arr[:max_peaks]
            peaks = [{"id": i + 1, "x": float(p[0]), "y": float(p[1])} for i, p in enumerate(arr)]
            peak_source = "method1_tips"
            seed_from_m1 = True
            peak_meta = {
                "engine": "openai_vision",
                "ok": True,
                "stage": "peaks",
                "n_approx": 0,
                "n_snapped": 0,
                "n_seeds": len(peaks),
                "seeded_from_method1": True,
                "note": "OpenAI used for ROI contours; tips seeded from Method 1",
            }

    if not seed_from_m1:
        approx, peak_meta = detect_apexes_openai(img, config)
        if approx and len(cv_locs) > 0:
            snapped = refine_with_cv(
                approx,
                cv_locs,
                snap_px,
                keep_unmatched=False,
            )
            snapped_pts = [[float(p[0]), float(p[1])] for p in snapped]
            peak_meta["n_snapped"] = len(snapped_pts)
            peak_meta["snap_px"] = snap_px

        seeds: list[list[float]] = []
        for p in snapped_pts:
            seeds.append(p)
        for p in cv_locs:
            if max_peaks > 0 and len(seeds) >= max_peaks:
                break
            pt = [float(p[0]), float(p[1])]
            if seeds and any(
                np.hypot(pt[0] - s[0], pt[1] - s[1]) < max(3.0, snap_px * 0.35) for s in seeds
            ):
                continue
            seeds.append(pt)

        if len(seeds) < min_seed and len(cv_locs) > 0:
            for p in cv_locs:
                if len(seeds) >= max(min_seed, max_peaks if max_peaks > 0 else min_seed):
                    break
                pt = [float(p[0]), float(p[1])]
                if any(np.hypot(pt[0] - s[0], pt[1] - s[1]) < 3.0 for s in seeds):
                    continue
                seeds.append(pt)

        if max_peaks > 0:
            seeds = seeds[:max_peaks]

        peaks = [{"id": i + 1, "x": s[0], "y": s[1]} for i, s in enumerate(seeds)]
        if snapped_pts and len(snapped_pts) >= min(3, len(peaks)):
            peak_source = "openai_snapped_cv"
        elif snapped_pts:
            peak_source = "openai_snapped_cv+cv_fill"
        else:
            peak_source = "cv_ridge_fallback"
            peak_meta["fell_back_to_cv_peaks"] = True

        peak_meta = {
            **peak_meta,
            "n_approx": len(approx or []),
            "n_seeds": len(peaks),
            "ok": peak_meta.get("ok", False) or bool(peaks),
        }

    log.info(
        "[APPROACH 3] seeds=%d source=%s openai_approx=%d snapped=%d cv=%d",
        len(peaks),
        peak_source,
        len(approx or []),
        len(snapped_pts),
        len(cv_locs),
    )

    empty["openai"] = {**peak_meta, "n_peaks": len(peaks), "peak_source": peak_source}
    empty["peak_count"] = len(peaks)

    if not peaks:
        return empty

    per_curve: list[dict] = []
    failed: list[dict] = []
    contour_ok = 0

    for i, peak in enumerate(peaks):
        px, py = float(peak["x"]), float(peak["y"])
        base: dict[str, Any] = {
            "peak_id": int(peak.get("id", i)),
            "peak_location": [px, py],
            "tip_point": [px, py],
            "approach": "openai_vlm_circle_fit",
            "source": f"{peak_source}+opencv_circle",
        }
        roi, x0, y0 = _crop_roi(img, px, py, half)
        contour_local, confidence, cmeta = outline_apex_openai(roi, config)
        if len(contour_local) < min_pts:
            contour_local, confidence = _fallback_contour_roi(roi, min_pts=min_pts)
            cmeta = {**cmeta, "fell_back_to_cv_contour": True, "ok": len(contour_local) >= min_pts}
        base["vlm_confidence"] = confidence
        base["contour_meta"] = cmeta

        if confidence < min_conf:
            base.update({"valid": False, "rejection_reason": "low_confidence"})
            failed.append(base)
            continue
        if len(contour_local) < min_pts:
            base.update({"valid": False, "rejection_reason": "insufficient_contour"})
            failed.append(base)
            continue

        contour_ok += 1
        contour_global = contour_local.copy()
        contour_global[:, 0] += x0
        contour_global[:, 1] += y0
        refined = refine_contour_opencv(
            img,
            contour_global,
            search_px=search_px,
            canny1=int(cfg.get("canny1", 40)),
            canny2=int(cfg.get("canny2", 120)),
        )
        if len(refined) < 5:
            # Last resort: use CV contour in ROI mapped to global
            fb, _ = _fallback_contour_roi(roi, min_pts=min_pts)
            if len(fb) >= 5:
                refined = fb.copy()
                refined[:, 0] += x0
                refined[:, 1] += y0
            else:
                base.update(
                    {
                        "valid": False,
                        "rejection_reason": "edge_refine_failed",
                        "contour_points": contour_global.tolist(),
                    }
                )
                failed.append(base)
                continue

        center = radius_px = residual = used_method = None
        apex = np.array([px, py], dtype=float)
        d = np.linalg.norm(refined - apex, axis=1)
        local = refined[d <= fit_window_px]
        if len(local) < 5:
            local = refined
        candidates = [local]
        if len(local) >= 8:
            y_cut = float(np.percentile(local[:, 1], 65))
            tip_band = local[local[:, 1] <= y_cut]
            if len(tip_band) >= 5:
                candidates.append(tip_band)

        max_r_px_eff = min(max_r_px, fit_window_px * 2.5)

        for pts_try in candidates:
            for method_try in (fit_method, "pratt", "taubin", "least_squares"):
                try:
                    c, r, res, m = fit_circle(pts_try, method_try)
                except Exception:
                    continue
                if not np.isfinite(r) or r <= 0:
                    continue
                r_nm = float(r * nm_per_pixel)
                if min_r_px <= r <= max_r_px_eff and r_nm <= max_r_nm:
                    center, radius_px, residual, used_method = c, r, res, m
                    break
            if center is not None:
                break

        if center is None or radius_px is None:
            # Retry with pure CV contour if VLM outline led to bad fit
            fb, conf_fb = _fallback_contour_roi(roi, min_pts=min_pts)
            if len(fb) >= 5:
                fb_g = fb.copy()
                fb_g[:, 0] += x0
                fb_g[:, 1] += y0
                d2 = np.linalg.norm(fb_g - apex, axis=1)
                fb_local = fb_g[d2 <= fit_window_px]
                if len(fb_local) < 5:
                    fb_local = fb_g
                try:
                    c, r, res, m = fit_circle(fb_local, "taubin")
                    r_nm = float(r * nm_per_pixel)
                    if min_r_px <= r <= max_r_px_eff and r_nm <= max_r_nm:
                        center, radius_px, residual, used_method = c, r, res, m
                        refined = fb_local
                        local = fb_local
                        base["contour_meta"] = {
                            **cmeta,
                            "fell_back_to_cv_contour_fit": True,
                        }
                        base["vlm_confidence"] = min(float(confidence), conf_fb)
                except Exception:
                    center = None

        if center is None or radius_px is None or not np.isfinite(radius_px):
            base.update(
                {
                    "valid": False,
                    "rejection_reason": "circle_fit_failed",
                    "contour_points": refined.tolist(),
                }
            )
            failed.append(base)
            continue

        radius_nm = float(radius_px * nm_per_pixel)
        if radius_nm > max_r_nm or float(radius_px) > max_r_px_eff:
            base.update(
                {
                    "valid": False,
                    "rejection_reason": "radius_out_of_range",
                    "radius_px": float(radius_px),
                    "radius_nm": radius_nm,
                    "center": [float(center[0]), float(center[1])],
                    "contour_points": refined.tolist(),
                }
            )
            failed.append(base)
            continue

        rec = {
            **base,
            "valid": True,
            "center": [float(center[0]), float(center[1])],
            "radius_px": float(radius_px),
            "radius_nm": radius_nm,
            "fit_residual_px": float(residual) if residual is not None else None,
            "fit_method": used_method,
            "contour_points": local.tolist() if len(local) >= 5 else refined.tolist(),
            "vlm_contour_points": contour_global.tolist(),
            "roi": {"x0": x0, "y0": y0, "half": half},
            "fit_window_px": fit_window_px,
        }
        per_curve.append(rec)

    radii = [c["radius_nm"] for c in per_curve if c.get("radius_nm") is not None]
    stats = summarize_values(radii, headline="mean")
    summary = {
        "approach": "openai_vlm_circle_fit",
        "label": "Approach 3 — OpenAI peaks + contour → circle fit",
        "headline": "mean",
        **stats,
        "median_radius_nm": stats.get("median"),
        "mean_radius_nm": stats.get("mean"),
        "std_radius_nm": stats.get("std"),
        "count": len(per_curve),
        "peak_count": len(peaks),
        "n_marked": len(per_curve) + len(failed),
        "per_curve": per_curve,
        "failed_curves": failed,
        "nm_per_pixel": nm_per_pixel,
        "openai": {
            **peak_meta,
            "n_peaks": len(peaks),
            "n_contours_ok": contour_ok,
            "n_fitted": len(per_curve),
            "circle_fit": fit_method,
            "peak_source": peak_source,
        },
        "params": {
            "roi_half_size_px": half,
            "circle_fit": fit_method,
            "min_contour_points": min_pts,
            "edge_search_px": search_px,
            "min_radius_px": min_r_px,
            "max_radius_px": max_r_px,
        },
    }
    log.info(
        "[APPROACH 3] OpenAI circle-fit: peaks=%d fitted=%d fail=%d mean=%s std=%s",
        len(peaks),
        len(per_curve),
        len(failed),
        summary.get("mean_radius_nm"),
        summary.get("std_radius_nm"),
    )
    return summary
