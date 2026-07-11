"""OpenAI Vision OCR — used ONLY for SEM info-bar / footer text extraction.

Requires OPENAI_API_KEY from .env (loaded via python-dotenv).
Not used for tip geometry, peak detection, or radius fitting.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

log = logging.getLogger("sem-api.stages")

_ENV_LOADED = False


def _ensure_dotenv() -> None:
    """Load backend .env once (safe if already loaded by app.py)."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    try:
        from dotenv import load_dotenv

        root = Path(__file__).resolve().parents[2]  # radius-backend-main/
        load_dotenv(root / ".env", override=False)
    except Exception:
        pass
    _ENV_LOADED = True


_SEM_OCR_PROMPT = """You are extracting ALL visible text and instrument values from a Zeiss / SEM image info bar (footer) or SEM micrograph overlay.

Return a JSON object with:
1) "raw_text": full verbatim OCR of every readable label
2) "values": flat key→value map of every instrument field you can read
   (magnification, working_distance_mm, eht_kv, detector, scale_bar_nm, date, time,
    pixel_size_nm, stage_tilt_deg, file_name, system, and any other labeled fields)

Use numeric values without units in the map when possible; put units in companion keys like "eht_kv_unit".
If a field is missing, omit it. Do not invent values.
Respond with JSON only."""


def openai_api_key() -> str:
    _ensure_dotenv()
    return (os.environ.get("OPENAI_API_KEY") or "").strip()


def openai_configured() -> bool:
    return bool(openai_api_key())


def _encode_png_b64(image: np.ndarray) -> str:
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode image for OpenAI OCR")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _parse_json_content(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    # Fallback: treat whole reply as raw text
    return {"raw_text": content, "values": {}}


def run_openai_sem_ocr(
    image: np.ndarray,
    *,
    model: str | None = None,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """
    Call OpenAI vision on a SEM footer/region image.

    Only invoked when OPENAI_API_KEY is set. Returns same shape as other OCR engines.
    """
    key = openai_api_key()
    if not key:
        return {
            "engine": "openai",
            "ok": False,
            "raw_text": "",
            "lines": [],
            "values": {},
            "errors": ["OPENAI_API_KEY not set"],
        }

    model = (model or os.environ.get("OPENAI_OCR_MODEL") or "gpt-4o-mini").strip()
    b64 = _encode_png_b64(np.asarray(image))

    try:
        from openai import OpenAI

        client = OpenAI(api_key=key, timeout=timeout_s)
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _SEM_OCR_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
        )
        content = resp.choices[0].message.content or ""
        parsed = _parse_json_content(content)
        raw_text = str(parsed.get("raw_text") or content or "").strip()
        values = parsed.get("values") if isinstance(parsed.get("values"), dict) else {}
        lines = [{"text": ln} for ln in raw_text.splitlines() if ln.strip()]
        if not lines and raw_text:
            lines = [{"text": raw_text}]

        ok = bool(raw_text or values)
        log.info(
            "[OCR] OpenAI model=%s ok=%s text_len=%d values=%d",
            model,
            ok,
            len(raw_text),
            len(values or {}),
        )
        return {
            "engine": "openai",
            "ok": ok,
            "raw_text": raw_text,
            "raw_model_output": content,
            "lines": lines,
            "values": values or {},
            "model": model,
        }
    except Exception as exc:
        log.warning("[OCR] OpenAI SEM OCR failed: %s", exc)
        return {
            "engine": "openai",
            "ok": False,
            "raw_text": "",
            "lines": [],
            "values": {},
            "errors": [f"openai: {exc}"],
        }
