"""Baidu Unlimited-OCR client (API or local transformers).

https://github.com/baidu/Unlimited-OCR

Priority:
  1) OpenAI-compatible server (UNLIMITED_OCR_URL / config ocr.unlimited_url)
  2) Local transformers model (baidu/Unlimited-OCR) when torch+CUDA available
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

log = logging.getLogger("sem-api.stages")

_REF_RE = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>", re.S)
_DET_RE = re.compile(r"<\|det\|>.*?<\|/det\|>", re.S)
_SPECIAL_RE = re.compile(r"<\|[^|]+\|>")


def clean_unlimited_output(text: str) -> str:
    """Strip grounding tokens; keep readable markdown / plain text."""
    if not text:
        return ""
    refs = _REF_RE.findall(text)
    if refs:
        cleaned = "\n".join(r.strip() for r in refs if r.strip())
    else:
        cleaned = _DET_RE.sub("", text)
        cleaned = _SPECIAL_RE.sub("", cleaned)
    return cleaned.strip()


def _array_to_temp_png(image: np.ndarray) -> Path:
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image
    fd, name = tempfile.mkstemp(suffix=".png", prefix="unlimited_ocr_")
    os.close(fd)
    path = Path(name)
    cv2.imwrite(str(path), bgr)
    return path


def _encode_image_b64(path: Path) -> dict[str, Any]:
    ext = path.suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def run_unlimited_ocr_api(
    image: np.ndarray,
    server_url: str,
    *,
    prompt: str = "<image>document parsing.",
    image_mode: str = "gundam",
    ngram_window: int = 128,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    """Call Unlimited-OCR OpenAI-compatible /v1/chat/completions endpoint."""
    import urllib.request

    tmp = _array_to_temp_png(image)
    try:
        content = [{"type": "text", "text": prompt}, _encode_image_b64(tmp)]
        payload: dict[str, Any] = {
            "model": "Unlimited-OCR",
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 32768,
            "skip_special_tokens": False,
            "stream": False,
        }
        # SGLang-style extras (ignored by plain OpenAI servers)
        payload["images_config"] = {"image_mode": image_mode}
        payload["custom_params"] = {"ngram_size": 35, "window_size": ngram_window}

        url = server_url.rstrip("/") + "/v1/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        raw = body["choices"][0]["message"]["content"]
        text = clean_unlimited_output(raw if isinstance(raw, str) else str(raw))
        return {
            "engine": "unlimited-ocr-api",
            "ok": bool(text.strip()),
            "raw_text": text,
            "raw_model_output": raw if isinstance(raw, str) else str(raw),
            "lines": [{"text": line} for line in text.splitlines() if line.strip()],
            "server_url": server_url.rstrip("/"),
        }
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


_local_model = None
_local_tokenizer = None


def run_unlimited_ocr_local(
    image: np.ndarray,
    *,
    model_name: str = "baidu/Unlimited-OCR",
    prompt: str = "<image>document parsing.",
    max_length: int = 32768,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Run baidu/Unlimited-OCR via transformers (GPU preferred)."""
    global _local_model, _local_tokenizer
    import torch
    from transformers import AutoModel, AutoTokenizer

    tmp = _array_to_temp_png(image)
    out_dir = Path(output_dir or tempfile.mkdtemp(prefix="unlimited_ocr_out_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if _local_tokenizer is None:
            _local_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if _local_model is None:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            _local_model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=dtype,
            )
            _local_model = _local_model.eval()
            if torch.cuda.is_available():
                _local_model = _local_model.cuda()

        result = _local_model.infer(
            _local_tokenizer,
            prompt=prompt,
            image_file=str(tmp),
            output_path=str(out_dir),
            base_size=1024,
            image_size=640,
            crop_mode=True,
            max_length=max_length,
            no_repeat_ngram_size=35,
            ngram_window=128,
            save_results=True,
        )
        raw = ""
        if isinstance(result, str):
            raw = result
        elif result is not None:
            raw = str(result)
        # Also read any saved .md / .txt in output dir
        if not raw.strip():
            for p in sorted(out_dir.glob("*")):
                if p.suffix.lower() in (".md", ".txt", ".mmd"):
                    raw = p.read_text(encoding="utf-8", errors="ignore")
                    if raw.strip():
                        break
        text = clean_unlimited_output(raw)
        return {
            "engine": "unlimited-ocr-local",
            "ok": bool(text.strip()),
            "raw_text": text,
            "raw_model_output": raw,
            "lines": [{"text": line} for line in text.splitlines() if line.strip()],
            "model": model_name,
        }
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def run_unlimited_ocr(
    image: np.ndarray,
    config: dict | None = None,
) -> dict[str, Any]:
    """
    Unlimited-OCR entrypoint.

    Config keys under ocr.* :
      unlimited_url, unlimited_model, unlimited_prompt, max_length
    Env: UNLIMITED_OCR_URL overrides config URL.
    """
    cfg = (config or {}).get("ocr", {}) if config else {}
    url = (os.environ.get("UNLIMITED_OCR_URL") or cfg.get("unlimited_url") or "").strip()
    model_name = cfg.get("unlimited_model") or "baidu/Unlimited-OCR"
    prompt = cfg.get("unlimited_prompt") or "<image>document parsing."
    max_length = int(cfg.get("max_length", 32768))
    errors: list[str] = []

    if url:
        try:
            out = run_unlimited_ocr_api(
                image,
                url,
                prompt=prompt,
                image_mode=str(cfg.get("image_mode", "gundam")),
                ngram_window=int(cfg.get("ngram_window", 128)),
                timeout_s=float(cfg.get("timeout_s", 600)),
            )
            if out.get("ok"):
                log.info("[OCR] Unlimited-OCR API ok text_len=%d", len(out.get("raw_text") or ""))
                return out
            errors.append("unlimited_api_empty")
        except Exception as exc:
            errors.append(f"unlimited_api: {exc}")
            log.warning("[OCR] Unlimited-OCR API failed: %s", exc)

    # Local transformers (optional heavy path)
    try:
        import torch  # noqa: F401
        from transformers import AutoModel  # noqa: F401

        out = run_unlimited_ocr_local(
            image,
            model_name=model_name,
            prompt=prompt,
            max_length=max_length,
        )
        if out.get("ok"):
            log.info("[OCR] Unlimited-OCR local ok text_len=%d", len(out.get("raw_text") or ""))
            return out
        errors.append("unlimited_local_empty")
    except Exception as exc:
        errors.append(f"unlimited_local: {exc}")
        log.info("[OCR] Unlimited-OCR local unavailable: %s", exc)

    return {
        "engine": "unlimited-ocr",
        "ok": False,
        "raw_text": "",
        "lines": [],
        "errors": errors,
    }
