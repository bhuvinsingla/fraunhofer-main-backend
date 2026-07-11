"""Qwen2.5-VL hybrid V-apex detection.

A vision-language model (Qwen2.5-VL) understands the SEM structure and proposes
*approximate* V-shaped apex coordinates. Those are then snapped to the pipeline's
classical-CV peaks (1D-ridge / edge points) for pixel accuracy:

    Qwen2.5-VL  ->  approximate apex (x, y)  ->  snap to nearest CV peak  ->  Method 1

Runs completely locally/offline once the model weights are downloaded.

Heavy dependencies (torch, transformers, qwen-vl-utils) are imported lazily inside
the functions, so the API keeps working when tip_detection.mode != "qwen" and the
packages are not installed.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any

import numpy as np

log = logging.getLogger("sem-api.stages")

# Cache the loaded model/processor between images (keyed by model id).
_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


DEFAULT_PROMPT = (
    "This is a SEM image of a stack of V-shaped lamellae. "
    "Locate every V-shaped apex (the sharp tip of each layer).\n"
    "Return ONLY JSON, no prose, in exactly this format:\n"
    '[{"id":1,"x":120,"y":240},{"id":2,"x":130,"y":310}]\n'
    "Coordinates are pixel positions in the given image."
)


class QwenUnavailableError(RuntimeError):
    """Raised when Qwen dependencies or weights cannot be loaded."""


def _to_pil_rgb(image: np.ndarray):
    """Convert a grayscale/float/BGR array into a PIL RGB image."""
    from PIL import Image  # local import; pillow is a light dep

    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        # Assume float in [0, 1] (preprocessed) or arbitrary range -> normalise
        a = arr.astype(np.float64)
        amin, amax = float(np.nanmin(a)), float(np.nanmax(a))
        if amax <= 1.0 + 1e-6 and amin >= 0.0:
            a = a * 255.0
        else:
            a = (a - amin) / (amax - amin + 1e-9) * 255.0
        arr = np.clip(a, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        # OpenCV images are BGR; convert to RGB for the VLM
        arr = arr[:, :, ::-1]
    return Image.fromarray(arr)


def _resolve_dtype(torch_dtype: Any):
    import torch

    if torch_dtype is None or torch_dtype == "auto":
        return "auto" if torch.cuda.is_available() else torch.float32
    if isinstance(torch_dtype, str):
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "auto": "auto",
        }
        return mapping.get(torch_dtype.lower(), torch.float32)
    return torch_dtype


def _model_device(model) -> Any:
    """Best-effort device for moving inputs onto the model."""
    import torch

    try:
        for p in model.parameters():
            if p.device.type != "meta":
                return p.device
    except StopIteration:
        pass
    return torch.device("cpu")


def _load_model(cfg: dict) -> tuple[Any, Any]:
    """Load (and cache) the Qwen2.5-VL model + processor. Raises QwenUnavailableError."""
    model_id = str(cfg.get("model_id", "Qwen/Qwen2.5-VL-7B-Instruct"))
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]

    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except Exception as exc:  # noqa: BLE001
        raise QwenUnavailableError(
            "transformers with Qwen2.5-VL support is not installed. "
            "Install optional deps: pip install -r requirements-qwen.txt"
        ) from exc

    import torch

    device_map = cfg.get("device_map", "auto")
    dtype = _resolve_dtype(cfg.get("torch_dtype", "auto"))
    # float16 on CPU often segfaults — force float32 when no CUDA
    if not torch.cuda.is_available() and dtype in (torch.float16, torch.bfloat16):
        log.warning("[QWEN] forcing float32 on CPU (fp16/bf16 unstable without CUDA)")
        dtype = torch.float32
    offload_folder = cfg.get("offload_folder")
    max_memory = cfg.get("max_memory")  # e.g. {"cpu": "3GiB"}
    log.info("[QWEN] loading model %s (device_map=%s dtype=%s)", model_id, device_map, dtype)
    try:
        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        # Low-RAM machines: accelerate can page weights to disk
        if offload_folder:
            from pathlib import Path

            Path(offload_folder).mkdir(parents=True, exist_ok=True)
            kwargs["offload_folder"] = str(offload_folder)
            kwargs["offload_state_dict"] = True
            if max_memory:
                kwargs["max_memory"] = max_memory
            if device_map in (None, "cpu", "none"):
                device_map = "auto"
            kwargs["device_map"] = device_map
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        elif device_map in (None, "cpu", "none"):
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
            model = model.to("cpu")
        else:
            kwargs["device_map"] = device_map
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        model.eval()
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception as exc:  # noqa: BLE001
        raise QwenUnavailableError(f"failed to load Qwen2.5-VL weights: {exc}") from exc

    _MODEL_CACHE[model_id] = (model, processor)
    log.info("[QWEN] model ready on %s", _model_device(model))
    return model, processor


def _parse_points(text: str) -> list[dict]:
    """Extract a JSON list of {id,x,y} from raw model output (tolerant)."""
    if not text:
        return []
    # Grab the outermost [...] block
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    blob = text[start : end + 1]
    try:
        data = json.loads(blob)
    except Exception:  # noqa: BLE001
        # Repair common issues: trailing commas, single quotes
        repaired = re.sub(r",\s*]", "]", blob).replace("'", '"')
        try:
            data = json.loads(repaired)
        except Exception:  # noqa: BLE001
            return []
    out: list[dict] = []
    if isinstance(data, list):
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            if "x" not in item or "y" not in item:
                continue
            try:
                out.append(
                    {"id": int(item.get("id", i + 1)), "x": float(item["x"]), "y": float(item["y"])}
                )
            except (TypeError, ValueError):
                continue
    return out


def detect_apex_coordinates(image: np.ndarray, config: dict) -> list[dict]:
    """Run Qwen2.5-VL to get approximate V-apex coordinates in image pixels.

    Returns a list of {"id", "x", "y"}. Raises QwenUnavailableError if the model
    cannot be used.
    """
    cfg = (config or {}).get("qwen", {}) or {}
    model, processor = _load_model(cfg)

    pil = _to_pil_rgb(image)
    w, h = pil.size
    prompt = str(cfg.get("prompt") or DEFAULT_PROMPT)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        from qwen_vl_utils import process_vision_info
    except Exception as exc:  # noqa: BLE001
        raise QwenUnavailableError(
            "qwen-vl-utils is not installed. pip install -r requirements-qwen.txt"
        ) from exc

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt"
    )
    device = _model_device(model)
    inputs = inputs.to(device)

    import torch

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=int(cfg.get("max_new_tokens", 1024)))
    # Strip the prompt tokens from the generated sequence
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    decoded = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    raw = decoded[0] if decoded else ""
    log.info("[QWEN] raw output (first 300 chars): %s", raw[:300].replace("\n", " "))

    pts = _parse_points(raw)

    # Optional coordinate-space rescale (some Qwen variants emit 0..1000 normalised)
    coord_space = str(cfg.get("coord_space", "absolute")).lower()
    if coord_space in ("normalized_1000", "norm1000", "1000"):
        for p in pts:
            p["x"] = p["x"] * w / 1000.0
            p["y"] = p["y"] * h / 1000.0

    # Clamp into image bounds
    for p in pts:
        p["x"] = float(np.clip(p["x"], 0, w - 1))
        p["y"] = float(np.clip(p["y"], 0, h - 1))

    log.info("[QWEN] parsed %d approximate apexes", len(pts))
    return pts


def refine_with_cv(
    approx_pts: list[dict],
    cv_peaks: np.ndarray,
    snap_px: float,
    *,
    keep_unmatched: bool = True,
) -> np.ndarray:
    """Snap each approximate Qwen apex to the nearest classical-CV peak.

    - If a CV peak lies within snap_px, use its (sub-pixel) coordinates.
    - Otherwise keep the Qwen coordinate (when keep_unmatched) or drop it.
    Deduplicates the resulting apex set.
    """
    cv = np.asarray(cv_peaks, dtype=np.float64).reshape(-1, 2) if cv_peaks is not None else np.empty((0, 2))
    refined: list[list[float]] = []
    for p in approx_pts:
        q = np.array([p["x"], p["y"]], dtype=np.float64)
        if len(cv) > 0:
            d = np.linalg.norm(cv - q, axis=1)
            j = int(np.argmin(d))
            if float(d[j]) <= snap_px:
                refined.append([float(cv[j, 0]), float(cv[j, 1])])
                continue
        if keep_unmatched:
            refined.append([float(q[0]), float(q[1])])

    if not refined:
        return np.empty((0, 2), dtype=np.float64)

    arr = np.array(refined, dtype=np.float64)
    # Dedup within snap_px
    kept: list[np.ndarray] = []
    for pt in arr:
        if all(np.linalg.norm(pt - k) >= max(3.0, snap_px * 0.5) for k in kept):
            kept.append(pt)
    out = np.array(kept, dtype=np.float64)
    if len(out) > 0:
        out = out[np.lexsort((out[:, 1], out[:, 0]))]
    return out


def detect_apex_coordinates_safe(
    image: np.ndarray,
    config: dict,
    *,
    timeout_s: float | None = None,
) -> list[dict]:
    """Run Qwen in a child process so a native crash cannot kill the API.

    Falls back to in-process call only when subprocess cannot start.
    """
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    cfg = (config or {}).get("qwen", {}) or {}
    timeout_s = float(timeout_s if timeout_s is not None else cfg.get("timeout_s", 900))

    # Resolve relative offload folder against backend root
    off = cfg.get("offload_folder")
    if off and not Path(off).is_absolute():
        root = Path(__file__).resolve().parents[1]
        cfg = {**cfg, "offload_folder": str(root / off)}
        config = {**(config or {}), "qwen": cfg}

    with tempfile.TemporaryDirectory(prefix="qwen_job_") as tmp:
        tmp_path = Path(tmp)
        img_path = tmp_path / "image.npy"
        cfg_path = tmp_path / "config.json"
        out_path = tmp_path / "peaks.json"
        np.save(img_path, np.asarray(image))
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"qwen": cfg}, f)

        worker = Path(__file__).with_name("_qwen_worker.py")
        if not worker.exists():
            # Inline fallback: write a tiny worker next to this module if missing
            worker.write_text(
                "import json,sys,numpy as np\n"
                "from pathlib import Path\n"
                "from sem_analysis.qwen_peaks import detect_apex_coordinates\n"
                "img=np.load(sys.argv[1])\n"
                "cfg=json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))\n"
                "pts=detect_apex_coordinates(img,cfg)\n"
                "Path(sys.argv[3]).write_text(json.dumps(pts),encoding='utf-8')\n",
                encoding="utf-8",
            )

        env = os.environ.copy()
        # Keep HF cache / mirror if already set
        cmd = [sys.executable, str(worker), str(img_path), str(cfg_path), str(out_path)]
        log.info("[QWEN] launching isolated worker (timeout=%.0fs)", timeout_s)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=str(Path(__file__).resolve().parents[1]),
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise QwenUnavailableError(f"Qwen worker timed out after {timeout_s}s") from exc

        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "")[-800:]
            raise QwenUnavailableError(
                f"Qwen worker crashed (exit={proc.returncode}): {err}"
            )
        if not out_path.exists():
            raise QwenUnavailableError("Qwen worker finished without writing peaks.json")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise QwenUnavailableError("Qwen worker returned non-list JSON")
        return data


def detect_peaks_qwen_refined(
    image: np.ndarray,
    config: dict,
    cv_peaks: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Full hybrid: Qwen approximate apexes -> CV snap. Falls back to cv_peaks on failure.

    Returns (peaks_Nx2, meta).
    """
    cfg = (config or {}).get("qwen", {}) or {}
    snap_px = float(cfg.get("snap_px", 25.0))
    keep_unmatched = bool(cfg.get("keep_unmatched", True))
    use_subprocess = bool(cfg.get("subprocess", True))
    meta: dict[str, Any] = {"method": "qwen2.5-vl+cv"}

    try:
        if use_subprocess:
            approx = detect_apex_coordinates_safe(image, config)
        else:
            approx = detect_apex_coordinates(image, config)
    except QwenUnavailableError as exc:
        log.warning("[QWEN] unavailable (%s) — falling back to CV ridge peaks", exc)
        meta.update({"status": "unavailable", "error": str(exc), "fell_back": True})
        return np.asarray(cv_peaks, dtype=np.float64).reshape(-1, 2), meta
    except Exception as exc:  # noqa: BLE001
        log.warning("[QWEN] inference error (%s) — falling back to CV ridge peaks", exc)
        meta.update({"status": "error", "error": str(exc), "fell_back": True})
        return np.asarray(cv_peaks, dtype=np.float64).reshape(-1, 2), meta

    refined = refine_with_cv(approx, cv_peaks, snap_px, keep_unmatched=keep_unmatched)
    if len(refined) == 0:
        log.warning("[QWEN] no apexes after refinement — falling back to CV ridge peaks")
        meta.update({"status": "empty", "n_approx": len(approx), "fell_back": True})
        return np.asarray(cv_peaks, dtype=np.float64).reshape(-1, 2), meta

    meta.update(
        {
            "status": "ok",
            "n_approx": len(approx),
            "n_refined": int(len(refined)),
            "snap_px": snap_px,
            "fell_back": False,
        }
    )
    log.info("[QWEN] %d approx -> %d refined apexes (snap_px=%.1f)", len(approx), len(refined), snap_px)
    return refined, meta
