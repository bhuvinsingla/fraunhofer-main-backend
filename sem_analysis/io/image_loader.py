"""Multi-format SEM image I/O with metadata parsing."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image


class ResizedImageError(ValueError):
    """Raised when a post-export resize is flagged without a known scale factor."""


@dataclass
class SEMImage:
    """Loaded SEM image with calibration metadata."""

    data: np.ndarray
    nm_per_pixel: float
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)
    bit_depth: int = 8
    nm_per_pixel_raw: float | None = None
    tilt_correction: dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]


def tilt_scale_factor(tilt_angle_deg: float) -> float:
    """Foreshortening correction: L_true = L_image / cos(θ)."""
    cos_t = math.cos(math.radians(tilt_angle_deg))
    if abs(cos_t) < 1e-9:
        raise ValueError(f"Invalid tilt angle {tilt_angle_deg}° (cos ≈ 0)")
    return 1.0 / cos_t


# Default: do NOT apply blind foreshortening scale (store metadata only)
_DEFAULT_TILT = {
    "enabled": False,
    "tilt_angle_deg": 60.0,
    "warning": (
        "Stage tilt metadata is recorded but measurements are NOT scaled by "
        "1/cos(tilt). Report projected_* values until SEM geometry is confirmed."
    ),
}


def apply_tilt_correction(
    nm_per_pixel: float,
    config: dict | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Optionally scale nm/px for specimen tilt.

    Default: disabled. Blind 1/cos(60°)=2× is unsafe without confirmed
    measurement plane / tilt axis / scan orientation.
    """
    cal = (config or {}).get("calibration", {})
    tilt_cfg = {**_DEFAULT_TILT, **(cal.get("tilt_correction") or {})}
    angle = float(tilt_cfg.get("tilt_angle_deg", 60.0))
    info = {
        "enabled": bool(tilt_cfg.get("enabled", False)),
        "applied": False,
        "tilt_angle_deg": angle,
        "cos_tilt": float(math.cos(math.radians(angle))),
        "scale_factor": 1.0,
        "nm_per_pixel_raw": float(nm_per_pixel),
        "nm_per_pixel_corrected": float(nm_per_pixel),
        "formula": "none — projected measurements only",
        "warning": (
            tilt_cfg.get("warning")
            or (
                f"Stage tilt {angle:g}° recorded; no blind foreshortening correction applied. "
                "Values are projected_* until geometry is confirmed."
            )
        ).strip(),
    }
    if not tilt_cfg.get("enabled", False):
        return nm_per_pixel, info

    factor = tilt_scale_factor(angle)
    corrected = nm_per_pixel * factor
    info.update({
        "applied": True,
        "scale_factor": float(factor),
        "nm_per_pixel_corrected": float(corrected),
        "formula": "L_true = L_image / cos(tilt_angle)",
    })
    return corrected, info


def _load_companion_calibration_record(path: Path) -> dict[str, Any] | None:
    """Load full calibration record from companion JSON/CSV next to the image."""
    stem = path.stem
    parent = path.parent

    json_path = parent / f"{stem}_calibration.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        record = dict(data)
        if "nm_per_pixel" not in record and "pixel_size_nm" in record:
            record["nm_per_pixel"] = record["pixel_size_nm"]
        if "scale_bar_nm" in record and "scale_bar_pixels" in record:
            sb_nm = float(record["scale_bar_nm"])
            sb_px = float(record["scale_bar_pixels"])
            if sb_px > 0:
                record["nm_per_pixel"] = sb_nm / sb_px
                record["calibration_source"] = record.get("calibration_source", "scale_bar")
        return record

    csv_path = parent / f"{stem}_calibration.csv"
    if csv_path.exists():
        import pandas as pd

        df = pd.read_csv(csv_path)
        row = df.iloc[0].to_dict()
        record: dict[str, Any] = {}
        for col in ("nm_per_pixel", "pixel_size_nm", "scale_bar_nm", "scale_bar_pixels", "magnification"):
            if col in row and row[col] == row[col]:  # not NaN
                record[col] = row[col]
        if "nm_per_pixel" not in record and "pixel_size_nm" in record:
            record["nm_per_pixel"] = float(record["pixel_size_nm"])
        if "scale_bar_nm" in record and "scale_bar_pixels" in record:
            sb_px = float(record["scale_bar_pixels"])
            if sb_px > 0:
                record["nm_per_pixel"] = float(record["scale_bar_nm"]) / sb_px
                record["calibration_source"] = "scale_bar"
        return record or None

    return None


def _load_companion_calibration(path: Path) -> float | None:
    """Load nm/px from companion JSON or CSV next to the image."""
    record = _load_companion_calibration_record(path)
    if record is None:
        return None
    if "nm_per_pixel" in record:
        return float(record["nm_per_pixel"])
    return None


def build_calibration_record(
    sem_image: SEMImage,
    tilt_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Phase-1 calibration record for API / audit trail."""
    meta = sem_image.metadata or {}
    companion = _load_companion_calibration_record(Path(sem_image.source_path)) or {}
    tilt = tilt_info or sem_image.tilt_correction or {}
    raw = float(sem_image.nm_per_pixel_raw if sem_image.nm_per_pixel_raw is not None else sem_image.nm_per_pixel)
    return {
        "image": Path(sem_image.source_path).name,
        "magnification": companion.get("magnification") or meta.get("magnification"),
        "scale_bar_nm": companion.get("scale_bar_nm"),
        "scale_bar_pixels": companion.get("scale_bar_pixels"),
        "nm_per_pixel_raw": raw,
        "nm_per_pixel": float(sem_image.nm_per_pixel),
        "stage_tilt_deg": tilt.get("tilt_angle_deg"),
        "tilt_corrected": bool(tilt.get("applied")),
        "resized": bool(companion.get("resized", False)),
        "calibration_source": companion.get("calibration_source")
        or ("companion" if companion else ("tiff_metadata" if meta else "config_default")),
    }


def _parse_tiff_metadata(path: Path, default_nm: float) -> tuple[float, dict[str, Any]]:
    """Extract pixel size from TIFF tags when available."""
    metadata: dict[str, Any] = {}
    nm_per_pixel = default_nm

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        tags = {tag.name: tag.value for tag in page.tags.values() if tag.name}
        metadata.update(tags)

        for key in ("pixel_size_nm", "PixelSize", "XResolution"):
            if key in tags:
                try:
                    val = tags[key]
                    if isinstance(val, (tuple, list)) and len(val) == 2:
                        nm_per_pixel = float(val[0]) / float(val[1])
                    else:
                        nm_per_pixel = float(val)
                    break
                except (TypeError, ValueError):
                    continue

    companion = _load_companion_calibration(path)
    if companion is not None:
        nm_per_pixel = companion

    return nm_per_pixel, metadata


def load_image(
    path: str | Path,
    default_nm_per_pixel: float = 1.0,
    config: dict | None = None,
) -> SEMImage:
    """Load PNG, TIFF (8/16-bit), or JPG SEM image with calibration."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    companion_record = _load_companion_calibration_record(path)
    cal_cfg = (config or {}).get("calibration", {})
    if cal_cfg.get("reject_resized_without_factor", False):
        if companion_record and companion_record.get("resized"):
            factor = companion_record.get("resize_factor")
            if factor is None or float(factor) <= 0:
                raise ResizedImageError(
                    f"{path.name}: image marked resized without resize_factor; "
                    "per-image nm/px would be invalid."
                )

    suffix = path.suffix.lower()
    metadata: dict[str, Any] = {}
    nm_per_pixel = default_nm_per_pixel

    if suffix in (".tif", ".tiff"):
        data = tifffile.imread(path)
        nm_per_pixel, metadata = _parse_tiff_metadata(path, default_nm_per_pixel)
        bit_depth = 16 if data.dtype == np.uint16 else 8
    else:
        with Image.open(path) as img:
            metadata = dict(img.info)
            if img.mode != "L":
                img = img.convert("L")
            data = np.array(img)
        bit_depth = 8
        companion = _load_companion_calibration(path)
        if companion is not None:
            nm_per_pixel = companion

    if data.ndim == 3:
        data = data[:, :, 0]

    if companion_record:
        metadata = {**metadata, "calibration_record": companion_record}

    return SEMImage(
        data=data,
        nm_per_pixel=nm_per_pixel,
        source_path=str(path),
        metadata=metadata,
        bit_depth=bit_depth,
        nm_per_pixel_raw=nm_per_pixel,
    )


def save_image(array: np.ndarray, path: str | Path) -> None:
    """Save uint8 or float image array to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if array.dtype in (np.float32, np.float64):
        array = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(array).save(path)
