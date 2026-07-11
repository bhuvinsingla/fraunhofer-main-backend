"""SEM edge probability preprocessing (Canny + Scharr score — not multi-method averaging)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EdgeProbabilityMaps:
    """Canny boundaries + Scharr gradient strength for scoring."""

    gray: np.ndarray
    denoised: np.ndarray
    canny: np.ndarray
    gradient: np.ndarray


def _as_gray_u8(image: np.ndarray) -> np.ndarray:
    """Force a contiguous CV_8U single-channel image for Canny / CLAHE / bilateral."""
    arr = np.asarray(image)
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        elif arr.dtype == np.uint8:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        else:
            # Float / uint16 color → uint8 first, then gray
            u8 = _scale_to_u8(arr)
            arr = cv2.cvtColor(u8, cv2.COLOR_BGR2GRAY)
            return np.ascontiguousarray(arr, dtype=np.uint8)
    if arr.dtype != np.uint8:
        arr = _scale_to_u8(arr)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return np.ascontiguousarray(arr, dtype=np.uint8)


def _scale_to_u8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return (arr / 256).astype(np.uint8)
    f = arr.astype(np.float32)
    mx = float(np.nanmax(f)) if f.size else 0.0
    if mx <= 1.0 + 1e-6:
        return (np.clip(f, 0, 1) * 255).astype(np.uint8)
    if mx <= 255.0 + 1e-3:
        return np.clip(f, 0, 255).astype(np.uint8)
    return (np.clip(f / mx, 0, 1) * 255).astype(np.uint8)


def preprocess_sem(image: np.ndarray) -> EdgeProbabilityMaps:
    """
    CLAHE → bilateral → auto-Canny + Scharr magnitude.

    Canny identifies connected boundaries; Scharr scores them.
    Do not average Canny/Sobel/threshold outputs as independent measurements.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid SEM image.")

    gray = _as_gray_u8(image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    denoised = cv2.bilateralFilter(enhanced, d=7, sigmaColor=40, sigmaSpace=40)
    denoised = np.ascontiguousarray(denoised, dtype=np.uint8)

    median = float(np.median(denoised))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    if upper <= lower:
        upper = min(255, lower + 1)

    canny = cv2.Canny(denoised, threshold1=lower, threshold2=upper, L2gradient=True)

    grad_x = cv2.Scharr(denoised, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(denoised, cv2.CV_32F, 0, 1)
    gradient = cv2.magnitude(grad_x, grad_y)
    gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return EdgeProbabilityMaps(
        gray=gray,
        denoised=denoised,
        canny=canny,
        gradient=gradient,
    )
