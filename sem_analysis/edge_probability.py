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


def preprocess_sem(image: np.ndarray) -> EdgeProbabilityMaps:
    """
    CLAHE → bilateral → auto-Canny + Scharr magnitude.

    Canny identifies connected boundaries; Scharr scores them.
    Do not average Canny/Sobel/threshold outputs as independent measurements.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid SEM image.")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        if gray.dtype != np.uint8:
            if gray.max() <= 1.0:
                gray = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
            else:
                gray = np.clip(gray, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    denoised = cv2.bilateralFilter(enhanced, d=7, sigmaColor=40, sigmaSpace=40)

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
