"""SEM preprocessing matching the reference OpenCV stack.

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)
filtered = cv2.bilateralFilter(enhanced, d=7, sigmaColor=50, sigmaSpace=50)
# Canny(filtered, 40, 120) applied in edge_detection
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage import restoration


@dataclass
class ProcessedImage:
    """Preprocessed image ready for analysis."""

    data: np.ndarray  # float32 [0, 1]
    nm_per_pixel: float
    original_shape: tuple[int, int]
    cropped_rows: int = 0


def to_grayscale_bgr(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale via COLOR_BGR2GRAY (or pass through if already gray)."""
    if image.ndim == 2:
        if image.dtype == np.uint8:
            return image
        if image.dtype == np.uint16:
            return (image / 256).astype(np.uint8)
        # float [0,1] or other
        arr = image.astype(np.float32)
        if arr.max() <= 1.0:
            return (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        return np.clip(arr, 0, 255).astype(np.uint8)

    if image.ndim == 3:
        # float RGB/BGR in [0,1]
        if image.dtype in (np.float32, np.float64) and image.max() <= 1.0:
            uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        elif image.dtype == np.uint16:
            uint8 = (image / 256).astype(np.uint8)
        else:
            uint8 = image.astype(np.uint8)

        if uint8.shape[2] == 1:
            return uint8[:, :, 0]
        if uint8.shape[2] == 4:
            return cv2.cvtColor(uint8, cv2.COLOR_BGRA2GRAY)
        # Prefer BGR2GRAY as in the reference snippet
        return cv2.cvtColor(uint8, cv2.COLOR_BGR2GRAY)

    return np.asarray(image, dtype=np.uint8)


def crop_zeiss_info_bar(image: np.ndarray, crop_ratio: float = 0.08) -> tuple[np.ndarray, int]:
    """Remove Zeiss SEM information / scale-bar area from the bottom of the image."""
    if crop_ratio <= 0:
        return image, 0
    h = image.shape[0]
    crop_px = int(h * crop_ratio)
    if crop_px < 1 or crop_px >= h - 10:
        return image, 0
    return image[: h - crop_px, :], crop_px


def apply_clahe(
    gray_u8: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> np.ndarray:
    """CLAHE local contrast — preferred over global for uneven SEM lighting."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray_u8)


def apply_bilateral(
    image_u8: np.ndarray,
    d: int = 7,
    sigma_color: float = 50,
    sigma_space: float = 50,
) -> np.ndarray:
    """Bilateral filter on uint8 image (reference: d=7, sigmaColor=50, sigmaSpace=50)."""
    return cv2.bilateralFilter(image_u8, d, sigma_color, sigma_space)


# --- Legacy helpers kept for tests / alternate configs ---

def _to_float01(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    arr = image.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / arr.max()
    return arr


def perona_malik_diffusion(
    image: np.ndarray,
    n_iter: int = 15,
    kappa: float = 0.1,
    gamma: float = 0.15,
) -> np.ndarray:
    img = image.astype(np.float64)
    for _ in range(n_iter):
        nabla_n = np.roll(img, -1, axis=0) - img
        nabla_s = np.roll(img, 1, axis=0) - img
        nabla_e = np.roll(img, -1, axis=1) - img
        nabla_w = np.roll(img, 1, axis=1) - img
        c_n = np.exp(-(nabla_n / kappa) ** 2)
        c_s = np.exp(-(nabla_s / kappa) ** 2)
        c_e = np.exp(-(nabla_e / kappa) ** 2)
        c_w = np.exp(-(nabla_w / kappa) ** 2)
        img += gamma * (c_n * nabla_n + c_s * nabla_s + c_e * nabla_e + c_w * nabla_w)
    return np.clip(img, 0, 1).astype(np.float32)


def denoise_bilateral(
    image: np.ndarray,
    d: int = 7,
    sigma_color: float = 0.08,
    sigma_space: float = 7,
) -> np.ndarray:
    """Float [0,1] bilateral (legacy). Prefer apply_bilateral on uint8."""
    uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    # If sigma_color looks like a fraction, scale to uint8 range
    sc = sigma_color * 255 if sigma_color <= 1.0 else sigma_color
    ss = sigma_space
    filtered = cv2.bilateralFilter(uint8, d, sc, ss)
    return filtered.astype(np.float32) / 255.0


def denoise(image: np.ndarray, method: str = "bilateral", **kwargs) -> np.ndarray:
    method = (method or "bilateral").lower()
    if method == "bilateral":
        return denoise_bilateral(
            image,
            d=kwargs.get("bilateral_d", 7),
            sigma_color=kwargs.get("bilateral_sigma_color", 50),
            sigma_space=kwargs.get("bilateral_sigma_space", 50),
        )
    if method in ("nlm", "non_local_means", "nl_means"):
        return restoration.denoise_nl_means(
            image,
            h=kwargs.get("nlm_h", 0.08),
            patch_size=kwargs.get("nlm_patch_size", 5),
            patch_distance=kwargs.get("nlm_patch_distance", 6),
            fast_mode=True,
        ).astype(np.float32)
    if method == "anisotropic":
        return perona_malik_diffusion(
            image,
            n_iter=kwargs.get("pm_iterations", 15),
            kappa=kwargs.get("pm_kappa", 0.1),
            gamma=kwargs.get("pm_gamma", 0.15),
        )
    if method == "gaussian":
        return cv2.GaussianBlur(image, (0, 0), kwargs.get("gaussian_sigma", 1.0))
    return denoise_bilateral(image)


def enhance_contrast_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> np.ndarray:
    uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    return apply_clahe(uint8, clip_limit, tile_size).astype(np.float32) / 255.0


enhance_contrast = enhance_contrast_clahe


def preprocess(image: np.ndarray, nm_per_pixel: float, config: dict) -> ProcessedImage:
    """
    Reference stack (uint8 path):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = CLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        filtered = cv2.bilateralFilter(enhanced, d=7, sigmaColor=50, sigmaSpace=50)

    Canny(filtered, 40, 120) runs in edge_detection.
    """
    cfg = config.get("preprocessing", {})
    original_shape = image.shape[:2]

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = to_grayscale_bgr(image)

    # Crop metadata / scale-bar (after grayscale)
    cropped, crop_rows = crop_zeiss_info_bar(gray, cfg.get("info_bar_crop_ratio", 0.08))
    if cropped.dtype != np.uint8:
        cropped = to_grayscale_bgr(cropped)

    # CLAHE first (local contrast before denoise, per reference snippet)
    enhanced = apply_clahe(
        cropped,
        clip_limit=cfg.get("clahe_clip_limit", 2.0),
        tile_size=cfg.get("clahe_tile_size", 8),
    )

    # Bilateral on CLAHE output
    denoise_method = (cfg.get("denoise_method", "bilateral") or "bilateral").lower()
    if denoise_method in ("nlm", "non_local_means", "nl_means"):
        float_enh = enhanced.astype(np.float32) / 255.0
        filtered_f = restoration.denoise_nl_means(
            float_enh,
            h=cfg.get("nlm_h", 0.08),
            patch_size=cfg.get("nlm_patch_size", 5),
            patch_distance=cfg.get("nlm_patch_distance", 6),
            fast_mode=True,
        )
        filtered = (np.clip(filtered_f, 0, 1) * 255).astype(np.uint8)
    elif denoise_method == "anisotropic":
        float_enh = enhanced.astype(np.float32) / 255.0
        filtered_f = perona_malik_diffusion(
            float_enh,
            n_iter=cfg.get("pm_iterations", 15),
            kappa=cfg.get("pm_kappa", 0.1),
            gamma=cfg.get("pm_gamma", 0.15),
        )
        filtered = (np.clip(filtered_f, 0, 1) * 255).astype(np.uint8)
    else:
        # filtered = cv2.bilateralFilter(enhanced, d=7, sigmaColor=50, sigmaSpace=50)
        filtered = apply_bilateral(
            enhanced,
            d=cfg.get("bilateral_d", 7),
            sigma_color=cfg.get("bilateral_sigma_color", 50),
            sigma_space=cfg.get("bilateral_sigma_space", 50),
        )

    return ProcessedImage(
        data=(filtered.astype(np.float32) / 255.0),
        nm_per_pixel=nm_per_pixel,
        original_shape=original_shape,
        cropped_rows=crop_rows,
    )
