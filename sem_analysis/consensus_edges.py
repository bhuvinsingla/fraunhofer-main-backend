"""Multi-algorithm edge detection with consensus voting (not averaging).

Compare classical detectors, then keep pixels that enough methods agree on.
Optional hooks for ridge / active contours / DL segmentation when available.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np


def _to_u8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def _binarize_magnitude(mag: np.ndarray, percentile: float = 85.0) -> np.ndarray:
    """Threshold gradient magnitude to a binary edge map (not a soft average)."""
    if mag.max() <= 0:
        return np.zeros(mag.shape, dtype=np.uint8)
    thr = float(np.percentile(mag, percentile))
    return (mag >= thr).astype(np.uint8) * 255


def edge_canny(uint8: np.ndarray, low: int = 40, high: int = 120) -> np.ndarray:
    return cv2.Canny(uint8, low, high)


def edge_sobel(uint8: np.ndarray, percentile: float = 85.0) -> np.ndarray:
    gx = cv2.Sobel(uint8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(uint8, cv2.CV_32F, 0, 1, ksize=3)
    return _binarize_magnitude(cv2.magnitude(gx, gy), percentile)


def edge_scharr(uint8: np.ndarray, percentile: float = 85.0) -> np.ndarray:
    gx = cv2.Scharr(uint8, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(uint8, cv2.CV_32F, 0, 1)
    return _binarize_magnitude(cv2.magnitude(gx, gy), percentile)


def edge_laplacian(uint8: np.ndarray, percentile: float = 90.0) -> np.ndarray:
    lap = cv2.Laplacian(uint8, cv2.CV_32F, ksize=3)
    return _binarize_magnitude(np.abs(lap), percentile)


def edge_adaptive_threshold(uint8: np.ndarray, block_size: int = 31, c: int = 5) -> np.ndarray:
    bs = block_size if block_size % 2 == 1 else block_size + 1
    binary = cv2.adaptiveThreshold(
        uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bs, c
    )
    # Edges ≈ boundaries of adaptive regions
    return cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))


def edge_otsu(uint8: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(uint8, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))


def edge_ridge(uint8: np.ndarray, percentile: float = 92.0) -> np.ndarray:
    """
    Ridge-like response via Frangi-style approximation using Hessian eigenvalues
    when skimage is available; otherwise Laplacian of Gaussian magnitude.
    """
    try:
        from skimage.filters import frangi

        resp = frangi(uint8.astype(np.float64) / 255.0)
        return _binarize_magnitude(resp.astype(np.float32), percentile)
    except Exception:
        blur = cv2.GaussianBlur(uint8, (0, 0), 1.5)
        lap = cv2.Laplacian(blur, cv2.CV_32F)
        return _binarize_magnitude(np.abs(lap), percentile)


def edge_active_contours(uint8: np.ndarray) -> np.ndarray:
    """
    Lightweight active-contour proxy: morphological geodesic active contour
    when skimage is available; otherwise empty (skipped in consensus).
    """
    try:
        from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient

        gimg = inverse_gaussian_gradient(uint8.astype(np.float64) / 255.0)
        # Seed from Otsu foreground
        _, seed = cv2.threshold(uint8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        init = seed.astype(np.float64)
        evolved = morphological_geodesic_active_contour(
            gimg, num_iter=30, init_level_set=init, smoothing=1
        )
        mask = (evolved > 0.5).astype(np.uint8) * 255
        return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    except Exception:
        return np.zeros_like(uint8)


def edge_deep_learning(uint8: np.ndarray) -> np.ndarray:
    """
    Placeholder for DL segmentation (e.g. U-Net / SAM).
    Returns empty until a model is configured — excluded from consensus votes.
    """
    return np.zeros_like(uint8)


_DETECTORS: dict[str, Callable[..., np.ndarray]] = {
    "canny": edge_canny,
    "sobel": edge_sobel,
    "scharr": edge_scharr,
    "laplacian": edge_laplacian,
    "adaptive_threshold": edge_adaptive_threshold,
    "otsu": edge_otsu,
    "ridge": edge_ridge,
    "active_contours": edge_active_contours,
    "deep_learning": edge_deep_learning,
}


def run_edge_detectors(
    image: np.ndarray,
    methods: list[str],
    config: dict,
) -> dict[str, np.ndarray]:
    """Run selected edge detectors; skip empty / failed maps."""
    cfg = config.get("edge_detection", {})
    uint8 = _to_u8(image)
    results: dict[str, np.ndarray] = {}

    for name in methods:
        key = name.lower().strip()
        fn = _DETECTORS.get(key)
        if fn is None:
            continue
        try:
            if key == "canny":
                edges = fn(
                    uint8,
                    low=int(cfg.get("canny_low", 40)),
                    high=int(cfg.get("canny_high", 120)),
                )
            elif key in ("sobel", "scharr", "laplacian", "ridge"):
                edges = fn(uint8, percentile=float(cfg.get("gradient_percentile", 85.0)))
            elif key == "adaptive_threshold":
                edges = fn(
                    uint8,
                    block_size=int(cfg.get("adaptive_block_size", 31)),
                    c=int(cfg.get("adaptive_c", 5)),
                )
            else:
                edges = fn(uint8)
        except Exception:
            continue

        if edges is None or edges.size == 0:
            continue
        if int(np.count_nonzero(edges)) == 0:
            continue
        results[key] = (edges > 0).astype(np.uint8) * 255

    return results


def consensus_edge_map(
    detector_maps: dict[str, np.ndarray],
    min_votes: int | None = None,
    vote_ratio: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """
    Build consensus by voting — do NOT average soft responses.

    A pixel is an edge if at least max(min_votes, ceil(n_methods * vote_ratio))
    detectors mark it as edge.
    """
    if not detector_maps:
        return np.zeros((1, 1), dtype=np.uint8), {"methods": [], "votes_required": 0}

    names = list(detector_maps.keys())
    stack = np.stack([(m > 0).astype(np.uint8) for m in detector_maps.values()], axis=0)
    votes = stack.sum(axis=0)
    n = len(names)

    if min_votes is None:
        required = max(1, int(np.ceil(n * vote_ratio)))
    else:
        required = max(1, min(int(min_votes), n))

    consensus = (votes >= required).astype(np.uint8) * 255
    meta = {
        "methods": names,
        "n_methods": n,
        "votes_required": required,
        "vote_ratio": vote_ratio,
        "strategy": "majority_vote_not_average",
        "per_method_edge_pixels": {k: int(np.count_nonzero(v)) for k, v in detector_maps.items()},
        "consensus_edge_pixels": int(np.count_nonzero(consensus)),
    }
    return consensus, meta


def build_consensus_edges(image: np.ndarray, config: dict) -> tuple[np.ndarray, dict]:
    """Full multi-detector → consensus pipeline."""
    cfg = config.get("edge_detection", {})
    methods = cfg.get(
        "consensus_methods",
        ["canny", "sobel", "scharr", "laplacian", "adaptive_threshold", "otsu"],
    )
    maps = run_edge_detectors(image, methods, config)
    if not maps:
        # Fallback: Canny alone
        uint8 = _to_u8(image)
        fallback = edge_canny(
            uint8,
            low=int(cfg.get("canny_low", 40)),
            high=int(cfg.get("canny_high", 120)),
        )
        return fallback, {"methods": ["canny"], "votes_required": 1, "fallback": True}

    min_votes = cfg.get("consensus_min_votes")
    vote_ratio = float(cfg.get("consensus_vote_ratio", 0.5))
    return consensus_edge_map(maps, min_votes=min_votes, vote_ratio=vote_ratio)
