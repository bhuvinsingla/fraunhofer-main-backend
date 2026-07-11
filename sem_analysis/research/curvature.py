"""Arc-length curvature estimation κ(s) = dθ/ds along blade contours."""

from __future__ import annotations

import numpy as np


def compute_curvature_profile(contour: np.ndarray, smooth_window: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute arc length s, tangent angle θ, and curvature κ along an ordered contour.

    Returns (s, kappa, points) aligned to interior indices.
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 5:
        return np.array([]), np.array([]), pts

    dx = np.gradient(pts[:, 0])
    dy = np.gradient(pts[:, 1])
    ds = np.sqrt(dx**2 + dy**2)
    ds[ds < 1e-9] = 1e-9
    s = np.cumsum(ds)
    s -= s[0]

    theta = np.arctan2(dy, dx)
    dtheta = np.gradient(theta, s)
    kappa = np.abs(dtheta)

    if smooth_window > 1 and len(kappa) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        kappa = np.convolve(kappa, kernel, mode="same")

    return s, kappa, pts


def find_curvature_tip(
    contour: np.ndarray,
    peak_hint: tuple[float, float] | None = None,
    tip_region_fraction: float = 0.35,
    smooth_window: int = 5,
) -> tuple[tuple[float, float], float, int] | None:
    """
    Locate tip at maximum curvature in the upper portion of the contour.

    Returns (tip_point, kappa_max, index) or None.
    """
    s, kappa, pts = compute_curvature_profile(contour, smooth_window=smooth_window)
    if len(kappa) == 0:
        return None

    if peak_hint is not None:
        py = peak_hint[1]
        upper_mask = pts[:, 1] <= py + (pts[:, 1].max() - py) * tip_region_fraction
    else:
        y_min = pts[:, 1].min()
        upper_mask = pts[:, 1] <= y_min + (pts[:, 1].max() - y_min) * tip_region_fraction

    if not upper_mask.any():
        upper_mask = np.ones(len(kappa), dtype=bool)

    local_kappa = kappa.copy()
    local_kappa[~upper_mask] = 0.0
    idx = int(np.argmax(local_kappa))
    if local_kappa[idx] <= 0:
        return None

    tip = (float(pts[idx, 0]), float(pts[idx, 1]))
    return tip, float(local_kappa[idx]), idx
