"""Robust line fitting for blade flank estimation."""

from __future__ import annotations

import cv2
import numpy as np


def fit_line_tls(points: np.ndarray) -> tuple[float, float, list[float]]:
    """Total least squares line fit via cv2.fitLine. Returns (m, b, segment)."""
    if len(points) < 2:
        raise ValueError("Need at least 2 points")

    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = float(vx.item()), float(vy.item()), float(x0.item()), float(y0.item())

    if abs(vx) < 1e-9:
        m = float("inf")
        b = x0
    else:
        m = vy / vx
        b = y0 - m * x0

    ys = points[:, 1]
    y_min, y_max = float(ys.min()), float(ys.max())
    return m, b, _line_segment(m, b, y_min, y_max)


def fit_line_ransac(
    points: np.ndarray,
    max_iterations: int = 200,
    threshold: float = 2.0,
    min_inliers: int = 5,
) -> tuple[float, float, np.ndarray, list[float]] | None:
    """RANSAC line fit. Returns (m, b, inlier_mask, segment) or None."""
    n = len(points)
    if n < min_inliers:
        return None

    best_inliers: np.ndarray | None = None
    best_count = 0
    rng = np.random.default_rng(42)

    for _ in range(max_iterations):
        idx = rng.choice(n, size=2, replace=False)
        p1, p2 = points[idx[0]], points[idx[1]]
        if abs(p2[0] - p1[0]) < 1e-9:
            continue

        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - m * p1[0]

        dists = _point_line_distances(points, m, b)
        inliers = dists < threshold
        count = int(np.sum(inliers))

        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < min_inliers:
        return None

    inlier_pts = points[best_inliers]
    m, b, segment = fit_line_tls(inlier_pts)
    return m, b, best_inliers, segment


def _point_line_distances(points: np.ndarray, m: float, b: float) -> np.ndarray:
    if m == float("inf"):
        return np.abs(points[:, 0] - b)
    return np.abs(m * points[:, 0] - points[:, 1] + b) / np.sqrt(m * m + 1)


def _line_segment(m: float, b: float, y_start: float, y_end: float) -> list[float]:
    if m == float("inf"):
        return [b, y_start, b, y_end]
    if abs(m) < 1e-9:
        x = b
        return [x, y_start, x, y_end]
    x1 = (y_start - b) / m
    x2 = (y_end - b) / m
    return [float(x1), float(y_start), float(x2), float(y_end)]


def line_intersection(m1: float, b1: float, m2: float, b2: float) -> tuple[float, float] | None:
    if abs(m1 - m2) < 1e-10:
        return None
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return float(x), float(y)
