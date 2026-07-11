"""Extract ordered left/right blade edge contours from Canny edge map."""

from __future__ import annotations

import numpy as np


def extract_left_right_contours(
    edge_points: np.ndarray,
    peak: tuple[float, float],
    window_y_px: float = 80.0,
    window_x_px: float = 40.0,
    min_points: int = 8,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Split local edge pixels into ordered left L and right R point sets."""
    if edge_points is None or len(edge_points) < min_points * 2:
        return None

    px, py = float(peak[0]), float(peak[1])
    pts = edge_points.reshape(-1, 2).astype(np.float64)

    mask = (
        (np.abs(pts[:, 0] - px) <= window_x_px)
        & (pts[:, 1] >= py - window_y_px * 0.25)
        & (pts[:, 1] <= py + window_y_px)
    )
    local = pts[mask]
    if len(local) < min_points * 2:
        return None

    left = local[local[:, 0] <= px]
    right = local[local[:, 0] > px]
    if len(left) < min_points or len(right) < min_points:
        return None

    left = left[np.argsort(left[:, 1])]
    right = right[np.argsort(right[:, 1])]
    return left, right


def merge_contour_arc(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Ordered contour: left ascending y, then right descending y."""
    if len(right) > 0:
        return np.vstack([left, right[::-1]])
    return left.copy()
