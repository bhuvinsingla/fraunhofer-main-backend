"""Extract a local blade-edge contour segment around a serration peak."""

from __future__ import annotations

import numpy as np


def extract_local_contour(
    edge_points: np.ndarray,
    peak: tuple[float, float] | np.ndarray,
    window_y_px: float = 80.0,
    window_x_px: float = 40.0,
    min_points: int = 12,
) -> np.ndarray | None:
    """Build a closed contour segment centered on a peak, with peak as local tip."""
    if edge_points is None or len(edge_points) < min_points:
        return None

    px, py = float(peak[0]), float(peak[1])
    pts = edge_points.reshape(-1, 2).astype(np.float64)

    mask = (
        (np.abs(pts[:, 0] - px) <= window_x_px)
        & (pts[:, 1] >= py - window_y_px * 0.25)
        & (pts[:, 1] <= py + window_y_px)
    )
    local = pts[mask]
    if len(local) < min_points:
        return None

    # Order points along edge: left flank ascending y, then right flank descending y
    left = local[local[:, 0] <= px]
    right = local[local[:, 0] > px]

    if len(left) < 3 or len(right) < 3:
        return None

    left_sorted = left[np.argsort(left[:, 1])]
    right_sorted = right[np.argsort(right[:, 1])]

    ordered = np.vstack([left_sorted, right_sorted[::-1]])
    if len(ordered) < min_points:
        return None

    # Ensure peak region is represented at the top (min y)
    tip_idx = np.argmin(ordered[:, 1])
    ordered = np.roll(ordered, -tip_idx, axis=0)

    return ordered.reshape(-1, 1, 2).astype(np.float32)
