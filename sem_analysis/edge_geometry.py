"""Edge geometry: resampling, smoothing, TLS lines, circumcircle, local tip axis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import interpolate


@dataclass
class FittedBranch:
    """Smoothed branch with residual against raw points."""

    raw_points: np.ndarray
    smooth_points: np.ndarray
    residual_px: float


def sort_by_distance_from_apex(points: np.ndarray, apex: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    apex = np.asarray(apex, dtype=float).reshape(2)
    d = np.linalg.norm(pts - apex, axis=1)
    return pts[np.argsort(d)]


def resample_arc_length(points: np.ndarray, spacing_px: float = 1.0) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < spacing_px:
        return pts
    n = max(2, int(np.floor(total / spacing_px)) + 1)
    s_new = np.linspace(0.0, total, n)
    x = np.interp(s_new, s, pts[:, 0])
    y = np.interp(s_new, s, pts[:, 1])
    return np.column_stack([x, y])


def smooth_branch(
    points: np.ndarray,
    apex: np.ndarray,
    spacing_px: float = 1.0,
    smooth_factor: float = 0.5,
    max_residual_px: float = 2.0,
) -> FittedBranch | None:
    """
    Sort → uniform arc-length resample → cubic smoothing spline.
    Reject if mean residual > max_residual_px (~1–2 px).
    """
    pts = sort_by_distance_from_apex(points, apex)
    if len(pts) < 4:
        return None
    resampled = resample_arc_length(pts, spacing_px=spacing_px)
    if len(resampled) < 4:
        return None

    seg = np.linalg.norm(np.diff(resampled, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    # s=0 at apex-nearest
    try:
        # UnivariateSpline needs strictly increasing s
        s_u, idx = np.unique(s, return_index=True)
        xu = resampled[idx, 0]
        yu = resampled[idx, 1]
        if len(s_u) < 4:
            return None

        # Mild smoothing: s ~ smooth_factor * n * sigma^2 with sigma~0.5 px
        n = len(s_u)
        spl_s = float(smooth_factor * n * (0.5**2))
        sx = interpolate.UnivariateSpline(s_u, xu, s=spl_s, k=3)
        sy = interpolate.UnivariateSpline(s_u, yu, s=spl_s, k=3)
        xs = sx(s_u)
        ys = sy(s_u)
        smooth = np.column_stack([xs, ys])
        residual = float(np.mean(np.hypot(xs - xu, ys - yu)))
    except Exception:
        return None

    if residual > max_residual_px:
        return None
    return FittedBranch(raw_points=pts, smooth_points=smooth, residual_px=residual)


def local_symmetry_axis(
    left: np.ndarray,
    right: np.ndarray,
    apex: np.ndarray,
) -> np.ndarray:
    """Unit axis from apex toward the mean of deep flank midpoints (down the tip)."""
    apex = np.asarray(apex, dtype=float).reshape(2)
    left = np.asarray(left, dtype=float).reshape(-1, 2)
    right = np.asarray(right, dtype=float).reshape(-1, 2)
    if len(left) == 0 or len(right) == 0:
        return np.array([0.0, 1.0])
    # Use points farthest from apex
    mid = 0.5 * (left[-1] + right[-1])
    axis = mid - apex
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return np.array([0.0, 1.0])
    return axis / n


def rotate_to_axis(
    points: np.ndarray,
    apex: np.ndarray,
    axis_unit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform so apex→axis becomes +Y.
    Returns (transformed_points, R_matrix, apex).
    """
    apex = np.asarray(apex, dtype=float).reshape(2)
    axis_unit = np.asarray(axis_unit, dtype=float).reshape(2)
    # Target is (0, 1)
    target = np.array([0.0, 1.0])
    # 2D rotation aligning axis_unit to target
    a = axis_unit
    cos_t = float(np.clip(np.dot(a, target), -1, 1))
    sin_t = float(a[0] * target[1] - a[1] * target[0])  # z-component of cross
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    # Actually: rotate a onto target: R @ a = target
    # Using angle from a to target
    ang_a = np.arctan2(a[1], a[0])
    ang_t = np.arctan2(target[1], target[0])
    theta = ang_t - ang_a
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    local = (pts - apex) @ R.T
    return local, R, apex


def inverse_rotate(local_pts: np.ndarray, R: np.ndarray, apex: np.ndarray) -> np.ndarray:
    return local_pts @ R + apex


def circumcircle_radius(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    epsilon: float = 1e-9,
) -> float:
    """R = abc / (4A) via twice_area = 2A → R = abc / (2 * twice_area)."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float)
    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p1 - p3)
    c = np.linalg.norm(p1 - p2)
    twice_area = abs(float((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])))
    if twice_area <= epsilon:
        raise ValueError("The three points are collinear or unstable.")
    return float((a * b * c) / (2.0 * twice_area))


def circumcircle_center(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> tuple[float, float]:
    """Circumcenter of triangle p1,p2,p3."""
    A = np.asarray(p1, dtype=float)
    B = np.asarray(p2, dtype=float)
    C = np.asarray(p3, dtype=float)
    D = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
    if abs(D) < 1e-12:
        raise ValueError("Collinear")
    ux = (
        (A[0] ** 2 + A[1] ** 2) * (B[1] - C[1])
        + (B[0] ** 2 + B[1] ** 2) * (C[1] - A[1])
        + (C[0] ** 2 + C[1] ** 2) * (A[1] - B[1])
    ) / D
    uy = (
        (A[0] ** 2 + A[1] ** 2) * (C[0] - B[0])
        + (B[0] ** 2 + B[1] ** 2) * (A[0] - C[0])
        + (C[0] ** 2 + C[1] ** 2) * (B[0] - A[0])
    ) / D
    return float(ux), float(uy)


def fit_line_tls(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Total least-squares line: returns (centroid, unit direction)."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        raise ValueError("At least two 2D points are required.")
    centroid = points.mean(axis=0)
    _, singular_values, vh = np.linalg.svd(points - centroid)
    if singular_values[0] <= 1e-9:
        raise ValueError("Degenerate line points.")
    direction = vh[0].copy()
    direction /= np.linalg.norm(direction)
    return centroid, direction


def intersect_lines(
    p1: np.ndarray,
    d1: np.ndarray,
    p2: np.ndarray,
    d2: np.ndarray,
    min_cross: float = 0.15,
) -> np.ndarray:
    """Stable line intersection; reject near-parallel (|sin α| < min_cross)."""
    matrix = np.column_stack((d1, -d2))
    determinant = abs(np.linalg.det(matrix))
    if determinant < min_cross:
        raise ValueError("Lines are too close to parallel.")
    t, _ = np.linalg.solve(matrix, p2 - p1)
    return p1 + t * d1


def included_angle_degrees(
    apex: np.ndarray,
    left_point: np.ndarray,
    right_point: np.ndarray,
) -> float:
    left_vector = np.asarray(left_point, float) - np.asarray(apex, float)
    right_vector = np.asarray(right_point, float) - np.asarray(apex, float)
    left_norm = np.linalg.norm(left_vector)
    right_norm = np.linalg.norm(right_vector)
    if left_norm <= 1e-9 or right_norm <= 1e-9:
        raise ValueError("Invalid angle vectors.")
    cosine = float(np.dot(left_vector, right_vector) / (left_norm * right_norm))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def stability_ratio(radii: list[float]) -> float:
    """S = (max - min) / median for R at 0.9l, l, 1.1l."""
    arr = np.asarray([r for r in radii if r is not None and np.isfinite(r)], dtype=float)
    if len(arr) < 2:
        return 0.0
    med = float(np.median(arr))
    if med <= 1e-9:
        return 1e9
    return float((arr.max() - arr.min()) / med)
