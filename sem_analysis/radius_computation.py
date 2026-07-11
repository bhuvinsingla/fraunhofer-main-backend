"""Radius computation via Hough circles and cross-tangent opening angles."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np
from scipy.optimize import least_squares


@dataclass
class RadiusResult:
    """Per-peak radius measurement with opening angle."""

    peak_id: int
    shape_id: int
    radius_px: float
    radius_nm: float
    radius_angstrom: float
    fit_residual: float
    center: tuple[float, float]
    method: str
    confidence_score: float = 0.0
    opening_angle_deg: float | None = None
    peak_location: tuple[float, float] | None = None
    tangent_lines: list | None = None
    metadata: dict = field(default_factory=dict)


class TipCondition(str, Enum):
    SHARP = "sharp"
    MODERATE = "moderate"
    BLUNT = "blunt"


def taubin_circle_fit(points: np.ndarray) -> tuple[tuple[float, float], float, float]:
    """Algebraic circle fit (Taubin). Returns (center, radius, residual)."""
    if len(points) < 3:
        raise ValueError("Need at least 3 points for circle fit")

    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)

    design = np.column_stack([x, y, np.ones(len(x))])
    target = -(x**2 + y**2)
    coeffs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    d_coef, e_coef, f_coef = coeffs

    center_x = -d_coef / 2
    center_y = -e_coef / 2
    radius_sq = center_x**2 + center_y**2 - f_coef
    if radius_sq <= 0:
        raise ValueError("Degenerate circle fit")
    radius = np.sqrt(radius_sq)

    distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    residual = float(np.std(distances - radius))

    return (float(center_x), float(center_y)), float(radius), residual


def geometric_circle_fit(points: np.ndarray) -> tuple[tuple[float, float], float, float]:
    """Nonlinear least-squares geometric circle fit."""
    if len(points) < 3:
        raise ValueError("Need at least 3 points for circle fit")

    cx0, cy0 = np.mean(points, axis=0)
    r0 = np.mean(np.linalg.norm(points - [cx0, cy0], axis=1))

    def residuals(params):
        cx, cy, r = params
        return np.linalg.norm(points - np.array([cx, cy]), axis=1) - r

    result = least_squares(residuals, [cx0, cy0, r0], method="lm")
    cx, cy, r = result.x
    residual = float(np.sqrt(np.mean(result.fun**2)))
    return (float(cx), float(cy)), float(abs(r)), residual


def _circle_overlap_confidence(
    center: tuple[float, float],
    radius: float,
    edge_points: np.ndarray,
    n_samples: int = 72,
    tolerance: float = 3.0,
) -> float:
    """Confidence = fraction of circle circumference overlapping edge pixels."""
    cx, cy = center
    if radius <= 0 or len(edge_points) == 0:
        return 0.0

    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    circle_pts = np.column_stack([
        cx + radius * np.cos(angles),
        cy + radius * np.sin(angles),
    ])

    from scipy.spatial import cKDTree
    tree = cKDTree(edge_points)
    dists, _ = tree.query(circle_pts, k=1)
    overlap = np.sum(dists < tolerance) / n_samples
    return float(np.clip(overlap, 0.0, 1.0))


def compute_cross_tangent(
    center: tuple[float, float],
    radius: float,
    peak: tuple[float, float],
    edge_points: np.ndarray,
    tolerance: float = 4.0,
    line_length: float = 30.0,
) -> dict | None:
    """Compute cross-tangent lines and local opening angle at a peak."""
    cx, cy = center
    px, py = peak

    if len(edge_points) < 3 or radius <= 0:
        return None

    dists = np.abs(np.linalg.norm(edge_points - np.array([cx, cy]), axis=1) - radius)
    on_circle = edge_points[dists < tolerance]
    if len(on_circle) < 2:
        on_circle = edge_points[
            np.linalg.norm(edge_points - np.array([px, py]), axis=1) < radius * 2
        ]
    if len(on_circle) < 2:
        return None

    left = on_circle[on_circle[:, 0] < px]
    right = on_circle[on_circle[:, 0] >= px]
    if len(left) == 0 or len(right) == 0:
        mid = len(on_circle) // 2
        left = on_circle[:mid]
        right = on_circle[mid:]
    if len(left) == 0 or len(right) == 0:
        return None

    left_pt = left[np.argmin(np.abs(left[:, 1] - py))]
    right_pt = right[np.argmin(np.abs(right[:, 1] - py))]

    def tangent_line(pt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx, dy = pt[0] - cx, pt[1] - cy
        norm = np.hypot(dx, dy) + 1e-9
        tangent = np.array([-dy / norm, dx / norm])
        p1 = pt - tangent * line_length
        p2 = pt + tangent * line_length
        return p1, p2

    left_line = tangent_line(left_pt)
    right_line = tangent_line(right_pt)

    tan_l = left_line[1] - left_line[0]
    tan_r = right_line[1] - right_line[0]
    tan_l /= np.linalg.norm(tan_l) + 1e-9
    tan_r /= np.linalg.norm(tan_r) + 1e-9

    dot = np.clip(np.dot(tan_l, tan_r), -1.0, 1.0)
    opening_angle = float(np.degrees(np.arccos(abs(dot))))

    return {
        "opening_angle_deg": opening_angle,
        "left_contact": left_pt.tolist(),
        "right_contact": right_pt.tolist(),
        "tangent_lines": [
            [left_line[0].tolist(), left_line[1].tolist()],
            [right_line[0].tolist(), right_line[1].tolist()],
        ],
    }


def hough_circle_fit(
    image: np.ndarray,
    peak: tuple[float, float],
    edge_points: np.ndarray,
    config: dict,
) -> tuple[tuple[float, float], float, float] | None:
    """Localized Hough Circle Transform around a peak ROI."""
    cfg = config.get("radius", {})
    roi_size = cfg.get("hough_roi_size", 60)
    px, py = int(peak[0]), int(peak[1])
    h, w = image.shape[:2]

    x1 = max(0, px - roi_size)
    y1 = max(0, py - roi_size)
    x2 = min(w, px + roi_size)
    y2 = min(h, py + roi_size)

    roi = (np.clip(image[y1:y2, x1:x2], 0, 1) * 255).astype(np.uint8)
    if roi.size == 0:
        return None

    circles = cv2.HoughCircles(
        roi,
        cv2.HOUGH_GRADIENT,
        dp=cfg.get("hough_dp", 1.2),
        minDist=cfg.get("hough_min_dist", 10),
        param1=cfg.get("hough_param1", 50),
        param2=cfg.get("hough_param2", 25),
        minRadius=cfg.get("hough_min_radius", 3),
        maxRadius=cfg.get("hough_max_radius", roi_size),
    )

    if circles is not None:
        c = circles[0][0]
        cx = float(c[0] + x1)
        cy = float(c[1] + y1)
        r = float(c[2])
        dists = np.abs(np.linalg.norm(edge_points - np.array([cx, cy]), axis=1) - r) if len(edge_points) else np.array([0])
        residual = float(np.mean(dists)) if len(dists) else 0.0
        return (cx, cy), r, residual

    # Fallback: Taubin on nearby edge points
    nearby = edge_points[
        np.linalg.norm(edge_points - np.array([px, py]), axis=1) < roi_size
    ] if len(edge_points) > 0 else np.empty((0, 2))
    if len(nearby) >= 3:
        try:
            return taubin_circle_fit(nearby)
        except (ValueError, np.linalg.LinAlgError):
            pass
    return None


def compute_radius(
    edge_points: np.ndarray,
    peak_id: int,
    shape_id: int,
    nm_per_pixel: float,
    method: str = "hough",
    confidence: float = 0.0,
    image: np.ndarray | None = None,
    peak: tuple[float, float] | None = None,
    config: dict | None = None,
) -> RadiusResult | None:
    """Fit circle at a peak and compute radius + opening angle."""
    if peak is None and len(edge_points) < 3:
        return None

    center = None
    radius_px = None
    residual = 0.0
    fit_method = method

    if method == "hough" and image is not None and peak is not None and config:
        fit = hough_circle_fit(image, peak, edge_points, config)
        if fit:
            center, radius_px, residual = fit
            fit_method = "hough"
        else:
            method = "taubin"

    if center is None:
        pts = edge_points
        if peak is not None and len(edge_points) > 0:
            window = (config or {}).get("radius", {}).get("edge_window_px", 25)
            pts = edge_points[np.linalg.norm(edge_points - np.array(peak), axis=1) < window]
        if len(pts) < 3:
            return None
        try:
            if method == "least_squares":
                center, radius_px, residual = geometric_circle_fit(pts)
                fit_method = "least_squares"
            else:
                center, radius_px, residual = taubin_circle_fit(pts)
                fit_method = "taubin"
        except (ValueError, np.linalg.LinAlgError):
            return None

    conf = _circle_overlap_confidence(center, radius_px, edge_points)
    if confidence > 0:
        conf = max(conf, confidence)

    opening_angle = None
    tangent_lines = None
    if peak is not None:
        ct = compute_cross_tangent(center, radius_px, peak, edge_points)
        if ct:
            opening_angle = ct["opening_angle_deg"]
            tangent_lines = ct["tangent_lines"]

    radius_nm = radius_px * nm_per_pixel
    return RadiusResult(
        peak_id=peak_id,
        shape_id=shape_id,
        radius_px=radius_px,
        radius_nm=radius_nm,
        radius_angstrom=radius_nm * 10.0,
        fit_residual=residual,
        center=center,
        method=fit_method,
        confidence_score=conf,
        opening_angle_deg=opening_angle,
        peak_location=peak,
        tangent_lines=tangent_lines,
    )


def aggregate_radii(results: list[RadiusResult], method: str = "mean") -> dict:
    """Aggregate per-peak radii into summary statistics."""
    if not results:
        return {
            "mean_radius_nm": None,
            "std_radius_nm": None,
            "mean_radius_angstrom": None,
            "mean_opening_angle_deg": None,
            "count": 0,
        }

    radii_nm = np.array([r.radius_nm for r in results])
    angles = [r.opening_angle_deg for r in results if r.opening_angle_deg is not None]

    if method == "median":
        mean_r = float(np.median(radii_nm))
        std_r = float(np.median(np.abs(radii_nm - mean_r)))
    else:
        mean_r = float(np.mean(radii_nm))
        std_r = float(np.std(radii_nm))

    return {
        "mean_radius_nm": mean_r,
        "std_radius_nm": std_r,
        "mean_radius_angstrom": mean_r * 10.0,
        "mean_opening_angle_deg": float(np.mean(angles)) if angles else None,
        "count": len(results),
    }


def classify_tip_condition(radius_nm: float, config: dict) -> TipCondition:
    """Classify tip as Sharp / Moderate / Blunt based on radius thresholds."""
    cfg = config.get("tip_classification", {})
    sharp_max = cfg.get("sharp_max_nm", 10.0)
    moderate_max = cfg.get("moderate_max_nm", 50.0)

    if radius_nm <= sharp_max:
        return TipCondition.SHARP
    if radius_nm <= moderate_max:
        return TipCondition.MODERATE
    return TipCondition.BLUNT
