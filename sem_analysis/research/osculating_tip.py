"""Research-grade osculating circle tip radius measurement."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sem_analysis.radius_computation import (
    _circle_overlap_confidence,
    geometric_circle_fit,
    hough_circle_fit,
    taubin_circle_fit,
)
from sem_analysis.research.contour_extraction import extract_left_right_contours, merge_contour_arc
from sem_analysis.research.curvature import find_curvature_tip
from sem_analysis.research.geometric_validation import (
    composite_confidence,
    expected_radius_from_geometry,
    validate_geometry,
)
from sem_analysis.research.line_fitting import fit_line_ransac, line_intersection


@dataclass
class OsculatingTipResult:
    """Full research-grade measurement for one serration tip."""

    peak_id: int
    peak_location: tuple[float, float]
    virtual_apex: tuple[float, float]
    physical_tip: tuple[float, float]
    included_angle_deg: float
    distance_l_px: float
    distance_l_nm: float
    center: tuple[float, float]
    radius_px: float
    radius_nm: float
    radius_um: float
    fit_residual_px: float
    fit_residual_nm: float
    confidence_score: float
    geometric_valid: bool
    expected_radius_px: float | None
    geometric_error: float | None
    curvature_kappa: float
    left_line: list[float] = field(default_factory=list)
    right_line: list[float] = field(default_factory=list)
    vertical_l_line: list[float] = field(default_factory=list)
    inlier_fraction_left: float = 0.0
    inlier_fraction_right: float = 0.0
    method: str = "osculating_circle"
    rejected: bool = False
    rejection_reason: str | None = None


def _included_angle(m_left: float, m_right: float) -> float:
    import math
    a1 = math.atan(m_left)
    a2 = math.atan(m_right)
    angle = abs(a2 - a1)
    if angle > math.pi:
        angle = 2 * math.pi - angle
    return math.degrees(angle)


def _select_tip_region_points(
    contour: np.ndarray,
    tip: tuple[float, float],
    radius_hint: float,
    window_factor: float = 1.5,
) -> np.ndarray:
    pts = contour.reshape(-1, 2)
    tx, ty = tip
    window = max(8.0, radius_hint * window_factor)
    near = pts[
        (np.abs(pts[:, 0] - tx) < window)
        & (pts[:, 1] >= ty - 5)
        & (pts[:, 1] <= ty + window)
    ]
    return near if len(near) >= 5 else pts


def measure_osculating_tip(
    image: np.ndarray,
    edge_points: np.ndarray,
    peak_id: int,
    peak: tuple[float, float],
    nm_per_pixel: float,
    config: dict,
) -> OsculatingTipResult | None:
    """
    Estimate osculating circle at blade tip using the full research pipeline:

    Canny contour → RANSAC flank lines → virtual apex → curvature tip →
    Hough initial guess → nonlinear refinement → geometric validation → confidence.
    """
    cfg = config.get("research_grade", {})
    method_cfg = config.get("measurement_methods", {})
    window_y = method_cfg.get("local_contour_window_y_px", 80.0)
    window_x = method_cfg.get("local_contour_window_x_px", 40.0)
    apex_exclusion = cfg.get("apex_exclusion_px", 15.0)
    flank_fraction = cfg.get("flank_fraction", 0.35)

    split = extract_left_right_contours(
        edge_points, peak, window_y_px=window_y, window_x_px=window_x
    )
    if split is None:
        return None
    left_pts, right_pts = split
    contour = merge_contour_arc(left_pts, right_pts)

    # Curvature-based physical tip (Stage 7)
    curv = find_curvature_tip(
        contour,
        peak_hint=peak,
        tip_region_fraction=cfg.get("tip_region_fraction", 0.35),
        smooth_window=cfg.get("curvature_smooth_window", 5),
    )
    if curv is None:
        physical_tip = peak
        kappa_max = 0.0
    else:
        physical_tip, kappa_max, _ = curv

    # RANSAC flank fitting (Stage 5) — exclude rounded apex
    left_flank = left_pts[left_pts[:, 1] > physical_tip[1] + apex_exclusion]
    right_flank = right_pts[right_pts[:, 1] > physical_tip[1] + apex_exclusion]
    if len(left_flank) < 5:
        left_flank = left_pts[left_pts[:, 1] >= physical_tip[1]]
    if len(right_flank) < 5:
        right_flank = right_pts[right_pts[:, 1] >= physical_tip[1]]
    if len(left_flank) < 5 or len(right_flank) < 5:
        return None

    n_l = max(5, int(len(left_flank) * flank_fraction))
    n_r = max(5, int(len(right_flank) * flank_fraction))
    left_upper = left_flank[np.argsort(left_flank[:, 1])[:n_l]]
    right_upper = right_flank[np.argsort(right_flank[:, 1])[:n_r]]

    ransac_thresh = cfg.get("ransac_threshold_px", 2.0)
    left_fit = fit_line_ransac(left_upper, threshold=ransac_thresh)
    right_fit = fit_line_ransac(right_upper, threshold=ransac_thresh)
    if left_fit is None or right_fit is None:
        return None

    m_l, b_l, in_l, left_line = left_fit
    m_r, b_r, in_r, right_line = right_fit

    # Virtual apex (Stage 6)
    virtual = line_intersection(m_l, b_l, m_r, b_r)
    if virtual is None:
        virtual = (physical_tip[0], m_l * physical_tip[0] + b_l)

    included_angle = _included_angle(m_l, m_r)
    conv_y_at_tip = m_l * physical_tip[0] + b_l
    distance_l_px = abs(physical_tip[1] - conv_y_at_tip)
    vertical_l = [physical_tip[0], conv_y_at_tip, physical_tip[0], physical_tip[1]]

    # Hough initial guess (Stage 8)
    hough_guess = hough_circle_fit(image, physical_tip, edge_points, config)
    radius_hint = hough_guess[1] if hough_guess else max(10.0, distance_l_px)

    # Tip region for refinement (Stage 9)
    tip_region = _select_tip_region_points(
        contour, physical_tip, radius_hint, window_factor=cfg.get("tip_window_factor", 1.5)
    )

    center = None
    radius_px = None
    residual = float("inf")
    fit_method = "osculating_circle"

    if hough_guess:
        center, radius_px, _ = hough_guess

    if len(tip_region) >= 5:
        try:
            if center is not None:
                init_pts = tip_region
            else:
                init_pts = tip_region
            center, radius_px, residual = geometric_circle_fit(init_pts)
            fit_method = "osculating_circle_refined"
        except (ValueError, np.linalg.LinAlgError):
            if center is None and len(tip_region) >= 3:
                try:
                    center, radius_px, residual = taubin_circle_fit(tip_region)
                    fit_method = "osculating_circle_taubin"
                except (ValueError, np.linalg.LinAlgError):
                    return None
            elif center is None:
                return None
    elif center is None:
        return None

    # Geometric validation (Stage 10)
    geo_valid, expected_r, geo_error = validate_geometry(
        radius_px,
        distance_l_px,
        included_angle,
        tolerance_ratio=cfg.get("geometry_tolerance_ratio", 0.25),
    )

    overlap = _circle_overlap_confidence(center, radius_px, edge_points)
    inlier_frac = (float(np.mean(in_l)) + float(np.mean(in_r))) / 2.0

    confidence = composite_confidence(
        residual,
        overlap,
        inlier_frac,
        geo_valid,
        geo_error,
        weights=cfg.get("confidence_weights"),
    )

    min_confidence = cfg.get("min_confidence", 0.3)
    rejected = confidence < min_confidence
    rejection_reason = "low_confidence" if rejected else None

    radius_nm = radius_px * nm_per_pixel
    return OsculatingTipResult(
        peak_id=peak_id,
        peak_location=peak,
        virtual_apex=virtual,
        physical_tip=physical_tip,
        included_angle_deg=included_angle,
        distance_l_px=distance_l_px,
        distance_l_nm=distance_l_px * nm_per_pixel,
        center=center,
        radius_px=radius_px,
        radius_nm=radius_nm,
        radius_um=radius_nm / 1000.0,
        fit_residual_px=residual,
        fit_residual_nm=residual * nm_per_pixel,
        confidence_score=confidence,
        geometric_valid=geo_valid,
        expected_radius_px=expected_r,
        geometric_error=geo_error,
        curvature_kappa=kappa_max,
        left_line=left_line,
        right_line=right_line,
        vertical_l_line=vertical_l,
        inlier_fraction_left=float(np.mean(in_l)),
        inlier_fraction_right=float(np.mean(in_r)),
        method=fit_method,
        rejected=rejected,
        rejection_reason=rejection_reason,
    )


def measure_all_osculating_tips(
    image: np.ndarray,
    peaks: np.ndarray,
    edge_points: np.ndarray,
    nm_per_pixel: float,
    config: dict,
) -> tuple[list[OsculatingTipResult], dict]:
    """Run osculating circle measurement on every peak."""
    results: list[OsculatingTipResult] = []
    for i, peak in enumerate(peaks):
        px, py = float(peak[0]), float(peak[1])
        r = measure_osculating_tip(image, edge_points, i, (px, py), nm_per_pixel, config)
        if r:
            results.append(r)

    accepted = [r for r in results if not r.rejected]
    radii = [r.radius_nm for r in accepted]
    confidences = [r.confidence_score for r in accepted]

    summary = {
        "count": len(results),
        "accepted_count": len(accepted),
        "mean_radius_nm": float(np.mean(radii)) if radii else None,
        "mean_radius_um": float(np.mean(radii)) / 1000.0 if radii else None,
        "mean_confidence": float(np.mean(confidences)) if confidences else None,
        "mean_included_angle_deg": (
            float(np.mean([r.included_angle_deg for r in accepted])) if accepted else None
        ),
    }
    return results, summary


def osculating_tip_to_dict(result: OsculatingTipResult) -> dict:
    return {
        "peak_id": result.peak_id,
        "peak_location": [float(result.peak_location[0]), float(result.peak_location[1])],
        "virtual_apex": [float(result.virtual_apex[0]), float(result.virtual_apex[1])],
        "physical_tip": [float(result.physical_tip[0]), float(result.physical_tip[1])],
        "included_angle_deg": float(result.included_angle_deg),
        "distance_l_px": float(result.distance_l_px),
        "distance_l_nm": float(result.distance_l_nm),
        "radius_px": float(result.radius_px),
        "radius_nm": float(result.radius_nm),
        "radius_um": float(result.radius_um),
        "center": [float(result.center[0]), float(result.center[1])],
        "fit_residual_px": float(result.fit_residual_px),
        "fit_residual_nm": float(result.fit_residual_nm),
        "confidence_score": float(result.confidence_score),
        "geometric_valid": result.geometric_valid,
        "expected_radius_px": float(result.expected_radius_px) if result.expected_radius_px else None,
        "geometric_error": float(result.geometric_error) if result.geometric_error is not None else None,
        "curvature_kappa": float(result.curvature_kappa),
        "left_line": [float(v) for v in result.left_line],
        "right_line": [float(v) for v in result.right_line],
        "vertical_l_line": [float(v) for v in result.vertical_l_line],
        "inlier_fraction_left": float(result.inlier_fraction_left),
        "inlier_fraction_right": float(result.inlier_fraction_right),
        "method": result.method,
        "rejected": result.rejected,
        "rejection_reason": result.rejection_reason,
    }
