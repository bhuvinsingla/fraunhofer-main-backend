"""Method 2 — projected tip distance via TLS lines and signed axis projection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sem_analysis.blade_value import (
    area_between_line_and_edge_nm2,
    flank_included_angle_deg,
)
from sem_analysis.edge_geometry import fit_line_tls, intersect_lines, local_symmetry_axis


@dataclass
class Method2Result:
    tip_id: int
    tip_point: tuple[float, float]
    convergence_point: tuple[float, float] | None
    distance_px: float | None
    distance_nm: float | None
    fit_band_nm: tuple[float, float]
    left_line: list[float] = field(default_factory=list)
    right_line: list[float] = field(default_factory=list)
    vertical_l_line: list[float] = field(default_factory=list)
    included_angle_deg: float | None = None
    area_under_curve_nm2: float | None = None
    valid: bool = False
    rejection_reason: str | None = None
    method: str = "projected_tip_distance"
    # Legacy fields for annotation
    left_edge_slope: float = 0.0
    right_edge_slope: float = 0.0
    tip_apex_arc: object | None = None
    line_r2_left: float = 0.0
    line_r2_right: float = 0.0


ProjectedTipDistanceResult = Method2Result


def _points_in_band(
    branch: np.ndarray,
    apex: np.ndarray,
    axis: np.ndarray,
    y0_nm: float,
    y1_nm: float,
    nm_per_px: float,
) -> np.ndarray:
    """Select branch points whose depth along tip axis is in [y0, y1] nm."""
    pts = np.asarray(branch, dtype=float).reshape(-1, 2)
    apex = np.asarray(apex, dtype=float).reshape(2)
    depth_px = (pts - apex) @ axis
    d0 = y0_nm / max(nm_per_px, 1e-9)
    d1 = y1_nm / max(nm_per_px, 1e-9)
    mask = (depth_px >= d0) & (depth_px <= d1)
    return pts[mask]


def measure_projected_tip_distance(
    contour: np.ndarray | None = None,
    nm_per_pixel: float = 1.0,
    fit_band_nm: tuple[float, float] | list[float] = (50.0, 200.0),
    min_flank_points: int = 5,
    apex: np.ndarray | None = None,
    left: np.ndarray | None = None,
    right: np.ndarray | None = None,
    tip_id: int = 0,
    min_cross: float = 0.15,
    **legacy_kwargs,
) -> Method2Result | None:
    """
    TLS flank fits → intersection → signed distance along local symmetry axis.
    Reject near-parallel lines; never clip impossible values.
    """
    _ = contour, legacy_kwargs
    band = list(fit_band_nm)
    y0_nm, y1_nm = float(band[0]), float(band[1])
    result = Method2Result(
        tip_id=tip_id,
        tip_point=(0.0, 0.0),
        convergence_point=None,
        distance_px=None,
        distance_nm=None,
        fit_band_nm=(y0_nm, y1_nm),
    )

    if apex is None or left is None or right is None:
        if contour is None:
            result.rejection_reason = "missing_geometry"
            return result
        pts = np.asarray(contour, dtype=float).reshape(-1, 2)
        apex_idx = int(np.argmin(pts[:, 1]))
        apex = pts[apex_idx]
        left = pts[pts[:, 0] < apex[0]]
        right = pts[pts[:, 0] > apex[0]]

    apex = np.asarray(apex, dtype=float).reshape(2)
    result.tip_point = (float(apex[0]), float(apex[1]))
    axis = local_symmetry_axis(left, right, apex)

    left_band = _points_in_band(left, apex, axis, y0_nm, y1_nm, nm_per_pixel)
    right_band = _points_in_band(right, apex, axis, y0_nm, y1_nm, nm_per_pixel)
    if len(left_band) < min_flank_points or len(right_band) < min_flank_points:
        result.rejection_reason = "insufficient_flank_points"
        return result

    try:
        c_l, d_l = fit_line_tls(left_band)
        c_r, d_r = fit_line_tls(right_band)
        projected = intersect_lines(c_l, d_l, c_r, d_r, min_cross=min_cross)
    except ValueError as exc:
        result.rejection_reason = str(exc)
        return result

    # Intersection should be above (toward -axis from deep flanks) relative to apex
    # along axis: projected should have smaller depth than apex mid-flank, typically
    # "above" apex means opposite to +axis (into the tip).
    # Reject if intersection is below the actual apex along +axis
    depth_proj = float(np.dot(projected - apex, axis))
    if depth_proj > 5.0:  # more than 5 px below apex along tip axis
        result.rejection_reason = "intersection_below_apex"
        return result

    # Signed projection along local symmetry axis
    distance_px = abs(float(np.dot(apex - projected, axis)))
    # Do not clip — reject absurd distances
    max_plausible_nm = 2000.0
    distance_nm = distance_px * nm_per_pixel
    if distance_nm > max_plausible_nm:
        result.rejection_reason = "distance_implausible"
        return result

    # Drawable segments
    def _seg(c, d, scale=80.0):
        p0 = c - d * scale
        p1 = c + d * scale
        return [float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1])]

    result.convergence_point = (float(projected[0]), float(projected[1]))
    result.distance_px = distance_px
    result.distance_nm = distance_nm
    result.left_line = _seg(c_l, d_l)
    result.right_line = _seg(c_r, d_r)
    result.vertical_l_line = [
        float(apex[0]), float(apex[1]),
        float(projected[0]), float(projected[1]),
    ]
    try:
        result.included_angle_deg = flank_included_angle_deg(d_l, d_r)
    except ValueError:
        result.included_angle_deg = None
    area_l = area_between_line_and_edge_nm2(left_band, c_l, d_l, nm_per_pixel)
    area_r = area_between_line_and_edge_nm2(right_band, c_r, d_r, nm_per_pixel)
    result.area_under_curve_nm2 = float(area_l + area_r)
    result.valid = True
    return result


def projected_tip_distance_to_dict(result: Method2Result) -> dict:
    return {
        "tip_id": result.tip_id,
        "distance_l_nm": result.distance_nm,
        "distance_l_px": result.distance_px,
        "fit_band_nm": [result.fit_band_nm[0], result.fit_band_nm[1]],
        "tip_point": list(result.tip_point),
        "convergence_point": list(result.convergence_point) if result.convergence_point else None,
        "left_line": result.left_line,
        "right_line": result.right_line,
        "vertical_l_line": result.vertical_l_line,
        "included_angle_deg": result.included_angle_deg,
        "area_under_curve_nm2": result.area_under_curve_nm2,
        "valid": result.valid,
        "rejection_reason": result.rejection_reason,
    }
