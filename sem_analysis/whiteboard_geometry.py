"""Whiteboard geometry matching the annotated SEM reference image.

Visual construction (yellow V + cyan circle + red α/d):
  1. TLS flank lines → projected tip (virtual apex)
  2. Included angle α at the projected tip
  3. Distance d from projected tip to ultimate tip (along tip axis)
  4. Inscribed circle tangent to both yellow flanks, centre on the angle bisector
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sem_analysis.blade_value import flank_included_angle_deg
from sem_analysis.edge_geometry import fit_line_tls, intersect_lines, local_symmetry_axis


@dataclass
class WhiteboardGeometry:
    """Drawable + measurable constructs for one tip (image coordinates)."""

    tip_id: int
    ultimate_tip: tuple[float, float]
    projected_tip: tuple[float, float]
    left_line: list[float]  # x1,y1,x2,y2 extended through projected tip
    right_line: list[float]
    edge_left: list[list[float]]  # polyline [[x,y], ...]
    edge_right: list[list[float]]
    included_angle_deg: float
    d_px: float
    d_nm: float
    circle_center: tuple[float, float]
    circle_radius_px: float
    circle_radius_nm: float
    radius_spoke: list[float]  # center → rim (horizontal-ish)
    diameter_line: list[float]
    d_bracket: list[float]  # projected → ultimate (or circle top)
    alpha_arc: dict  # center, radius, start_deg, end_deg
    valid: bool = True
    rejection_reason: str | None = None


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(2)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([0.0, 1.0])
    return v / n


def _orient_down(direction: np.ndarray) -> np.ndarray:
    d = _unit(direction)
    return d if d[1] >= 0 else -d


def _extend_line_through(
    projected: np.ndarray,
    direction: np.ndarray,
    length_up: float,
    length_down: float,
) -> list[float]:
    """Yellow flank: from above projected tip down into the tip body."""
    d = _unit(direction)
    # Ensure one end goes "up" (smaller y) and one "down"
    if d[1] < 0:
        d = -d  # d points down (+y)
    p_up = projected - d * length_up
    p_down = projected + d * length_down
    return [float(p_up[0]), float(p_up[1]), float(p_down[0]), float(p_down[1])]


def _angle_of(vec: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(vec[1], vec[0])))


def inscribed_circle_tangent_to_flanks(
    projected: np.ndarray,
    dir_left: np.ndarray,
    dir_right: np.ndarray,
    ultimate_tip: np.ndarray,
    nm_per_px: float,
    radius_px_override: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """
    Circle tangent to both flank lines; centre on the tip angle bisector.

    If radius_px_override is set (e.g. Method-1 R), place that circle on the bisector.
    Otherwise derive R from d = |projected − ultimate| along the bisector:

        d = R / sin(α/2) − R  ⇒  R = d · sin(α/2) / (1 − sin(α/2))

    (top of circle near the ultimate tip — matches the whiteboard sketch).
    """
    projected = np.asarray(projected, dtype=float).reshape(2)
    tip = np.asarray(ultimate_tip, dtype=float).reshape(2)

    # Directions from projected tip into the tip body (toward / past ultimate tip)
    into = tip - projected
    if np.linalg.norm(into) < 1e-9:
        into = np.array([0.0, 1.0])

    dl = _orient_down(dir_left)
    dr = _orient_down(dir_right)
    # Flip if pointing away from tip body
    if np.dot(dl, into) < 0:
        dl = -dl
    if np.dot(dr, into) < 0:
        dr = -dr

    alpha = flank_included_angle_deg(dl, dr)
    half = np.radians(alpha / 2.0)
    sin_h = float(np.sin(half))
    if sin_h < 1e-6:
        raise ValueError("Included angle too small for inscribed circle.")

    bisector = _unit(dl + dr)
    if np.dot(bisector, into) < 0:
        bisector = -bisector

    d_along = float(np.dot(tip - projected, bisector))
    if d_along < 1e-6:
        d_along = float(np.linalg.norm(tip - projected))

    if radius_px_override is not None and radius_px_override > 0:
        r_px = float(radius_px_override)
    else:
        # Top of circle ≈ ultimate tip
        denom = 1.0 / sin_h - 1.0
        if denom < 1e-6:
            raise ValueError("Degenerate α for R(d) formula.")
        r_px = d_along / denom

    # Centre is R / sin(α/2) from projected tip along bisector
    dist_pc = r_px / sin_h
    center = projected + bisector * dist_pc
    return center, float(r_px), float(alpha)


def build_whiteboard_geometry(
    tip_id: int,
    apex: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    nm_per_px: float,
    fit_band_nm: tuple[float, float] = (50.0, 200.0),
    method1_radius_px: float | None = None,
    min_flank_points: int = 5,
    min_cross: float = 0.15,
    line_extend_up_px: float = 120.0,
    line_extend_down_px: float = 200.0,
) -> WhiteboardGeometry | None:
    """Build the full whiteboard construct for one validated tip."""
    apex = np.asarray(apex, dtype=float).reshape(2)
    left = np.asarray(left, dtype=float).reshape(-1, 2)
    right = np.asarray(right, dtype=float).reshape(-1, 2)
    axis = local_symmetry_axis(left, right, apex)

    y0, y1 = float(fit_band_nm[0]), float(fit_band_nm[1])
    d0 = y0 / max(nm_per_px, 1e-9)
    d1 = y1 / max(nm_per_px, 1e-9)
    depth_l = (left - apex) @ axis
    depth_r = (right - apex) @ axis
    left_band = left[(depth_l >= d0) & (depth_l <= d1)]
    right_band = right[(depth_r >= d0) & (depth_r <= d1)]
    if len(left_band) < min_flank_points or len(right_band) < min_flank_points:
        return None

    try:
        c_l, dir_l = fit_line_tls(left_band)
        c_r, dir_r = fit_line_tls(right_band)
        projected = intersect_lines(c_l, dir_l, c_r, dir_r, min_cross=min_cross)
        center, r_px, alpha = inscribed_circle_tangent_to_flanks(
            projected, dir_l, dir_r, apex, nm_per_px, radius_px_override=method1_radius_px
        )
    except ValueError:
        return None

    d_px = float(np.linalg.norm(apex - projected))
    # Prefer axis-aligned d (signed projection) for the bracket
    d_axis = abs(float(np.dot(apex - projected, axis)))
    if d_axis > 1e-6:
        d_px = d_axis

    # Yellow lines through projected tip along TLS directions
    left_line = _extend_line_through(projected, dir_l, line_extend_up_px, line_extend_down_px)
    right_line = _extend_line_through(projected, dir_r, line_extend_up_px, line_extend_down_px)

    # Radius spoke: horizontal from center to right rim (as in the sketch)
    spoke_end = center + np.array([r_px, 0.0])
    diameter = [float(center[0] - r_px), float(center[1]), float(center[0] + r_px), float(center[1])]

    # α arc at projected tip
    into = apex - projected
    dl = _orient_down(dir_l)
    dr = _orient_down(dir_r)
    if np.dot(dl, into) < 0:
        dl = -dl
    if np.dot(dr, into) < 0:
        dr = -dr
    a1 = _angle_of(dl)
    a2 = _angle_of(dr)
    # OpenCV ellipse uses degrees, CCW from +x; draw the smaller interior arc
    start, end = sorted([a1, a2])
    if end - start > 180:
        start, end = end, start + 360
    arc_r = max(18.0, min(45.0, 0.35 * d_px + 12.0))

    # Subsample edges for drawing
    def _poly(pts: np.ndarray, max_n: int = 80) -> list[list[float]]:
        if len(pts) <= max_n:
            return pts.astype(float).tolist()
        idx = np.linspace(0, len(pts) - 1, max_n).astype(int)
        return pts[idx].astype(float).tolist()

    return WhiteboardGeometry(
        tip_id=tip_id,
        ultimate_tip=(float(apex[0]), float(apex[1])),
        projected_tip=(float(projected[0]), float(projected[1])),
        left_line=left_line,
        right_line=right_line,
        edge_left=_poly(left),
        edge_right=_poly(right),
        included_angle_deg=float(alpha),
        d_px=float(d_px),
        d_nm=float(d_px * nm_per_px),
        circle_center=(float(center[0]), float(center[1])),
        circle_radius_px=float(r_px),
        circle_radius_nm=float(r_px * nm_per_px),
        radius_spoke=[float(center[0]), float(center[1]), float(spoke_end[0]), float(spoke_end[1])],
        diameter_line=diameter,
        d_bracket=[
            float(projected[0]), float(projected[1]),
            float(apex[0]), float(apex[1]),
        ],
        alpha_arc={
            "center": [float(projected[0]), float(projected[1])],
            "radius": float(arc_r),
            "start_deg": float(start),
            "end_deg": float(end),
        },
        valid=True,
    )


def whiteboard_to_dict(g: WhiteboardGeometry) -> dict:
    return {
        "tip_id": g.tip_id,
        "ultimate_tip": list(g.ultimate_tip),
        "projected_tip": list(g.projected_tip),
        "peak_location": list(g.ultimate_tip),
        "left_line": g.left_line,
        "right_line": g.right_line,
        "edge_left": g.edge_left,
        "edge_right": g.edge_right,
        "included_angle_deg": g.included_angle_deg,
        "distance_l_nm": g.d_nm,
        "distance_l_px": g.d_px,
        "d_nm": g.d_nm,
        "d_px": g.d_px,
        "circle_center": list(g.circle_center),
        "center": list(g.circle_center),
        "circle_radius_px": g.circle_radius_px,
        "radius_px": g.circle_radius_px,
        "radius_nm": g.circle_radius_nm,
        "radius_spoke": g.radius_spoke,
        "diameter_line": g.diameter_line,
        "d_bracket": g.d_bracket,
        "vertical_l_line": g.d_bracket,
        "alpha_arc": g.alpha_arc,
        "valid": g.valid,
        "rejection_reason": g.rejection_reason,
    }
