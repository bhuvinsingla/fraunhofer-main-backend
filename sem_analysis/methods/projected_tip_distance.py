"""Method 2 — Distance from 'projected' tip (PDF brainstorming).

Procedure (image coordinates, tip opens downward / apex has smaller y):
  1. Fit yellow lines to left/right flanks in a depth band below the apex
  2. Extend those lines to their convergent point (projected tip) above the apex
  3. Draw a vertical red line from the projected tip down to the ultimate tip (blue)
  4. Output l = vertical length of that red line (nm)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sem_analysis.blade_value import (
    area_between_line_and_edge_nm2,
    flank_included_angle_deg,
)
from sem_analysis.edge_geometry import fit_line_tls, intersect_lines


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
    tip_apex_arc: list = field(default_factory=list)  # blue ultimate-tip curve
    included_angle_deg: float | None = None
    area_under_curve_nm2: float | None = None
    valid: bool = False
    rejection_reason: str | None = None
    method: str = "projected_tip_distance"
    left_edge_slope: float = 0.0
    right_edge_slope: float = 0.0
    line_r2_left: float = 0.0
    line_r2_right: float = 0.0


ProjectedTipDistanceResult = Method2Result

# Fallback bands (nm below apex) tried in order until a valid intersection is found
_DEFAULT_BANDS: tuple[tuple[float, float], ...] = (
    (50.0, 200.0),
    (30.0, 150.0),
    (20.0, 120.0),
    (40.0, 250.0),
    (15.0, 100.0),
)


def _points_in_vertical_band(
    branch: np.ndarray,
    apex: np.ndarray,
    y0_nm: float,
    y1_nm: float,
    nm_per_px: float,
) -> np.ndarray:
    """Flank points whose image-Y depth below the apex is in [y0, y1] nm."""
    pts = np.asarray(branch, dtype=float).reshape(-1, 2)
    apex = np.asarray(apex, dtype=float).reshape(2)
    if len(pts) == 0:
        return pts
    dy = pts[:, 1] - apex[1]  # +down in image coords
    d0 = y0_nm / max(nm_per_px, 1e-9)
    d1 = y1_nm / max(nm_per_px, 1e-9)
    lo, hi = min(d0, d1), max(d0, d1)
    return pts[(dy >= lo) & (dy <= hi)]


def _line_through(c: np.ndarray, d: np.ndarray, p_deep: np.ndarray, p_proj: np.ndarray) -> list[float]:
    """Segment from deep flank toward / through the projected tip (PDF yellow V)."""
    # Prefer endpoints: deep flank point → projected tip (extend past if needed)
    v = p_proj - p_deep
    if float(np.linalg.norm(v)) < 1e-6:
        p0 = c - d * 40.0
        p1 = c + d * 40.0
        return [float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1])]
    # Extend slightly past projected tip for a clear V
    extra = 0.15 * float(np.linalg.norm(v))
    u = v / np.linalg.norm(v)
    p_end = p_proj + u * extra
    return [float(p_deep[0]), float(p_deep[1]), float(p_end[0]), float(p_end[1])]


def _tip_apex_arc(
    left: np.ndarray,
    right: np.ndarray,
    apex: np.ndarray,
    radius_px: float = 25.0,
) -> list[list[float]]:
    """Blue ultimate-tip curve: nearby edge points sorted by x around the apex."""
    pts = np.vstack(
        [
            np.asarray(left, dtype=float).reshape(-1, 2),
            np.asarray(right, dtype=float).reshape(-1, 2),
        ]
    )
    if len(pts) == 0:
        return [[float(apex[0]), float(apex[1])]]
    d = np.linalg.norm(pts - apex.reshape(1, 2), axis=1)
    near = pts[d <= radius_px]
    if len(near) < 3:
        near = pts[np.argsort(d)[: max(5, min(12, len(pts)))]]
    # Keep the uppermost band (rounded tip)
    y_cut = float(np.percentile(near[:, 1], 55))
    tip = near[near[:, 1] <= y_cut]
    if len(tip) < 3:
        tip = near
    tip = tip[np.argsort(tip[:, 0])]
    # Subsample for a clean polyline
    if len(tip) > 24:
        idx = np.linspace(0, len(tip) - 1, 24, dtype=int)
        tip = tip[idx]
    return tip.tolist()


def _try_band(
    apex: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    nm_per_pixel: float,
    y0_nm: float,
    y1_nm: float,
    min_flank_points: int,
    min_cross: float,
) -> tuple | None:
    left_band = _points_in_vertical_band(left, apex, y0_nm, y1_nm, nm_per_pixel)
    right_band = _points_in_vertical_band(right, apex, y0_nm, y1_nm, nm_per_pixel)
    if len(left_band) < min_flank_points or len(right_band) < min_flank_points:
        return None
    try:
        c_l, d_l = fit_line_tls(left_band)
        c_r, d_r = fit_line_tls(right_band)
        projected = intersect_lines(c_l, d_l, c_r, d_r, min_cross=min_cross)
    except ValueError:
        return None
    return left_band, right_band, c_l, d_l, c_r, d_r, projected


def measure_projected_tip_distance(
    contour: np.ndarray | None = None,
    nm_per_pixel: float = 1.0,
    fit_band_nm: tuple[float, float] | list[float] = (50.0, 200.0),
    min_flank_points: int = 4,
    apex: np.ndarray | None = None,
    left: np.ndarray | None = None,
    right: np.ndarray | None = None,
    tip_id: int = 0,
    min_cross: float = 0.08,
    max_distance_nm: float = 250.0,
    **legacy_kwargs,
) -> Method2Result:
    """
    PDF Method 2: yellow flank projection → convergent point → vertical red l → tip.
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
    left = np.asarray(left, dtype=float).reshape(-1, 2)
    right = np.asarray(right, dtype=float).reshape(-1, 2)
    result.tip_point = (float(apex[0]), float(apex[1]))

    # Try primary band first, then fallbacks
    bands: list[tuple[float, float]] = [(y0_nm, y1_nm)]
    for b in _DEFAULT_BANDS:
        if b not in bands and (b[0], b[1]) != (y0_nm, y1_nm):
            bands.append(b)

    chosen = None
    used_band = (y0_nm, y1_nm)
    for b0, b1 in bands:
        for mc in (min_cross, 0.05, 0.03):
            trial = _try_band(
                apex, left, right, nm_per_pixel, b0, b1, min_flank_points, mc
            )
            if trial is None:
                continue
            left_band, right_band, c_l, d_l, c_r, d_r, projected = trial
            # PDF: convergent (projected) tip is ABOVE the ultimate tip → smaller image-y
            if float(projected[1]) >= float(apex[1]) - 0.25:
                # Reject intersections at/below the physical tip
                continue
            # Reject far-away projections (near-parallel flanks give runaway l)
            cand_nm = abs(float(apex[1]) - float(projected[1])) * nm_per_pixel
            if cand_nm > max_distance_nm:
                continue
            chosen = (left_band, right_band, c_l, d_l, c_r, d_r, projected)
            used_band = (b0, b1)
            break
        if chosen is not None:
            break

    if chosen is None:
        result.rejection_reason = "insufficient_flank_or_no_intersection"
        return result

    left_band, right_band, c_l, d_l, c_r, d_r, projected = chosen
    result.fit_band_nm = used_band

    # Vertical distance (PDF: length of red line l)
    distance_px = abs(float(apex[1]) - float(projected[1]))
    distance_nm = distance_px * nm_per_pixel
    if distance_nm > max_distance_nm:
        result.rejection_reason = "distance_implausible"
        return result
    # Near-zero l is valid for an almost-sharp tip
    if distance_px < 0.25:
        distance_px = 0.25
        distance_nm = distance_px * nm_per_pixel

    # Deep flank anchors for yellow V (farthest below apex in band)
    deep_l = left_band[int(np.argmax(left_band[:, 1]))]
    deep_r = right_band[int(np.argmax(right_band[:, 1]))]

    result.convergence_point = (float(projected[0]), float(projected[1]))
    result.distance_px = float(distance_px)
    result.distance_nm = float(distance_nm)
    result.left_line = _line_through(c_l, d_l, deep_l, projected)
    result.right_line = _line_through(c_r, d_r, deep_r, projected)

    # Vertical red l at projected x (PDF: down from convergent point to tip height)
    x_l = float(projected[0])
    result.vertical_l_line = [
        x_l,
        float(projected[1]),
        x_l,
        float(apex[1]),
    ]

    tip_arc_px = max(12.0, min(40.0, 30.0 / max(nm_per_pixel, 1e-9)))
    result.tip_apex_arc = _tip_apex_arc(left, right, apex, radius_px=tip_arc_px)

    try:
        result.included_angle_deg = flank_included_angle_deg(d_l, d_r)
    except ValueError:
        result.included_angle_deg = None
    area_l = area_between_line_and_edge_nm2(left_band, c_l, d_l, nm_per_pixel)
    area_r = area_between_line_and_edge_nm2(right_band, c_r, d_r, nm_per_pixel)
    result.area_under_curve_nm2 = float(area_l + area_r)

    # Slopes for diagnostics (dy/dx in image coords)
    if abs(float(d_l[0])) > 1e-9:
        result.left_edge_slope = float(d_l[1] / d_l[0])
    if abs(float(d_r[0])) > 1e-9:
        result.right_edge_slope = float(d_r[1] / d_r[0])

    result.valid = True
    return result


def projected_tip_distance_to_dict(result: Method2Result) -> dict:
    return {
        "tip_id": result.tip_id,
        "distance_l_nm": result.distance_nm,
        "distance_l_px": result.distance_px,
        "fit_band_nm": [result.fit_band_nm[0], result.fit_band_nm[1]],
        "tip_point": list(result.tip_point),
        "peak_location": list(result.tip_point),
        "convergence_point": list(result.convergence_point) if result.convergence_point else None,
        "projected_tip": list(result.convergence_point) if result.convergence_point else None,
        "left_line": result.left_line,
        "right_line": result.right_line,
        "vertical_l_line": result.vertical_l_line,
        "tip_apex_arc": result.tip_apex_arc,
        "included_angle_deg": result.included_angle_deg,
        "area_under_curve_nm2": result.area_under_curve_nm2,
        "valid": result.valid,
        "rejection_reason": result.rejection_reason,
    }
