"""Method 2 — distance from projected (theoretical) tip → length l.

Shared upstream: ultimate tip T and left/right edge contours.

Procedure:
  1. Fit straight lines to the approximately linear flank segments of edge_L
     and edge_R (depth band below T — protocol.method2_fit_band_nm).
  2. Extrapolate both lines upward until they meet at P_proj (zero-rounding apex).
  3. Vertical distance from P_proj down to actual tip T.
  4. Output = length l = |P_proj − T| (vertical / straight-line in image coords).

Larger l = blunter / more rounded tip (more “rounding depth”).
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
    n_left_band: int = 0
    n_right_band: int = 0
    fit_strategy: str | None = None
    attempts: list = field(default_factory=list)


ProjectedTipDistanceResult = Method2Result

# Prefer mid-flank bands that skip tip rounding and stop before the next serration.
# Deep (0–*) or very deep bands often fit Λ valley walls → projected_not_above_tip.
_DEFAULT_BANDS: tuple[tuple[float, float], ...] = (
    (25.0, 100.0),
    (30.0, 120.0),
    (20.0, 90.0),
    (35.0, 140.0),
    (40.0, 160.0),
    (50.0, 200.0),
    (15.0, 70.0),
    (18.0, 80.0),
    (12.0, 55.0),
    (10.0, 50.0),
    (8.0, 45.0),
    (22.0, 110.0),
    (28.0, 95.0),
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


def _outer_flank_points(band: np.ndarray, apex: np.ndarray, side: str) -> np.ndarray:
    """Keep the more lateral half of the band (outer wall of the V)."""
    if len(band) < 4:
        return band
    dx = np.abs(band[:, 0] - float(apex[0]))
    thr = float(np.percentile(dx, 40))
    outer = band[dx >= thr]
    return outer if len(outer) >= 3 else band


def _opens_downward(left_band: np.ndarray, right_band: np.ndarray, apex: np.ndarray) -> bool:
    """
    True if flanks open like a V going down (left goes leftward, right rightward).

    Rejects Λ valley walls between serrations that converge downward.
    """
    ax = float(apex[0])

    def _side_ok(band: np.ndarray, want_left: bool) -> bool:
        if len(band) < 3:
            return False
        order = np.argsort(band[:, 1])
        n = max(1, len(band) // 4)
        near = band[order[:n]]
        deep = band[order[-n:]]
        near_x = float(np.mean(near[:, 0]))
        deep_x = float(np.mean(deep[:, 0]))
        if want_left:
            # Deeper should be further left (or at least not crossing inward a lot)
            return deep_x <= near_x + 0.75
        return deep_x >= near_x - 0.75

    return _side_ok(left_band, True) and _side_ok(right_band, False) and (
        float(np.mean(left_band[:, 0])) < ax and float(np.mean(right_band[:, 0])) > ax
    )


def _fit_chord_line(band: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Line through mean of upper vs lower thirds of the band (stable mid-flank chord)."""
    order = np.argsort(band[:, 1])
    sorted_pts = band[order]
    n = len(sorted_pts)
    k = max(2, n // 3)
    near = sorted_pts[:k].mean(axis=0)
    deep = sorted_pts[-k:].mean(axis=0)
    d = deep - near
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-9:
        return fit_line_tls(band)
    return near, d / nrm


def _line_r2(pts: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Coefficient of determination for TLS line fit (1 = perfect)."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if len(pts) < 2:
        return 0.0
    d = np.asarray(d, dtype=float).reshape(2)
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-12:
        return 0.0
    d = d / nrm
    n = np.array([-d[1], d[0]], dtype=float)
    resid = (pts - c.reshape(1, 2)) @ n
    ss_res = float(np.sum(resid ** 2))
    mean = pts.mean(axis=0)
    ss_tot = float(np.sum(np.sum((pts - mean) ** 2, axis=1)))
    if ss_tot < 1e-12:
        return 1.0
    return float(max(0.0, min(1.0, 1.0 - ss_res / ss_tot)))


def _line_through(c: np.ndarray, d: np.ndarray, p_deep: np.ndarray, p_proj: np.ndarray) -> list[float]:
    """Segment from deep flank toward / through the projected tip (PDF yellow V)."""
    v = p_proj - p_deep
    if float(np.linalg.norm(v)) < 1e-6:
        p0 = c - d * 40.0
        p1 = c + d * 40.0
        return [float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1])]
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
    y_cut = float(np.percentile(near[:, 1], 55))
    tip = near[near[:, 1] <= y_cut]
    if len(tip) < 3:
        tip = near
    tip = tip[np.argsort(tip[:, 0])]
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
    strategy: str = "tls",
) -> tuple[dict | None, str]:
    """
    Attempt a flank-line fit in one depth band.

    Strategies:
      tls          — total least squares on full band
      chord        — upper/lower-third chord (skips tip-rounding curvature)
      outer_tls    — TLS on lateral outer half of each flank
      outer_chord  — chord on outer half
    """
    left_band = _points_in_vertical_band(left, apex, y0_nm, y1_nm, nm_per_pixel)
    right_band = _points_in_vertical_band(right, apex, y0_nm, y1_nm, nm_per_pixel)

    if strategy.startswith("outer"):
        left_band = _outer_flank_points(left_band, apex, "left")
        right_band = _outer_flank_points(right_band, apex, "right")

    n_l, n_r = len(left_band), len(right_band)
    if n_l < min_flank_points and n_r < min_flank_points:
        return None, f"insufficient_flank_points(L={n_l},R={n_r},need={min_flank_points})"
    if n_l < min_flank_points:
        return None, f"insufficient_left_flank(L={n_l},need={min_flank_points})"
    if n_r < min_flank_points:
        return None, f"insufficient_right_flank(R={n_r},need={min_flank_points})"

    if not _opens_downward(left_band, right_band, apex):
        return None, "flanks_not_opening_downward"

    try:
        if strategy.endswith("chord") or strategy == "chord":
            c_l, d_l = _fit_chord_line(left_band)
            c_r, d_r = _fit_chord_line(right_band)
        else:
            c_l, d_l = fit_line_tls(left_band)
            c_r, d_r = fit_line_tls(right_band)
        projected = intersect_lines(c_l, d_l, c_r, d_r, min_cross=min_cross)
    except ValueError as exc:
        msg = str(exc).lower()
        if "parallel" in msg or "cross" in msg:
            return None, "near_parallel_flanks"
        return None, f"line_fit_or_intersect_failed({exc})"

    # Image-y: smaller = above. Blunt tips → P_proj above tip; sharp tips → P_proj ≈ tip.
    # Only reject when the intersection is clearly BELOW the ultimate tip (valley / Λ fit).
    dy_above = float(apex[1]) - float(projected[1])
    if dy_above < -0.5:
        return None, "projected_below_tip"
    if dy_above < 0.25:
        # Near-zero l (almost-sharp): park projected a hair above tip for a drawable l
        projected = np.array(
            [float(projected[0]), float(apex[1]) - 0.25],
            dtype=float,
        )

    # Projected x should sit between the flanks (near the tip x)
    ax = float(apex[0])
    xl, xr = float(np.mean(left_band[:, 0])), float(np.mean(right_band[:, 0]))
    if not (min(xl, xr) - 15.0 <= float(projected[0]) <= max(xl, xr) + 15.0):
        if abs(float(projected[0]) - ax) > 40.0:
            return None, "projected_x_off_axis"

    return {
        "left_band": left_band,
        "right_band": right_band,
        "c_l": c_l,
        "d_l": d_l,
        "c_r": c_r,
        "d_r": d_r,
        "projected": projected,
        "n_left": n_l,
        "n_right": n_r,
        "strategy": strategy,
    }, "ok"


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
    max_band_depth_nm: float | None = None,
    **legacy_kwargs,
) -> Method2Result:
    """
    PDF Method 2: yellow flank projection → convergent point → vertical red l → tip.

    max_band_depth_nm: optional cap (e.g. half distance to neighboring peak) so the
    fit does not reach into the next serration's Λ valley walls.
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

    if len(left) < 2 or len(right) < 2:
        result.rejection_reason = (
            f"insufficient_branch_geometry(L={len(left)},R={len(right)})"
        )
        return result

    depth_l = float(np.max(left[:, 1] - apex[1])) * nm_per_pixel if len(left) else 0.0
    depth_r = float(np.max(right[:, 1] - apex[1])) * nm_per_pixel if len(right) else 0.0
    avail_depth_nm = max(0.0, min(depth_l, depth_r))
    depth_cap = float(max_band_depth_nm) if max_band_depth_nm and max_band_depth_nm > 0 else None

    # Primary band first, then mid-flank fallbacks; clip to neighbor spacing if given
    bands: list[tuple[float, float]] = []

    def _add_band(b0: float, b1: float) -> None:
        if depth_cap is not None:
            if b0 >= depth_cap:
                return
            b1 = min(b1, depth_cap)
        if b1 <= b0 + 5.0:
            return
        pair = (float(b0), float(b1))
        if pair not in bands:
            bands.append(pair)

    _add_band(y0_nm, y1_nm)
    for b0, b1 in _DEFAULT_BANDS:
        _add_band(b0, b1)
    # Extra shallow mid-flank bands when a neighbor cap is active
    if depth_cap is not None:
        for frac in (0.35, 0.5, 0.65, 0.8):
            hi = depth_cap * frac
            lo = max(8.0, hi * 0.25)
            _add_band(lo, hi)

    min_pts_levels = sorted(
        {int(min_flank_points), max(3, int(min_flank_points) - 1), 3},
        reverse=True,
    )
    cross_levels = sorted({float(min_cross), 0.05, 0.03, 0.02})
    strategies = ("chord", "outer_chord", "tls", "outer_tls")

    chosen = None
    used_band = (y0_nm, y1_nm)
    used_min_pts = int(min_flank_points)
    attempts: list[dict] = []
    reason_counts: dict[str, int] = {}

    for b0, b1 in bands:
        if b1 > avail_depth_nm + 5.0 and avail_depth_nm > 0 and b0 > avail_depth_nm:
            reason = f"band_beyond_available_flank_depth(avail={avail_depth_nm:.1f}nm)"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if len(attempts) < 30:
                attempts.append({"band": [b0, b1], "reason": reason})
            continue
        for strategy in strategies:
            for min_pts in min_pts_levels:
                for mc in cross_levels:
                    trial, reason = _try_band(
                        apex,
                        left,
                        right,
                        nm_per_pixel,
                        b0,
                        b1,
                        min_pts,
                        mc,
                        strategy=strategy,
                    )
                    if trial is None:
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
                        if len(attempts) < 30:
                            attempts.append(
                                {
                                    "band": [b0, b1],
                                    "strategy": strategy,
                                    "min_pts": min_pts,
                                    "min_cross": mc,
                                    "reason": reason,
                                }
                            )
                        continue
                    cand_nm = abs(float(apex[1]) - float(trial["projected"][1])) * nm_per_pixel
                    if cand_nm > max_distance_nm:
                        reason = f"projected_distance_implausible({cand_nm:.1f}>{max_distance_nm})"
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
                        if len(attempts) < 30:
                            attempts.append(
                                {
                                    "band": [b0, b1],
                                    "strategy": strategy,
                                    "reason": reason,
                                }
                            )
                        continue
                    chosen = trial
                    used_band = (b0, b1)
                    used_min_pts = min_pts
                    break
                if chosen is not None:
                    break
            if chosen is not None:
                break
        if chosen is not None:
            break

    result.attempts = attempts
    if chosen is None:
        if reason_counts:
            top = max(reason_counts.items(), key=lambda kv: kv[1])[0]
            primary = top.split("(")[0]
            result.rejection_reason = primary
            if avail_depth_nm < float(y1_nm) * 0.5:
                result.rejection_reason = (
                    f"{primary};shallow_flank_depth(avail={avail_depth_nm:.1f}nm,"
                    f"need~{y1_nm:.0f}nm)"
                )
        else:
            result.rejection_reason = "insufficient_flank_or_no_intersection"
        return result

    left_band = chosen["left_band"]
    right_band = chosen["right_band"]
    c_l, d_l = chosen["c_l"], chosen["d_l"]
    c_r, d_r = chosen["c_r"], chosen["d_r"]
    projected = chosen["projected"]
    result.fit_band_nm = used_band
    result.fit_strategy = str(chosen.get("strategy") or "tls")
    result.n_left_band = int(chosen["n_left"])
    result.n_right_band = int(chosen["n_right"])
    result.line_r2_left = _line_r2(left_band, c_l, d_l)
    result.line_r2_right = _line_r2(right_band, c_r, d_r)

    distance_px = abs(float(apex[1]) - float(projected[1]))
    distance_nm = distance_px * nm_per_pixel
    if distance_nm > max_distance_nm:
        result.rejection_reason = "distance_implausible"
        return result
    if distance_px < 0.25:
        distance_px = 0.25
        distance_nm = distance_px * nm_per_pixel

    deep_l = left_band[int(np.argmax(left_band[:, 1]))]
    deep_r = right_band[int(np.argmax(right_band[:, 1]))]

    result.convergence_point = (float(projected[0]), float(projected[1]))
    result.distance_px = float(distance_px)
    result.distance_nm = float(distance_nm)
    result.left_line = _line_through(c_l, d_l, deep_l, projected)
    result.right_line = _line_through(c_r, d_r, deep_r, projected)

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

    if abs(float(d_l[0])) > 1e-9:
        result.left_edge_slope = float(d_l[1] / d_l[0])
    if abs(float(d_r[0])) > 1e-9:
        result.right_edge_slope = float(d_r[1] / d_r[0])

    if used_band != (y0_nm, y1_nm) or used_min_pts < int(min_flank_points):
        result.rejection_reason = None

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
        "line_r2_left": result.line_r2_left,
        "line_r2_right": result.line_r2_right,
        "n_left_band": result.n_left_band,
        "n_right_band": result.n_right_band,
        "fit_strategy": result.fit_strategy,
        "attempts": result.attempts[:8] if result.attempts else [],
    }
