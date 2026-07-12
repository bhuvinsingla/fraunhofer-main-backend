"""Method 1 — fixed-distance inscribed circle → radius R.

Shared upstream: ultimate tip T = (x_t, y_t) and left/right edge contours.

Procedure:
  1. Identify ultimate tip T.
  2. At fixed vertical distance l below T, draw horizontal line y = y_t + l.
  3. Intersect with edge_L / edge_R → P_L, P_R.
  4. Circumcircle through T, P_L, P_R.
  5. Output = radius R of that circle.

l is a configurable constant (protocol.method1_primary_nm), held fixed across
samples. Small R = sharper tip; large R = more rounded / blunt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from sem_analysis.edge_geometry import (
    circumcircle_center,
    circumcircle_radius,
    stability_ratio,
)

log = logging.getLogger("sem-api.stages")


@dataclass
class Method1Result:
    tip_id: int
    label: str
    distance_l_nm: float
    distance_l_px: float
    radius_nm: float | None
    radius_px: float | None
    center: tuple[float, float] | None
    tip_point: tuple[float, float]
    intersection_left: tuple[float, float] | None
    intersection_right: tuple[float, float] | None
    scan_line: list[float] = field(default_factory=list)
    vertical_l_line: list[float] = field(default_factory=list)
    stability_s: float | None = None
    valid: bool = False
    rejection_reason: str | None = None
    # Projection measurement (tilt not blindly corrected)
    projected_radius_nm: float | None = None
    residual_px: float = 0.0


def _intersect_x_at_y(branch: np.ndarray, y: float) -> list[float]:
    """Find x intersections of polyline with horizontal line y."""
    pts = np.asarray(branch, dtype=float).reshape(-1, 2)
    xs = []
    for i in range(len(pts) - 1):
        y1, y2 = pts[i, 1], pts[i + 1, 1]
        if (y1 - y) * (y2 - y) <= 0 and y1 != y2:
            t = (y - y1) / (y2 - y1)
            xs.append(float(pts[i, 0] + t * (pts[i + 1, 0] - pts[i, 0])))
    return xs


def measure_method1_at_l(
    apex: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    nm_per_px: float,
    distance_l_nm: float,
    tip_id: int = 0,
) -> Method1Result:
    """
    Drop a fixed VERTICAL distance l below the apex, draw a HORIZONTAL chord,
    fit the circumcircle through apex + left/right edge crossings.

    All coordinates are image coordinates (y increases downward):
      apex        = highest point (top blue dot)
      y_scan      = apex_y + l_px
      left/right  = crossings of each branch with the horizontal line y_scan
    """
    label = f"R{int(round(distance_l_nm))}"
    apex_x, apex_y = float(apex[0]), float(apex[1])
    tip = (apex_x, apex_y)
    base = Method1Result(
        tip_id=tip_id,
        label=label,
        distance_l_nm=float(distance_l_nm),
        distance_l_px=float(distance_l_nm / max(nm_per_px, 1e-9)),
        radius_nm=None,
        radius_px=None,
        center=None,
        tip_point=tip,
        intersection_left=None,
        intersection_right=None,
        projected_radius_nm=None,
    )

    if nm_per_px <= 0 or distance_l_nm <= 0:
        base.rejection_reason = "invalid_scale"
        return base

    l_px = distance_l_nm / nm_per_px
    # Move straight down from the apex by the fixed vertical distance l
    y_scan = apex_y + l_px

    # Horizontal line intersections with each branch (image coords)
    left_xs = _intersect_x_at_y(left, y_scan)
    right_xs = _intersect_x_at_y(right, y_scan)
    log.info(
        "[STAGE 2/4 DETECT POINTS] tip_id=%s l=%.1fnm (%.2fpx) apex=(%.1f,%.1f) "
        "y_scan=%.2f left_xs=%s right_xs=%s n_left=%d n_right=%d",
        tip_id,
        distance_l_nm,
        l_px,
        apex_x,
        apex_y,
        y_scan,
        left_xs,
        right_xs,
        len(left),
        len(right),
    )

    # Crossings on the correct side of the apex
    left_cands = [x for x in left_xs if x < apex_x]
    right_cands = [x for x in right_xs if x > apex_x]
    if not left_cands or not right_cands:
        base.rejection_reason = "no_intersection"
        log.warning(
            "[STAGE 2/4 DETECT POINTS] tip_id=%s no_intersection at y_scan=%.2f "
            "(branch too short for this nm/px, or edges missing)",
            tip_id,
            y_scan,
        )
        return base

    # Pick the crossing on each side closest to the apex (inner blade edge),
    # robust to noisy multi-crossings farther out along the flank.
    x_l = max(left_cands)
    x_r = min(right_cands)
    if not (x_l < apex_x < x_r):
        base.rejection_reason = "intersections_same_side"
        return base

    p1 = np.array([apex_x, apex_y])
    p2 = np.array([x_l, y_scan])
    p3 = np.array([x_r, y_scan])

    try:
        r_px = circumcircle_radius(p1, p2, p3)
        cx, cy = circumcircle_center(p1, p2, p3)
    except ValueError:
        base.rejection_reason = "collinear_or_unstable"
        return base

    # Circle center should sit below the apex (between the branches)
    if cy < apex_y - 1e-3:
        base.rejection_reason = "circle_not_between_branches"
        return base

    r_nm = r_px * nm_per_px
    base.radius_px = float(r_px)
    base.radius_nm = float(r_nm)
    base.projected_radius_nm = float(r_nm)  # tilt not blindly corrected
    base.center = (float(cx), float(cy))
    base.intersection_left = (float(x_l), float(y_scan))
    base.intersection_right = (float(x_r), float(y_scan))
    base.scan_line = [float(x_l), float(y_scan), float(x_r), float(y_scan)]
    # Vertical red "l" straight down from apex to the scan-line height
    base.vertical_l_line = [apex_x, apex_y, apex_x, float(y_scan)]
    base.valid = True
    return base


def measure_method1_multi(
    apex: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    nm_per_px: float,
    distances_nm: list[float],
    tip_id: int = 0,
    stability_threshold: float = 0.20,
) -> dict[str, Method1Result]:
    """Measure R at each l; apply stability test around each primary l."""
    results: dict[str, Method1Result] = {}
    for l_nm in distances_nm:
        r = measure_method1_at_l(apex, left, right, nm_per_px, float(l_nm), tip_id)
        if r.valid:
            # Stability: R at 0.9l, l, 1.1l
            radii = []
            for factor in (0.9, 1.0, 1.1):
                rr = measure_method1_at_l(apex, left, right, nm_per_px, float(l_nm) * factor, tip_id)
                if rr.valid and rr.radius_nm is not None:
                    radii.append(rr.radius_nm)
            s = stability_ratio(radii)
            r.stability_s = s
            if s > stability_threshold:
                r.valid = False
                r.rejection_reason = "unstable_under_l_perturbation"
                r.radius_nm = None
                r.projected_radius_nm = None
        results[r.label] = r
    return results


def method1_to_dict(result: Method1Result) -> dict:
    return {
        "tip_id": result.tip_id,
        "label": result.label,
        "distance_l_nm": result.distance_l_nm,
        "distance_l_px": result.distance_l_px,
        "radius_nm": result.radius_nm,
        "projected_radius_nm": result.projected_radius_nm,
        "radius_px": result.radius_px,
        "center": list(result.center) if result.center else None,
        "tip_point": list(result.tip_point),
        "intersection_left": list(result.intersection_left) if result.intersection_left else None,
        "intersection_right": list(result.intersection_right) if result.intersection_right else None,
        "scan_line": result.scan_line,
        "vertical_l_line": result.vertical_l_line,
        "stability_s": result.stability_s,
        "valid": result.valid,
        "rejection_reason": result.rejection_reason,
    }


# --- Compatibility aliases for annotation / older tests ---

FixedDistanceCircleResult = Method1Result


def measure_fixed_distance_circle(
    image: np.ndarray,
    contour: np.ndarray,
    nm_per_pixel: float,
    distance_l_px: float | None = None,
    distance_l_nm: float | None = None,
    distances_nm: list[float] | None = None,
    fit_method: str = "taubin",
    primary_nm: float = 100.0,
) -> Method1Result | None:
    _ = image, fit_method
    pts = np.asarray(contour, dtype=float).reshape(-1, 2)
    apex = pts[int(np.argmin(pts[:, 1]))]
    left = pts[pts[:, 0] < apex[0]]
    right = pts[pts[:, 0] > apex[0]]
    if distance_l_nm is None and distance_l_px is not None:
        distance_l_nm = float(distance_l_px) * float(nm_per_pixel)
    if distances_nm:
        multi = measure_method1_multi(apex, left, right, nm_per_pixel, distances_nm)
        r = multi.get(f"R{int(round(primary_nm))}") or next((v for v in multi.values() if v.valid), None)
        return r if r and r.valid else None
    r = measure_method1_at_l(apex, left, right, nm_per_pixel, float(distance_l_nm or primary_nm))
    return r if r.valid else None


def measure_fixed_distance_at_l(
    contour: np.ndarray,
    nm_per_pixel: float,
    distance_l_nm: float,
    fit_method: str = "taubin",
) -> Method1Result | None:
    """Contour-based Method 1 at a single physical distance (test / API helper)."""
    return measure_fixed_distance_circle(
        np.zeros((8, 8), dtype=np.uint8),
        contour,
        nm_per_pixel,
        distance_l_nm=distance_l_nm,
        fit_method=fit_method,
    )


def measure_fixed_distance_multi(
    contour: np.ndarray,
    nm_per_pixel: float,
    distances_nm: list[float],
    fit_method: str = "taubin",
) -> list[Method1Result]:
    _ = fit_method
    pts = np.asarray(contour, dtype=float).reshape(-1, 2)
    apex = pts[int(np.argmin(pts[:, 1]))]
    left = pts[pts[:, 0] < apex[0]]
    right = pts[pts[:, 0] > apex[0]]
    multi = measure_method1_multi(apex, left, right, nm_per_pixel, distances_nm)
    return [r for r in multi.values() if r.valid]


def fixed_distance_circle_to_dict(result: Method1Result) -> dict:
    return method1_to_dict(result)


def fixed_distance_multi_to_dict(results: list[Method1Result]) -> dict:
    by_label = {r.label: method1_to_dict(r) for r in results}
    primary = next((r for r in results if r.label == "R100"), results[0] if results else None)
    out: dict = {"radii_by_l": by_label, "measurements": list(by_label.values())}
    if primary is not None:
        out.update(method1_to_dict(primary))
    return out

