"""Method 3 — included angle at fixed diameter D (Interpretation A).

Circle centred at the ultimate tip; intersect fitted edges; angle between
vectors from apex to first intersection on each branch.

Named: Included angle at D100 — not a radius.
Client must formally approve this definition (slide left Interpretation A vs B open).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sem_analysis.edge_geometry import included_angle_degrees


@dataclass
class InscribedAngleResult:
    tip_id: int
    tip_point: tuple[float, float]
    circle_center: tuple[float, float]
    circle_radius_px: float
    circle_diameter_px: float
    circle_diameter_nm: float
    intersection_left: tuple[float, float] | None
    intersection_right: tuple[float, float] | None
    angle_degrees: float | None
    angle_radians: float | None
    label: str = "angle_D100"
    left_tangent_line: list[float] = field(default_factory=list)
    right_tangent_line: list[float] = field(default_factory=list)
    valid: bool = False
    rejection_reason: str | None = None
    definition: str = "interpretation_A_circle_at_apex"
    method: str = "included_angle_at_D"
    tangent_from: str = "apex_to_intersections"


def _first_circle_intersection(
    branch: np.ndarray,
    apex: np.ndarray,
    radius: float,
) -> np.ndarray | None:
    """First intersection of branch polyline with circle centred at apex, away from apex."""
    pts = np.asarray(branch, dtype=float).reshape(-1, 2)
    apex = np.asarray(apex, dtype=float).reshape(2)
    # Order by distance from apex
    d = np.linalg.norm(pts - apex, axis=1)
    order = np.argsort(d)
    pts = pts[order]
    prev_inside = None
    for i in range(len(pts) - 1):
        p1, p2 = pts[i], pts[i + 1]
        d1 = np.linalg.norm(p1 - apex)
        d2 = np.linalg.norm(p2 - apex)
        # Crossing radius
        if (d1 - radius) * (d2 - radius) <= 0 and d1 != d2:
            t = (radius - d1) / (d2 - d1)
            t = float(np.clip(t, 0, 1))
            hit = p1 + t * (p2 - p1)
            if np.linalg.norm(hit - apex) > 1e-6:
                return hit
        prev_inside = d1 <= radius
    # Fallback: point nearest to radius
    if len(pts) == 0:
        return None
    idx = int(np.argmin(np.abs(np.linalg.norm(pts - apex, axis=1) - radius)))
    return pts[idx]


def measure_inscribed_angle(
    contour: np.ndarray | None = None,
    circle_diameter_nm: float | None = 100.0,
    nm_per_pixel: float = 1.0,
    circle_diameter_px: float | None = None,
    apex: np.ndarray | None = None,
    left: np.ndarray | None = None,
    right: np.ndarray | None = None,
    tip_id: int = 0,
    **legacy_kwargs,
) -> InscribedAngleResult | None:
    """Interpretation A: circle at apex, angle between apex→intersection vectors."""
    _ = legacy_kwargs
    if circle_diameter_nm is not None:
        d_nm = float(circle_diameter_nm)
        d_px = d_nm / max(nm_per_pixel, 1e-9)
    elif circle_diameter_px is not None:
        d_px = float(circle_diameter_px)
        d_nm = d_px * nm_per_pixel
    else:
        d_nm, d_px = 100.0, 100.0 / max(nm_per_pixel, 1e-9)

    label = f"angle_D{int(round(d_nm))}"

    if apex is None or left is None or right is None:
        if contour is None:
            return None
        pts = np.asarray(contour, dtype=float).reshape(-1, 2)
        apex_idx = int(np.argmin(pts[:, 1]))
        apex = pts[apex_idx]
        left = pts[pts[:, 0] < apex[0]]
        right = pts[pts[:, 0] > apex[0]]

    apex = np.asarray(apex, dtype=float).reshape(2)
    radius = d_px / 2.0
    result = InscribedAngleResult(
        tip_id=tip_id,
        tip_point=(float(apex[0]), float(apex[1])),
        circle_center=(float(apex[0]), float(apex[1])),
        circle_radius_px=radius,
        circle_diameter_px=d_px,
        circle_diameter_nm=d_nm,
        intersection_left=None,
        intersection_right=None,
        angle_degrees=None,
        angle_radians=None,
        label=label,
    )

    left_hit = _first_circle_intersection(left, apex, radius)
    right_hit = _first_circle_intersection(right, apex, radius)
    if left_hit is None or right_hit is None:
        result.rejection_reason = "missing_circle_intersection"
        return result

    # Ensure different branches (not same side)
    if (left_hit[0] - apex[0]) * (right_hit[0] - apex[0]) >= 0:
        # Try swap if needed
        if left_hit[0] > apex[0] and right_hit[0] < apex[0]:
            left_hit, right_hit = right_hit, left_hit
        elif left_hit[0] > apex[0] or right_hit[0] < apex[0]:
            result.rejection_reason = "intersections_same_branch"
            return result

    try:
        angle_deg = included_angle_degrees(apex, left_hit, right_hit)
    except ValueError:
        result.rejection_reason = "invalid_angle_vectors"
        return result

    if angle_deg < 1.0:
        result.rejection_reason = "angle_near_zero"
        return result

    result.intersection_left = (float(left_hit[0]), float(left_hit[1]))
    result.intersection_right = (float(right_hit[0]), float(right_hit[1]))
    result.angle_degrees = angle_deg
    result.angle_radians = float(np.radians(angle_deg))
    result.left_tangent_line = [
        float(apex[0]), float(apex[1]), float(left_hit[0]), float(left_hit[1])
    ]
    result.right_tangent_line = [
        float(apex[0]), float(apex[1]), float(right_hit[0]), float(right_hit[1])
    ]
    result.valid = True
    return result


def inscribed_angle_to_dict(result: InscribedAngleResult) -> dict:
    return {
        "tip_id": result.tip_id,
        "label": result.label,
        "definition": result.definition,
        "angle_degrees": result.angle_degrees,
        "angle_radians": result.angle_radians,
        "circle_diameter_nm": result.circle_diameter_nm,
        "circle_diameter_px": result.circle_diameter_px,
        "circle_radius_px": result.circle_radius_px,
        "tip_point": list(result.tip_point),
        "circle_center": list(result.circle_center),
        "intersection_left": list(result.intersection_left) if result.intersection_left else None,
        "intersection_right": list(result.intersection_right) if result.intersection_right else None,
        "left_tangent_line": result.left_tangent_line,
        "right_tangent_line": result.right_tangent_line,
        "valid": result.valid,
        "rejection_reason": result.rejection_reason,
        "tangent_from": result.tangent_from,
    }
