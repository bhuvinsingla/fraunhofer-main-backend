"""Blade Value table — per-tip metrics + averages (brainstorming / transcript).

Maps Shashwat's summary table onto hard-valid arch tips:
  α  = included angle between TLS flank lines
  D  = 2 × Method-1 primary radius (inscribed diameter at primary l)
  l  = Method-2 projected tip distance
  A  = area between fitted flanks and actual edge (nm²)
"""

from __future__ import annotations

import numpy as np

from sem_analysis.stats_summary import summarize_values


def flank_included_angle_deg(direction_left: np.ndarray, direction_right: np.ndarray) -> float:
    """Included angle α between two flank direction vectors (degrees)."""
    v1 = np.asarray(direction_left, dtype=float).reshape(2)
    v2 = np.asarray(direction_right, dtype=float).reshape(2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        raise ValueError("Degenerate flank direction.")
    v1 = v1 / n1
    v2 = v2 / n2
    # Orient both "down" the tip (positive Y in image coords) so α is the tip opening
    if v1[1] < 0:
        v1 = -v1
    if v2[1] < 0:
        v2 = -v2
    cosine = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def area_between_line_and_edge_nm2(
    points: np.ndarray,
    centroid: np.ndarray,
    direction: np.ndarray,
    nm_per_px: float,
) -> float:
    """
    Trapezoidal integral of |perp distance to fitted line| along the edge.
    Returns area in nm² (wear / deviation from ideal flank).
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) < 2:
        return 0.0
    c = np.asarray(centroid, dtype=float).reshape(2)
    d = np.asarray(direction, dtype=float).reshape(2)
    dn = np.linalg.norm(d)
    if dn < 1e-9:
        return 0.0
    d = d / dn
    normal = np.array([-d[1], d[0]])
    dist = (pts - c) @ normal
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    abs_d = np.abs(dist)
    area_px2 = float(np.sum(0.5 * (abs_d[:-1] + abs_d[1:]) * seg))
    return area_px2 * (float(nm_per_px) ** 2)


def build_blade_value_table(tips: list) -> dict:
    """
    Build per-tip Blade Value rows and averages from TipMeasurement list.

    Only hard-valid tips with at least one successful metric contribute to averages.
    """
    rows: list[dict] = []
    for t in tips:
        if not getattr(t, "hard_valid", False):
            continue
        m1 = t.method1 or {}
        m2 = t.method2 or {}
        r_nm = m1.get("projected_radius_nm") or m1.get("radius_nm")
        diameter_nm = (2.0 * float(r_nm)) if r_nm is not None else None
        row = {
            "tip_id": t.tip_id,
            "apex_x_px": t.apex_x_px,
            "apex_y_px": t.apex_y_px,
            "included_angle_deg": m2.get("included_angle_deg"),
            "inscribed_diameter_nm": diameter_nm,
            "inscribed_radius_nm": float(r_nm) if r_nm is not None else None,
            "distance_to_tip_nm": m2.get("distance_l_nm"),
            "area_under_curve_nm2": m2.get("area_under_curve_nm2"),
            "angle_D100_deg": (t.method3 or {}).get("angle_degrees"),
            "confidence": t.confidence,
            "method1_valid": t.method1_valid,
            "method2_valid": t.method2_valid,
            "method3_valid": t.method3_valid,
        }
        rows.append(row)

    def _avg(key: str) -> dict:
        return summarize_values(
            [r[key] for r in rows if r.get(key) is not None],
            headline="mean",  # transcript: Blade Value = average of tips
        )

    blade = {
        "included_angle_deg": _avg("included_angle_deg"),
        "inscribed_diameter_nm": _avg("inscribed_diameter_nm"),
        "inscribed_radius_nm": _avg("inscribed_radius_nm"),
        "distance_to_tip_nm": _avg("distance_to_tip_nm"),
        "area_under_curve_nm2": _avg("area_under_curve_nm2"),
        "angle_D100_deg": _avg("angle_D100_deg"),
    }

    return {
        "n_tips": len(rows),
        "per_tip": rows,
        "blade_value": {
            "included_angle_deg": blade["included_angle_deg"].get("mean"),
            "inscribed_diameter_nm": blade["inscribed_diameter_nm"].get("mean"),
            "inscribed_radius_nm": blade["inscribed_radius_nm"].get("mean"),
            "distance_to_tip_nm": blade["distance_to_tip_nm"].get("mean"),
            "area_under_curve_nm2": blade["area_under_curve_nm2"].get("mean"),
            "angle_D100_deg": blade["angle_D100_deg"].get("mean"),
        },
        "blade_value_stats": blade,
        "note": (
            "Blade Value = mean of hard-valid tips (transcript). "
            "α from TLS flank lines; D = 2·R at primary l; l = projected tip distance; "
            "A = flank deviation area (nm²). Method 3 θ100 kept separately."
        ),
    }
