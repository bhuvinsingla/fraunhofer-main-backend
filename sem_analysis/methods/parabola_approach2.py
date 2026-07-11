"""Approach 2 — detect ridge curve segments that are parabolas (vertex form).

Vertex form (see Math.SE / SO):
  y = a (x − h)² + k
  https://math.stackexchange.com/questions/1631819/vertex-form-of-parabola-why-does-it-work
  https://stackoverflow.com/questions/15587311/compute-parabola-using-python

Sliding windows along the skyline are fit to this form. Good fits are kept.

Visualization:
  - fitted parabola curve → purple
  - vertex (h, k) → pink

Osculating radius at the vertex: R = 1 / (2 |a|)  (pixels).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from sem_analysis.edge_detection import _build_edge_map, _extract_skyline_y
from sem_analysis.stats_summary import summarize_values

log = logging.getLogger("sem-api.stages")


def parabola_y(x, h: float, k: float, a: float) -> np.ndarray:
    """Vertex-form parabola: y = a (x − h)² + k."""
    x = np.asarray(x, dtype=np.float64)
    return a * (x - h) ** 2 + k


def _skyline_polyline(edge_map: np.ndarray) -> np.ndarray:
    skyline = _extract_skyline_y(edge_map)
    w = edge_map.shape[1]
    xs = np.arange(w, dtype=np.float64)
    valid = ~np.isnan(skyline)
    if int(valid.sum()) < 10:
        return np.empty((0, 2), dtype=np.float64)
    filled = skyline.copy()
    filled[~valid] = np.interp(xs[~valid], xs[valid], skyline[valid])
    return np.column_stack([xs, filled])


def fit_vertex_parabola(pts: np.ndarray) -> dict[str, Any] | None:
    """
    Fit y = a(x−h)² + k to points.

    Seed (h,k) at the topmost point (tip in image coords), solve for a,
    then optionally refine a,h,k with least_squares.
    """
    if len(pts) < 8:
        return None
    pts = np.asarray(pts, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]

    # Tip / vertex seed: smallest y (upward tip on SEM)
    i0 = int(np.argmin(y))
    h0 = float(x[i0])
    k0 = float(y[i0])

    dx2 = (x - h0) ** 2
    # Avoid singular: need some spread
    if float(np.max(dx2)) < 1e-6:
        return None
    # Least-squares a for fixed (h,k): y - k = a * (x-h)^2
    a0 = float(np.sum(dx2 * (y - k0)) / np.sum(dx2 * dx2))
    if not np.isfinite(a0) or abs(a0) < 1e-12:
        return None
    # Tip asperities open downward in image → a should be positive
    if a0 <= 0:
        return None

    def residuals(params):
        a, h, k = params
        return parabola_y(x, h, k, a) - y

    try:
        sol = least_squares(residuals, [a0, h0, k0], method="lm", max_nfev=80)
        a, h, k = (float(v) for v in sol.x)
    except Exception:
        a, h, k = a0, h0, k0

    if not np.isfinite(a) or a <= 0:
        return None
    if h < float(x.min()) - 2 or h > float(x.max()) + 2:
        return None

    y_hat = parabola_y(x, h, k, a)
    residual = float(np.sqrt(np.mean((y_hat - y) ** 2)))
    y_span = float(np.ptp(y)) + 1e-6
    rel_residual = residual / y_span

    # Osculating radius at vertex for y = a x^2 is 1/(2|a|)
    radius_px = 1.0 / (2.0 * abs(a))

    # Sample fitted curve for drawing
    xs_fit = np.linspace(float(x.min()), float(x.max()), max(20, len(pts)))
    ys_fit = parabola_y(xs_fit, h, k, a)
    curve_pts = np.column_stack([xs_fit, ys_fit])

    return {
        "a": a,
        "h": h,
        "k": k,
        "vertex": [h, k],
        "radius_px": float(radius_px),
        "residual_px": residual,
        "rel_residual": float(rel_residual),
        "curve_points": curve_pts.tolist(),
        "data_points": pts.tolist(),
    }


def _window_is_parabola(
    pts: np.ndarray,
    *,
    max_rel_residual: float,
    min_radius_px: float,
    max_radius_px: float,
    min_a: float,
) -> dict[str, Any] | None:
    hit = fit_vertex_parabola(pts)
    if hit is None:
        return None
    if hit["rel_residual"] > max_rel_residual:
        return None
    r = hit["radius_px"]
    if r < min_radius_px or r > max_radius_px:
        return None
    if hit["a"] < min_a:
        return None
    return hit


def find_parabola_curves(
    image: np.ndarray,
    nm_per_pixel: float,
    config: dict | None = None,
) -> dict[str, Any]:
    """
    Approach 2: find skyline segments that match a vertex-form parabola.

    Returns summary compatible with Approach 1 UI (per_curve, median_radius_nm).
    """
    cfg = (config or {}).get("parabola_approach", {}) if config else {}
    # Fall back to old circular_arc keys if present
    if not cfg and config:
        cfg = (config.get("circular_arc_approach") or {})

    window = int(cfg.get("window_px", 40))
    step = int(cfg.get("step_px", 8))
    max_rel = float(cfg.get("max_rel_residual", 0.08))
    min_r = float(cfg.get("min_radius_px", 5.0))
    max_r = float(cfg.get("max_radius_px", 150.0))
    min_a = float(cfg.get("min_curvature_a", 1e-5))
    dedup_dist = float(cfg.get("dedup_vertex_px", 12.0))

    edge_map = _build_edge_map(image, config or {})
    poly = _skyline_polyline(edge_map)
    if len(poly) < window:
        log.info("[APPROACH 2] insufficient skyline for parabola fit (%d)", len(poly))
        return {
            "approach": "vertex_parabola",
            "label": "Approach 2 — Vertex-form parabola curves",
            "count": 0,
            "per_curve": [],
            "median_radius_nm": None,
            "mean_radius_nm": None,
            "n_windows_tested": 0,
            "reference": [
                "https://stackoverflow.com/questions/15587311/compute-parabola-using-python",
                "https://math.stackexchange.com/questions/1631819/vertex-form-of-parabola-why-does-it-work",
            ],
        }

    candidates: list[dict[str, Any]] = []
    n_tested = 0
    for start in range(0, len(poly) - window + 1, max(1, step)):
        pts = poly[start : start + window]
        n_tested += 1
        hit = _window_is_parabola(
            pts,
            max_rel_residual=max_rel,
            min_radius_px=min_r,
            max_radius_px=max_r,
            min_a=min_a,
        )
        if hit is None:
            continue
        hit["window_start"] = int(start)
        hit["window_end"] = int(start + window - 1)
        candidates.append(hit)

    candidates.sort(key=lambda c: c["rel_residual"])
    kept: list[dict[str, Any]] = []
    for c in candidates:
        hx, hy = c["vertex"]
        if any((hx - k["vertex"][0]) ** 2 + (hy - k["vertex"][1]) ** 2 < dedup_dist**2 for k in kept):
            continue
        kept.append(c)

    per_curve: list[dict[str, Any]] = []
    for i, c in enumerate(kept):
        r_nm = float(c["radius_px"]) * float(nm_per_pixel)
        per_curve.append(
            {
                "peak_id": i,
                "curve_id": i,
                "peak_location": list(c["vertex"]),
                "tip_point": list(c["vertex"]),
                "vertex": list(c["vertex"]),
                "center": list(c["vertex"]),  # UI/CSV reuse center columns as vertex
                "a": c["a"],
                "h": c["h"],
                "k": c["k"],
                "radius_px": c["radius_px"],
                "radius_nm": r_nm,
                "residual_px": c["residual_px"],
                "rel_residual": c["rel_residual"],
                "curve_points": c["curve_points"],
                "arc_points": c["curve_points"],  # annotation reuse
                "valid": True,
                "approach": "vertex_parabola",
                "equation": f"y = {c['a']:.6g}(x - {c['h']:.3f})^2 + {c['k']:.3f}",
            }
        )

    radii = [c["radius_nm"] for c in per_curve]
    stats = summarize_values(radii)
    summary = {
        "approach": "vertex_parabola",
        "label": "Approach 2 — Vertex-form parabola (purple) + pink vertex",
        "headline": "median",
        **stats,
        "median_radius_nm": stats.get("median"),
        "mean_radius_nm": stats.get("mean"),
        "count": len(per_curve),
        "per_curve": per_curve,
        "failed_curves": [],
        "n_windows_tested": n_tested,
        "n_candidates_before_dedup": len(candidates),
        "params": {
            "window_px": window,
            "step_px": step,
            "max_rel_residual": max_rel,
            "min_radius_px": min_r,
            "max_radius_px": max_r,
        },
        "nm_per_pixel": nm_per_pixel,
        "reference": [
            "https://stackoverflow.com/questions/15587311/compute-parabola-using-python",
            "https://math.stackexchange.com/questions/1631819/vertex-form-of-parabola-why-does-it-work",
        ],
    }
    log.info(
        "[APPROACH 2] parabolas=%d (windows=%d) median_R_vertex=%.3f nm",
        len(per_curve),
        n_tested,
        stats["median"] if stats.get("median") is not None else float("nan"),
    )
    return summary


# Back-compat alias so old imports keep working during transition
find_circular_arc_curves = find_parabola_curves
