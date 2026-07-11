"""Run all three brainstorming measurement methods (physical nm protocol)."""

from __future__ import annotations

import numpy as np

from sem_analysis.contour_features import (
    accept_tip,
    compute_contour_features,
    deduplicate_tips,
    group_parallel_ridges,
)
from sem_analysis.methods.fixed_distance_circle import (
    FixedDistanceCircleResult,
    fixed_distance_circle_to_dict,
    fixed_distance_multi_to_dict,
    measure_fixed_distance_circle,
    measure_fixed_distance_multi,
)
from sem_analysis.methods.inscribed_angle import (
    InscribedAngleResult,
    inscribed_angle_to_dict,
    measure_inscribed_angle,
)
from sem_analysis.methods.local_contour import extract_local_contour
from sem_analysis.methods.projected_tip_distance import (
    ProjectedTipDistanceResult,
    measure_projected_tip_distance,
    projected_tip_distance_to_dict,
)
from sem_analysis.stats_summary import (
    fit_score_from_residual,
    summarize_values,
    symmetry_from_branches,
    tip_confidence,
)


def _protocol(config: dict) -> dict:
    """Merge protocol block with measurement_methods (protocol wins)."""
    method_cfg = dict(config.get("measurement_methods", {}) or {})
    proto = dict(config.get("protocol", {}) or {})
    return {
        "approved": bool(proto.get("approved", False)),
        "distances_nm": list(
            proto.get("method1_distances_nm")
            or method_cfg.get("fixed_distance_circle", {}).get("distances_nm")
            or [25, 50, 100, 200]
        ),
        "primary_nm": float(
            proto.get("method1_primary_nm")
            or method_cfg.get("fixed_distance_circle", {}).get("primary_nm")
            or 100.0
        ),
        "fit_band_nm": list(
            proto.get("method2_fit_band_nm")
            or method_cfg.get("projected_tip_distance", {}).get("fit_band_nm")
            or [50, 200]
        ),
        "circle_diameter_nm": float(
            proto.get("method3_circle_diameter_nm")
            or method_cfg.get("inscribed_angle", {}).get("circle_diameter_nm")
            or 100.0
        ),
        "min_flank_points": int(
            method_cfg.get("projected_tip_distance", {}).get("min_flank_points", 5)
        ),
        "min_branch_depth_nm": float(method_cfg.get("min_branch_depth_nm", 50.0)),
        "dedup_distance_nm": float(method_cfg.get("dedup_distance_nm", 30.0)),
        "plausible_radius_nm": list(method_cfg.get("plausible_radius_nm") or [2, 500]),
        "parallel_sep_nm": float(method_cfg.get("parallel_sep_nm", 10.0)),
        "confidence_weights": method_cfg.get("confidence_weights")
        or {
            "edge": 0.25,
            "symmetry": 0.20,
            "fit": 0.25,
            "continuity": 0.15,
            "consensus": 0.15,
        },
        "require_outer": bool(method_cfg.get("require_outer_surface", False)),
    }


def run_brainstorming_per_peak(
    image: np.ndarray,
    local_contour: np.ndarray,
    peak_id: int,
    peak_location: tuple[float, float],
    nm_per_pixel: float,
    config: dict,
    *,
    features_dict: dict | None = None,
    accepted: bool = True,
) -> tuple[dict, dict]:
    """Run all three methods on one local serration contour."""
    proto = _protocol(config)
    method_cfg = config.get("measurement_methods", {})
    ia_cfg = method_cfg.get("inscribed_angle", {})
    raw: dict = {}
    serialized: dict = {
        "peak_id": peak_id,
        "peak_location": [float(peak_location[0]), float(peak_location[1])],
        "accepted": accepted,
    }
    if features_dict:
        serialized["contour_features"] = features_dict

    if not accepted:
        return serialized, raw

    multi = measure_fixed_distance_multi(
        local_contour,
        nm_per_pixel,
        proto["distances_nm"],
    )
    if multi:
        raw["fixed_distance_circle"] = multi
        serialized["fixed_distance_circle"] = fixed_distance_multi_to_dict(multi)
        # Plausible range check on primary
        rmin, rmax = proto["plausible_radius_nm"]
        primary = next(
            (r for r in multi if abs(r.distance_l_nm - proto["primary_nm"]) < 1e-6),
            multi[0],
        )
        if not (rmin <= primary.radius_nm <= rmax):
            serialized["accepted"] = False
            serialized["rejection_reason"] = "radius_out_of_range"
            return serialized, raw

    ptd = measure_projected_tip_distance(
        local_contour,
        nm_per_pixel,
        fit_band_nm=proto["fit_band_nm"],
        min_flank_points=proto["min_flank_points"],
    )
    if ptd:
        raw["projected_tip_distance"] = ptd
        serialized["projected_tip_distance"] = projected_tip_distance_to_dict(ptd)

    ia = measure_inscribed_angle(
        local_contour,
        circle_diameter_nm=proto["circle_diameter_nm"],
        nm_per_pixel=nm_per_pixel,
        tangent_from=ia_cfg.get("tangent_from", "intersections"),
        tangent_length=ia_cfg.get("tangent_length", 80.0),
    )
    if ia:
        raw["inscribed_angle"] = ia
        serialized["inscribed_angle"] = inscribed_angle_to_dict(ia)

    # Confidence from available scores
    residual = 0.0
    if multi:
        residual = float(np.mean([r.residual_px for r in multi]))
    sym = 0.5
    if features_dict:
        sym = symmetry_from_branches(
            float(features_dict.get("left_branch_depth_nm", 0)),
            float(features_dict.get("right_branch_depth_nm", 0)),
        )
    r2 = 0.5
    if ptd:
        r2 = 0.5 * (ptd.line_r2_left + ptd.line_r2_right)
    serialized["confidence"] = tip_confidence(
        edge_score=float((features_dict or {}).get("pointed_arch_score", 0.5)),
        symmetry_score=sym,
        fit_score=fit_score_from_residual(residual),
        continuity_score=float(min(1.0, (features_dict or {}).get("length_nm", 100) / 200.0)),
        consensus_score=0.5,
        weights=proto["confidence_weights"],
    )

    return serialized, raw


def run_brainstorming_all_peaks(
    image: np.ndarray,
    peaks: np.ndarray,
    edge_points: np.ndarray,
    nm_per_pixel: float,
    config: dict,
) -> tuple[dict, dict]:
    """Run all three brainstorming methods on every serration peak with tip gates."""
    method_cfg = config.get("measurement_methods", {})
    proto = _protocol(config)
    window_y = method_cfg.get("local_contour_window_y_px", 80.0)
    window_x = method_cfg.get("local_contour_window_x_px", 40.0)
    # Scale local window with physical depth when possible
    band = proto["fit_band_nm"]
    window_y_nm = float(band[1]) * 1.2
    window_y = max(window_y, window_y_nm / max(nm_per_pixel, 1e-9))

    # Extract local contours first
    locals_list: list[tuple[int, tuple[float, float], np.ndarray]] = []
    for peak_id, peak in enumerate(peaks):
        px, py = float(peak[0]), float(peak[1])
        local = extract_local_contour(
            edge_points,
            (px, py),
            window_y_px=window_y,
            window_x_px=window_x,
        )
        if local is not None:
            locals_list.append((peak_id, (px, py), local))

    # Group parallel ridges among local contours
    if locals_list:
        grouped = group_parallel_ridges(
            [c for _, _, c in locals_list],
            nm_per_pixel,
            parallel_sep_nm=proto["parallel_sep_nm"],
        )
        # Map representatives back (by tip proximity)
        rep_tips = []
        for g in grouped:
            from sem_analysis.contour_features import find_apex

            rep_tips.append(find_apex(g.reshape(-1, 2)))
        filtered: list[tuple[int, tuple[float, float], np.ndarray]] = []
        used = set()
        for peak_id, loc, contour in locals_list:
            tip = loc
            best_i = None
            best_d = 1e18
            for i, rt in enumerate(rep_tips):
                if i in used:
                    continue
                d = (tip[0] - rt[0]) ** 2 + (tip[1] - rt[1]) ** 2
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i is not None and best_d < (proto["parallel_sep_nm"] / nm_per_pixel) ** 2 * 4:
                used.add(best_i)
                filtered.append((peak_id, loc, grouped[best_i]))
            else:
                filtered.append((peak_id, loc, contour))
        locals_list = filtered

    all_tips = [loc for _, loc, _ in locals_list]
    features_list = []
    for peak_id, loc, contour in locals_list:
        others = [t for t in all_tips if t != loc]
        feat = compute_contour_features(
            contour,
            peak_id,
            nm_per_pixel,
            image.shape,
            config,
            other_tips=others,
            all_tips=all_tips,
        )
        accept_tip(feat, config, require_outer=proto["require_outer"])
        features_list.append(feat)

    features_list = deduplicate_tips(
        features_list, nm_per_pixel, proto["dedup_distance_nm"]
    )
    feat_by_id = {f.contour_id: f for f in features_list}

    n_detected = len(locals_list)
    per_peak_serialized: list[dict] = []
    per_peak_raw: list[dict] = []

    for peak_id, loc, contour in locals_list:
        feat = feat_by_id.get(peak_id)
        accepted = bool(feat.accepted) if feat else False
        serialized, raw = run_brainstorming_per_peak(
            image,
            contour,
            peak_id,
            loc,
            nm_per_pixel,
            config,
            features_dict=feat.to_dict() if feat else None,
            accepted=accepted,
        )
        if feat and not feat.accepted:
            serialized["rejection_reason"] = feat.rejection_reason
        per_peak_serialized.append(serialized)
        per_peak_raw.append(raw)

    n_accepted = sum(1 for e in per_peak_serialized if e.get("accepted"))

    def _build_method_summary(method_key: str, value_key: str) -> dict:
        curves = []
        values = []
        for entry in per_peak_serialized:
            if not entry.get("accepted"):
                continue
            method_data = entry.get(method_key)
            if not method_data:
                continue
            curves.append({
                "peak_id": entry["peak_id"],
                "peak_location": entry["peak_location"],
                "confidence": entry.get("confidence"),
                **method_data,
            })
            val = method_data.get(value_key)
            if val is not None:
                values.append(float(val))

        stats = summarize_values(values, headline="median")
        return {
            "count": len(curves),
            "n_detected_candidates": n_detected,
            "n_accepted": n_accepted,
            "median": stats["median"],
            "mean": stats["mean"],
            "std": stats["std"],
            "iqr": stats["iqr"],
            "min": stats["min"],
            "max": stats["max"],
            "ci95": stats["ci95"],
            "headline": "median",
            "headline_value": stats["headline_value"],
            # Backward-compatible mean keys
            f"mean_{value_key}" if not value_key.startswith("mean_") else value_key: stats["mean"],
            f"median_{value_key}": stats["median"],
            "per_curve": curves,
            "protocol": {
                "approved": proto["approved"],
                "distances_nm": proto["distances_nm"],
                "fit_band_nm": proto["fit_band_nm"],
                "circle_diameter_nm": proto["circle_diameter_nm"],
            },
        }

    # Method 1: also summarize each R{l} separately
    m1_curves = []
    m1_primary_values = []
    radii_by_l: dict[str, list[float]] = {}
    for entry in per_peak_serialized:
        if not entry.get("accepted"):
            continue
        method_data = entry.get("fixed_distance_circle")
        if not method_data:
            continue
        m1_curves.append({
            "peak_id": entry["peak_id"],
            "peak_location": entry["peak_location"],
            "confidence": entry.get("confidence"),
            **method_data,
        })
        if method_data.get("radius_nm") is not None:
            m1_primary_values.append(float(method_data["radius_nm"]))
        for label, rd in (method_data.get("radii_by_l") or {}).items():
            radii_by_l.setdefault(label, []).append(float(rd["radius_nm"]))

    m1_stats = summarize_values(m1_primary_values, headline="median")
    m1_summary = {
        "count": len(m1_curves),
        "n_detected_candidates": n_detected,
        "n_accepted": n_accepted,
        "median": m1_stats["median"],
        "mean": m1_stats["mean"],
        "std": m1_stats["std"],
        "iqr": m1_stats["iqr"],
        "min": m1_stats["min"],
        "max": m1_stats["max"],
        "ci95": m1_stats["ci95"],
        "headline": "median",
        "headline_value": m1_stats["headline_value"],
        "mean_radius_nm": m1_stats["mean"],
        "median_radius_nm": m1_stats["median"],
        "radii_by_l_summary": {
            label: summarize_values(vals, headline="median") for label, vals in radii_by_l.items()
        },
        "per_curve": m1_curves,
        "protocol": {
            "approved": proto["approved"],
            "distances_nm": proto["distances_nm"],
            "primary_nm": proto["primary_nm"],
        },
    }

    m2 = _build_method_summary("projected_tip_distance", "distance_l_nm")
    m2["mean_distance_l_nm"] = m2.get("mean")
    m2["median_distance_l_nm"] = m2.get("median")

    m3 = _build_method_summary("inscribed_angle", "angle_degrees")
    m3["mean_angle_deg"] = m3.get("mean")
    m3["median_angle_deg"] = m3.get("median")

    serialized_all = {
        "fixed_distance_circle": m1_summary,
        "projected_tip_distance": m2,
        "inscribed_angle": m3,
        "tip_validation": {
            "n_detected_candidates": n_detected,
            "n_accepted": n_accepted,
            "rejected": [
                {
                    "peak_id": e["peak_id"],
                    "reason": e.get("rejection_reason"),
                    "features": e.get("contour_features"),
                }
                for e in per_peak_serialized
                if not e.get("accepted")
            ],
        },
        "protocol": {
            "approved": proto["approved"],
            "method1_distances_nm": proto["distances_nm"],
            "method1_primary_nm": proto["primary_nm"],
            "method2_fit_band_nm": proto["fit_band_nm"],
            "method3_circle_diameter_nm": proto["circle_diameter_nm"],
        },
    }

    raw_all = {"per_peak": per_peak_raw, "features": [f.to_dict() for f in features_list]}
    return serialized_all, raw_all


def run_brainstorming_methods(
    image: np.ndarray,
    contour: np.ndarray,
    nm_per_pixel: float,
    config: dict,
) -> tuple[dict, dict]:
    """Compute all three brainstorming methods on primary contour."""
    proto = _protocol(config)
    method_cfg = config.get("measurement_methods", {})
    results: dict = {}
    serialized: dict = {}

    multi = measure_fixed_distance_multi(contour, nm_per_pixel, proto["distances_nm"])
    if multi:
        results["fixed_distance_circle"] = multi
        serialized["fixed_distance_circle"] = fixed_distance_multi_to_dict(multi)
    else:
        fdc = measure_fixed_distance_circle(
            image, contour, nm_per_pixel, distances_nm=proto["distances_nm"], primary_nm=proto["primary_nm"]
        )
        if fdc:
            results["fixed_distance_circle"] = fdc
            serialized["fixed_distance_circle"] = fixed_distance_circle_to_dict(fdc)

    ptd = measure_projected_tip_distance(
        contour,
        nm_per_pixel,
        fit_band_nm=proto["fit_band_nm"],
        min_flank_points=proto["min_flank_points"],
    )
    if ptd:
        results["projected_tip_distance"] = ptd
        serialized["projected_tip_distance"] = projected_tip_distance_to_dict(ptd)

    ia_cfg = method_cfg.get("inscribed_angle", {})
    ia = measure_inscribed_angle(
        contour,
        circle_diameter_nm=proto["circle_diameter_nm"],
        nm_per_pixel=nm_per_pixel,
        tangent_from=ia_cfg.get("tangent_from", "intersections"),
        tangent_length=ia_cfg.get("tangent_length", 80.0),
    )
    if ia:
        results["inscribed_angle"] = ia
        serialized["inscribed_angle"] = inscribed_angle_to_dict(ia)

    return serialized, results
