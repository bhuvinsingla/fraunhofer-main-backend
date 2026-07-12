"""Unified tip measurement: same tip IDs across Methods 1–3; hard validity first."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sem_analysis.arch_detection import ValidatedArch, detect_validated_arches
from sem_analysis.blade_value import build_blade_value_table
from sem_analysis.edge_detection import detect_serration_peaks_global
from sem_analysis.edge_probability import preprocess_sem
from sem_analysis.methods.fixed_distance_circle import (
    measure_method1_at_l,
    measure_method1_multi,
    method1_to_dict,
)
from sem_analysis.methods.inscribed_angle import (
    inscribed_angle_to_dict,
    measure_inscribed_angle,
)
from sem_analysis.methods.projected_tip_distance import (
    measure_projected_tip_distance,
    projected_tip_distance_to_dict,
)
from sem_analysis.parabola_apex import detect_parabola_apexes
from sem_analysis.multi_scale import (
    filter_peaks_by_multi_scale_y,
    refine_ms_peaks_with_cv,
    run_multi_scale,
)
from sem_analysis.protocol import get_protocol
from sem_analysis.roi import MeasurementROI, extract_measurement_roi
from sem_analysis.stats_summary import summarize_values, tip_confidence
from sem_analysis.whiteboard_geometry import build_whiteboard_geometry, whiteboard_to_dict

log = logging.getLogger("sem-api.stages")

def _shift_xy(pt, ox: float, oy: float):
    if pt is None:
        return None
    return [float(pt[0]) + ox, float(pt[1]) + oy]


def _shift_line(line, ox: float, oy: float):
    if not line or len(line) < 4:
        return line
    return [
        float(line[0]) + ox, float(line[1]) + oy,
        float(line[2]) + ox, float(line[3]) + oy,
    ]


def _shift_poly(poly, ox: float, oy: float):
    if not poly:
        return poly
    return [[float(p[0]) + ox, float(p[1]) + oy] for p in poly]


@dataclass
class TipMeasurement:
    tip_id: int
    apex_x_px: float
    apex_y_px: float
    nm_per_px: float
    border_valid: bool
    left_branch_valid: bool
    right_branch_valid: bool
    fit_residual_px: float
    window_valid: bool
    hard_valid: bool
    method1: dict = field(default_factory=dict)
    method2: dict = field(default_factory=dict)
    method3: dict = field(default_factory=dict)
    whiteboard: dict = field(default_factory=dict)
    method1_valid: bool = False
    method2_valid: bool = False
    method3_valid: bool = False
    confidence: float = 0.0
    rejection_reason: str | None = None

    def to_row(self) -> dict:
        row = {
            "tip_id": self.tip_id,
            "apex_x_px": self.apex_x_px,
            "apex_y_px": self.apex_y_px,
            "nm_per_px": self.nm_per_px,
            "border_valid": self.border_valid,
            "left_branch_valid": self.left_branch_valid,
            "right_branch_valid": self.right_branch_valid,
            "fit_residual_px": self.fit_residual_px,
            "window_valid": self.window_valid,
            "hard_valid": self.hard_valid,
            "method1_valid": self.method1_valid,
            "method2_valid": self.method2_valid,
            "method3_valid": self.method3_valid,
            "confidence": self.confidence if self.hard_valid else 0.0,
            "rejection_reason": self.rejection_reason,
            "projected_tip_distance_nm": self.method2.get("distance_l_nm"),
            "included_angle_deg": self.method2.get("included_angle_deg"),
            "area_under_curve_nm2": self.method2.get("area_under_curve_nm2"),
            "angle_D100_deg": self.method3.get("angle_degrees"),
        }
        for label, data in (self.method1.get("radii_by_l") or {}).items():
            row[f"{label}_nm"] = data.get("projected_radius_nm") or data.get("radius_nm")
        # Ensure R columns exist
        for lab in ("R25", "R50", "R100", "R200"):
            row.setdefault(f"{lab}_nm", None)
        return row


def _hard_valid(arch: ValidatedArch) -> tuple[bool, str | None]:
    if not arch.window_valid:
        return False, arch.rejection_reason or "incomplete_measurement_window"
    if not arch.border_valid:
        return False, "touches_border"
    if not arch.left_branch_valid or not arch.right_branch_valid:
        return False, "missing_branch"
    if not arch.valid:
        return False, arch.rejection_reason or "arch_invalid"
    if arch.fit_residual_px > 2.0:
        return False, "fit_residual_too_high"
    return True, None


def _branches_from_edge_points(
    edge_points: np.ndarray,
    apex: np.ndarray,
    window_y_px: float,
    window_x_px: float,
    min_pts: int = 5,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Local left/right flanks around an asperity apex for Method 1."""
    px, py = float(apex[0]), float(apex[1])
    pts = np.asarray(edge_points, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return None
    mask = (
        (np.abs(pts[:, 0] - px) <= window_x_px)
        & (pts[:, 1] >= py - window_y_px * 0.2)
        & (pts[:, 1] <= py + window_y_px)
    )
    local = pts[mask]
    left = local[local[:, 0] <= px]
    right = local[local[:, 0] > px]
    if len(left) < min_pts or len(right) < min_pts:
        return None
    left = left[np.argsort(left[:, 1])]
    right = right[np.argsort(right[:, 1])]
    return left, right


def _apply_method1_to_tip(
    tm: TipMeasurement,
    apex: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    nm_per_px: float,
    proto: dict,
    tip_id: int,
    *,
    skip_stability: bool = False,
    stability_threshold: float = 0.20,
) -> tuple[object | None, dict, dict, dict]:
    """
    Run Method 1 and fill tm.method1.

    Returns (primary_result, point_rec, mark_rec, rad_rec) — records may be empty dicts.
    """
    distances = list(proto.get("method1_distances_nm") or [proto.get("method1_primary_nm", 100)])
    primary_nm = float(proto.get("method1_primary_nm", 100))
    primary_label = f"R{int(round(primary_nm))}"

    if skip_stability:
        primary = measure_method1_at_l(apex, left, right, nm_per_px, primary_nm, tip_id=tip_id)
        m1 = {primary.label: primary}
    else:
        m1 = measure_method1_multi(
            apex, left, right, nm_per_px, distances, tip_id=tip_id,
            stability_threshold=stability_threshold,
        )
        primary = m1.get(primary_label) or next(iter(m1.values()), None)
        # If stability rejected but geometry exists at primary l, keep drawable circle
        if primary is not None and not primary.valid and primary.rejection_reason == "unstable_under_l_perturbation":
            soft = measure_method1_at_l(apex, left, right, nm_per_px, primary_nm, tip_id=tip_id)
            if soft.valid:
                soft.rejection_reason = None
                soft.stability_s = primary.stability_s
                primary = soft
                m1[primary_label] = soft

    radii_by_l = {k: method1_to_dict(v) for k, v in m1.items()}
    tm.method1 = {
        "radii_by_l": radii_by_l,
        **(method1_to_dict(primary) if primary else {}),
    }
    tm.method1_valid = bool(primary and primary.valid)
    if primary and not primary.valid:
        tm.method1["method1_rejection_reason"] = primary.rejection_reason
        if not tm.rejection_reason:
            tm.rejection_reason = primary.rejection_reason
        if not tm.method1.get("tip_point"):
            tm.method1["tip_point"] = [float(apex[0]), float(apex[1])]
        tm.method1["valid"] = False

    point_rec: dict = {}
    mark_rec: dict = {}
    rad_rec: dict = {}
    if primary:
        tip_pt = primary.tip_point
        left_pt = primary.intersection_left
        right_pt = primary.intersection_right
        point_rec = {
            "tip_id": tip_id,
            "l_nm": primary.distance_l_nm,
            "l_px": primary.distance_l_px,
            "apex": list(tip_pt) if tip_pt else None,
            "left": list(left_pt) if left_pt else None,
            "right": list(right_pt) if right_pt else None,
            "scan_line": primary.scan_line,
            "vertical_l_line": primary.vertical_l_line,
            "found_all_three": bool(tip_pt and left_pt and right_pt),
            "rejection_reason": primary.rejection_reason,
        }
        mark_rec = {
            "tip_id": tip_id,
            "will_draw_apex": tip_pt is not None,
            "will_draw_left": left_pt is not None,
            "will_draw_right": right_pt is not None,
            "will_draw_scan_line": bool(primary.scan_line),
            "will_draw_vertical_l": bool(primary.vertical_l_line),
            "will_draw_circle": bool(primary.valid and primary.center and primary.radius_px),
        }
        rad_rec = {
            "tip_id": tip_id,
            "valid": primary.valid,
            "radius_nm": primary.radius_nm,
            "radius_px": primary.radius_px,
            "center": list(primary.center) if primary.center else None,
            "rejection_reason": primary.rejection_reason,
        }
    return primary, point_rec, mark_rec, rad_rec


def _build_diagnostics(
    tips: list[TipMeasurement],
    proto: dict,
    nm_per_px: float,
) -> dict:
    """Peak spacing vs protocol constants + Method 2 coverage notes."""
    ys = sorted(float(t.apex_y_px) for t in tips)
    spacings_nm: list[float] = []
    if len(ys) >= 2 and nm_per_px > 0:
        spacings_nm = [abs(ys[i + 1] - ys[i]) * nm_per_px for i in range(len(ys) - 1)]

    l1 = float(proto.get("method1_primary_nm", 50))
    d3 = float(proto.get("method3_circle_diameter_nm", 100))
    med_spacing = float(np.median(spacings_nm)) if spacings_nm else None

    warnings: list[str] = []
    if not proto.get("approved", False):
        warnings.append(
            f"Protocol constants not client-approved (l={l1:g} nm, D={d3:g} nm)."
        )
    if med_spacing is not None:
        if l1 >= 0.8 * med_spacing:
            warnings.append(
                f"Method 1 offset l={l1:g} nm is >=80% of median peak spacing "
                f"({med_spacing:.1f} nm) — chord may reach a neighboring peak."
            )
        if d3 >= 0.8 * med_spacing:
            warnings.append(
                f"Method 3 diameter D={d3:g} nm is >=80% of median peak spacing "
                f"({med_spacing:.1f} nm) — circle may intersect a neighbor."
            )

    m2_ok = sum(1 for t in tips if t.method2_valid)
    m2_fail = [
        {
            "peak_id": t.tip_id,
            "reason": (t.method2 or {}).get("rejection_reason") or "unknown",
            "y_px": t.apex_y_px,
        }
        for t in tips
        if not t.method2_valid
    ]

    return {
        "n_tips": len(tips),
        "peak_spacing_nm": {
            "median": med_spacing,
            "min": float(min(spacings_nm)) if spacings_nm else None,
            "max": float(max(spacings_nm)) if spacings_nm else None,
            "values": [round(s, 2) for s in spacings_nm],
        },
        "protocol_constants": {
            "method1_primary_nm": l1,
            "method3_circle_diameter_nm": d3,
            "approved": bool(proto.get("approved", False)),
        },
        "warnings": warnings,
        "method2_coverage": {
            "n_ok": m2_ok,
            "n_failed": len(m2_fail),
            "failed": m2_fail,
        },
        "multi_peak": len(tips) > 1,
        "reading_note": (
            "Serrated / multi-peak edge: report median R with IQR rather than a "
            "single tip radius."
            if len(tips) > 1
            else "Single tip detected."
        ),
    }


def measure_all_tips(
    roi: MeasurementROI,
    nm_per_px: float,
    config: dict,
    image_id: str = "",
) -> tuple[list[TipMeasurement], dict]:
    """
    Approach 1: find ALL asperity peaks (1D ridge projection), then Method 1 R100
    on every tip where left/right flanks allow a circle. Mark all findings.
    """
    proto = get_protocol(config)
    primary_nm = float(proto["method1_primary_nm"])
    l_px = primary_nm / max(nm_per_px, 1e-9)
    tip_cfg = config.get("tip_detection", {})
    mode = str(tip_cfg.get("mode", "ridge")).lower()  # ridge | arch
    method_cfg = config.get("measurement_methods", {})
    window_y = float(method_cfg.get("local_contour_window_y_px", 80.0))
    window_x = float(method_cfg.get("local_contour_window_x_px", 40.0))
    window_y = max(window_y, primary_nm * 1.5 / max(nm_per_px, 1e-9))
    min_branch = int(config.get("tip_validation", {}).get("min_branch_points", 5))
    min_branch = max(5, min(min_branch, 8))
    stab_thr = float(config.get("tip_validation", {}).get("method1_stability_threshold", 0.20))
    skip_stab = bool(tip_cfg.get("ridge_skip_stability", False))

    log.info("=" * 60)
    log.info(
        "[STAGE 2/4 DETECT POINTS] image=%s mode=%s nm/px=%.6f l=%s nm (%.2f px)",
        image_id, mode, nm_per_px, primary_nm, l_px,
    )

    ox, oy = roi.offset_x, roi.offset_y
    stage2_points: list[dict] = []
    stage3_marks: list[dict] = []
    stage4_radii: list[dict] = []
    tips: list[TipMeasurement] = []
    n_candidates = 0
    multi_scale_meta: dict = {"enabled": False}

    if mode != "arch":
        # —— Approach 1 core: 1D ridge asperity peaks (all tips along curve) ——
        edge = detect_serration_peaks_global(roi.image, config)
        peaks = edge.peak_locations
        edge_points = edge.edge_points

        # —— Parabola-apex mode: measure the topmost vertex of each central arch ——
        # Highest priority: when enabled it replaces the serration peaks with the
        # apexes of the nested ∧-shaped arches on the central spine.
        pa_cfg = config.get("parabola_apex", {}) or {}
        used_parabola_apex = False
        if bool(pa_cfg.get("enabled", False)):
            try:
                apexes = detect_parabola_apexes(roi.image, config)
                if len(apexes) >= int(pa_cfg.get("min_apexes", 2)):
                    peaks = [np.asarray(p, dtype=float) for p in apexes]
                    used_parabola_apex = True
                    multi_scale_meta = {
                        "enabled": True,
                        "strategy": "parabola_apex",
                        "n_apexes": len(peaks),
                    }
                    log.info(
                        "[STAGE 2/4 DETECT POINTS] parabola-apex: %d arch vertices",
                        len(peaks),
                    )
                else:
                    log.warning(
                        "[STAGE 2/4 DETECT POINTS] parabola-apex found %d (<min); "
                        "falling back to serration peaks",
                        len(apexes),
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning("[PARABOLA-APEX] skipped: %s", exc)

        # Optional multi-scale structural filter: keep only CV peaks aligned
        # with true asperities from wedge → top-hat ridge → bidirectional track.
        ms_cfg = config.get("multi_scale", {}) or {}
        if not used_parabola_apex and bool(ms_cfg.get("use_multi_scale_std", False)):
            try:
                ms = run_multi_scale(roi.image, config)
                before = len(peaks)
                filtered = filter_peaks_by_multi_scale_y(
                    peaks,
                    ms.peaks,
                    y_tol_px=float(ms_cfg.get("y_tol_px", 12.0)),
                    x_tol_px=ms_cfg.get("x_tol_px"),
                )
                min_keep = max(3, int(0.15 * max(before, 1)))
                if len(filtered) >= min_keep:
                    peaks = filtered
                    strategy = "filter_cv_by_ms_y"
                elif len(ms.peaks) > 0:
                    # Domains differ (common on synthetic / alternate orientations):
                    # use structural MS peaks, snap to CV/edge for sub-pixel.
                    peaks = refine_ms_peaks_with_cv(
                        ms.peaks,
                        edge.peak_locations,
                        edge.edge_points,
                        snap_px=float(ms_cfg.get("snap_px", 30.0)),
                    )
                    strategy = "ms_peaks_snap_cv"
                else:
                    peaks = edge.peak_locations
                    strategy = "passthrough_cv"
                multi_scale_meta = {
                    **ms.meta,
                    "n_cv_before": int(before),
                    "n_cv_after": int(len(peaks)),
                    "n_filtered": int(len(filtered)),
                    "strategy": strategy,
                    "enabled": True,
                }
                log.info(
                    "[STAGE 2/4 DETECT POINTS] multi-scale %s: %d → %d peaks",
                    strategy,
                    before,
                    len(peaks),
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("[MULTI-SCALE] filter skipped: %s", exc)
                multi_scale_meta = {"enabled": True, "error": str(exc)}

        # Optional Qwen2.5-VL hybrid: VLM proposes apexes, CV snaps to sub-pixel.
        if mode == "qwen" and not used_parabola_apex:
            from sem_analysis.qwen_peaks import detect_peaks_qwen_refined

            qwen_peaks, qwen_meta = detect_peaks_qwen_refined(
                roi.image, config, peaks if len(peaks) else edge.peak_locations
            )
            peaks = qwen_peaks
            log.info(
                "[STAGE 2/4 DETECT POINTS] Qwen hybrid status=%s peaks=%d (cv_candidates=%d)",
                qwen_meta.get("status"),
                len(peaks),
                len(edge.peak_locations),
            )

        n_candidates = len(peaks)
        log.info(
            "[STAGE 2/4 DETECT POINTS] mode=%s peaks=%d (all asperity candidates)",
            mode,
            n_candidates,
        )
        h, w = roi.image.shape[:2]
        border_px = int(config.get("tip_validation", {}).get("border_px", 10))
        # Cropped / edge-touching apexes are incomplete tips; drop them silently
        # so they don't clutter the Results panel as "marked without R".
        drop_border_tips = bool(config.get("tip_validation", {}).get("drop_border_tips", True))

        tip_id = 0
        # Precompute tip Ys so Method 2 can cap fit depth before the next serration
        peak_ys = [float(np.asarray(p, dtype=float)[1]) for p in peaks]

        for peak_i, peak in enumerate(peaks):
            apex = np.asarray(peak, dtype=float)
            ax, ay = float(apex[0]), float(apex[1])
            border_ok = border_px <= ax < (w - border_px) and border_px <= ay < (h - border_px)
            if not border_ok and drop_border_tips:
                # Skip incomplete border tips entirely (not a real measurable asperity)
                continue
            branches = _branches_from_edge_points(
                edge_points, apex, window_y, window_x, min_pts=min_branch
            )
            left_ok = branches is not None
            right_ok = branches is not None
            hard_ok = bool(border_ok and branches is not None)

            # Half-gap to nearest other peak (nm) — keeps Method 2 out of Λ valleys
            m2_depth_cap = None
            if len(peak_ys) >= 2:
                gaps = [abs(ay - other_y) for j, other_y in enumerate(peak_ys) if j != peak_i]
                if gaps:
                    m2_depth_cap = 0.45 * min(gaps) * nm_per_px
                    m2_depth_cap = float(max(40.0, min(m2_depth_cap, 180.0)))

            tm = TipMeasurement(
                tip_id=tip_id,
                apex_x_px=ax + ox,
                apex_y_px=ay + oy,
                nm_per_px=nm_per_px,
                border_valid=border_ok,
                left_branch_valid=left_ok,
                right_branch_valid=right_ok,
                fit_residual_px=0.0,
                window_valid=hard_ok,
                hard_valid=hard_ok,
                rejection_reason=None if hard_ok else (
                    "touches_border" if not border_ok else "insufficient_branches"
                ),
                confidence=0.0,
            )

            if not hard_ok:
                # Still record apex so annotation can mark the detected point
                tm.method1 = {
                    "tip_point": [ax, ay],
                    "peak_location": [ax, ay],
                    "valid": False,
                    "rejection_reason": tm.rejection_reason,
                }
                stage3_marks.append({
                    "tip_id": tip_id,
                    "will_draw_apex": True,
                    "will_draw_left": False,
                    "will_draw_right": False,
                    "will_draw_scan_line": False,
                    "will_draw_vertical_l": False,
                    "will_draw_circle": False,
                })
                tips.append(tm)
                tip_id += 1
                continue

            left, right = branches
            log.info(
                "[STAGE 2/4 DETECT POINTS] tip_id=%s apex=(%.1f,%.1f) branches L=%d R=%d — Method 1",
                tip_id, ax, ay, len(left), len(right),
            )
            primary, point_rec, mark_rec, rad_rec = _apply_method1_to_tip(
                tm, apex, left, right, nm_per_px, proto, tip_id,
                skip_stability=skip_stab,
                stability_threshold=stab_thr,
            )
            if point_rec:
                stage2_points.append(point_rec)
                log.info(
                    "[STAGE 2/4 DETECT POINTS] tip_id=%s apex=%s left=%s right=%s found=%s reason=%s",
                    tip_id, point_rec.get("apex"), point_rec.get("left"), point_rec.get("right"),
                    point_rec.get("found_all_three"), point_rec.get("rejection_reason"),
                )
            if mark_rec:
                stage3_marks.append(mark_rec)
            if rad_rec:
                stage4_radii.append(rad_rec)
                if rad_rec.get("valid"):
                    log.info(
                        "[STAGE 4/4 PROCEED RADIUS] tip_id=%s R=%.4f nm — OK",
                        tip_id, rad_rec["radius_nm"],
                    )
                else:
                    log.warning(
                        "[STAGE 4/4 PROCEED RADIUS] tip_id=%s FAILED reason=%s",
                        tip_id, rad_rec.get("rejection_reason"),
                    )

            # Methods 2–3 on same tip (best-effort — always record reason on failure)
            try:
                m2 = measure_projected_tip_distance(
                    nm_per_pixel=nm_per_px,
                    fit_band_nm=proto["method2_fit_band_nm"],
                    apex=apex,
                    left=left,
                    right=right,
                    tip_id=tip_id,
                    min_flank_points=int(
                        method_cfg.get("projected_tip_distance", {}).get("min_flank_points", 3)
                    ),
                    min_cross=float(
                        config.get("tip_validation", {}).get("method2_min_cross", 0.08)
                    ),
                    max_distance_nm=float(
                        config.get("tip_validation", {}).get("method2_max_l_nm", 250.0)
                    ),
                    max_band_depth_nm=m2_depth_cap,
                )
                tm.method2 = projected_tip_distance_to_dict(m2)
                tm.method2_valid = bool(m2.valid)
                if not tm.method2_valid:
                    log.info(
                        "[METHOD 2] tip_id=%s FAILED reason=%s",
                        tip_id,
                        tm.method2.get("rejection_reason"),
                    )
            except Exception as exc:
                log.warning("method2 tip %s failed: %s", tip_id, exc)
                tm.method2 = {
                    "tip_id": tip_id,
                    "tip_point": [float(apex[0]), float(apex[1])],
                    "peak_location": [float(apex[0]), float(apex[1])],
                    "valid": False,
                    "rejection_reason": f"exception:{type(exc).__name__}",
                }
                tm.method2_valid = False

            try:
                m3 = measure_inscribed_angle(
                    apex=apex,
                    left=left,
                    right=right,
                    nm_per_pixel=nm_per_px,
                    circle_diameter_nm=float(proto.get("method3_circle_diameter_nm", 100)),
                    tip_id=tip_id,
                )
                tm.method3 = inscribed_angle_to_dict(m3)
                tm.method3_valid = bool(m3.valid)
            except Exception as exc:
                log.warning("method3 tip %s failed: %s", tip_id, exc)
                tm.method3 = {
                    "tip_id": tip_id,
                    "tip_point": [float(apex[0]), float(apex[1])],
                    "peak_location": [float(apex[0]), float(apex[1])],
                    "valid": False,
                    "rejection_reason": f"exception:{type(exc).__name__}",
                }
                tm.method3_valid = False

            if tm.method1_valid:
                tm.confidence = tip_confidence(
                    edge_score=0.8,
                    symmetry_score=1.0,
                    fit_score=0.8,
                    continuity_score=1.0,
                    consensus_score=0.8,
                    weights={
                        "edge": 0.25, "continuity": 0.20, "fit": 0.20,
                        "symmetry": 0.15, "consensus": 0.20,
                    },
                )
            tips.append(tm)
            tip_id += 1

    else:
        # Legacy complete-arch path
        edge_maps = preprocess_sem(roi.image)
        arches = detect_validated_arches(roi.image, nm_per_px, config, edge_maps=edge_maps)
        n_candidates = len(arches)
        log.info("[STAGE 2/4 DETECT POINTS] arches_found=%d", len(arches))
        for arch in arches:
            if arch.tip_id < 0:
                continue
            hard_ok, reason = _hard_valid(arch)
            apex = np.array([arch.apex_x_px, arch.apex_y_px], dtype=float)
            left = arch.left_smooth if arch.left_smooth is not None else arch.left_raw
            right = arch.right_smooth if arch.right_smooth is not None else arch.right_raw
            tm = TipMeasurement(
                tip_id=arch.tip_id,
                apex_x_px=arch.apex_x_px + ox,
                apex_y_px=arch.apex_y_px + oy,
                nm_per_px=nm_per_px,
                border_valid=arch.border_valid,
                left_branch_valid=arch.left_branch_valid,
                right_branch_valid=arch.right_branch_valid,
                fit_residual_px=arch.fit_residual_px,
                window_valid=arch.window_valid,
                hard_valid=hard_ok,
                rejection_reason=reason,
                confidence=0.0,
            )
            if not hard_ok or left is None or right is None:
                tm.method2 = {
                    "tip_id": arch.tip_id,
                    "valid": False,
                    "rejection_reason": reason or "insufficient_branches",
                }
                tm.method3 = {
                    "tip_id": arch.tip_id,
                    "valid": False,
                    "rejection_reason": reason or "insufficient_branches",
                }
                tips.append(tm)
                continue
            primary, point_rec, mark_rec, rad_rec = _apply_method1_to_tip(
                tm, apex, left, right, nm_per_px, proto, arch.tip_id,
                skip_stability=False, stability_threshold=stab_thr,
            )
            if point_rec:
                stage2_points.append(point_rec)
            if mark_rec:
                stage3_marks.append(mark_rec)
            if rad_rec:
                stage4_radii.append(rad_rec)
            try:
                m2 = measure_projected_tip_distance(
                    nm_per_pixel=nm_per_px,
                    fit_band_nm=proto["method2_fit_band_nm"],
                    apex=apex,
                    left=left,
                    right=right,
                    tip_id=arch.tip_id,
                    min_flank_points=int(
                        method_cfg.get("projected_tip_distance", {}).get("min_flank_points", 3)
                    ),
                    min_cross=float(
                        config.get("tip_validation", {}).get("method2_min_cross", 0.08)
                    ),
                    max_distance_nm=float(
                        config.get("tip_validation", {}).get("method2_max_l_nm", 250.0)
                    ),
                )
                tm.method2 = projected_tip_distance_to_dict(m2)
                tm.method2_valid = bool(m2.valid)
            except Exception as exc:
                tm.method2 = {
                    "tip_id": arch.tip_id,
                    "valid": False,
                    "rejection_reason": f"exception:{type(exc).__name__}",
                }
                tm.method2_valid = False
            try:
                m3 = measure_inscribed_angle(
                    apex=apex,
                    left=left,
                    right=right,
                    nm_per_pixel=nm_per_px,
                    circle_diameter_nm=float(proto.get("method3_circle_diameter_nm", 100)),
                    tip_id=arch.tip_id,
                )
                tm.method3 = inscribed_angle_to_dict(m3)
                tm.method3_valid = bool(m3.valid)
            except Exception as exc:
                tm.method3 = {
                    "tip_id": arch.tip_id,
                    "valid": False,
                    "rejection_reason": f"exception:{type(exc).__name__}",
                }
                tm.method3_valid = False
            tips.append(tm)

    # Shift Method 1 geometry from ROI → full-image coords for annotation
    for t in tips:
        if t.method1:
            t.method1 = _shift_method_dict(t.method1, ox, oy)
        if t.method2:
            t.method2 = _shift_method_dict(t.method2, ox, oy)
        if t.method3:
            t.method3 = _shift_method_dict(t.method3, ox, oy)

    for rec in stage2_points:
        for k in ("apex", "left", "right"):
            if rec.get(k):
                rec[k] = _shift_xy(rec[k], ox, oy)
        if rec.get("scan_line"):
            rec["scan_line"] = _shift_line(rec["scan_line"], ox, oy)
        if rec.get("vertical_l_line"):
            rec["vertical_l_line"] = _shift_line(rec["vertical_l_line"], ox, oy)
    for rec in stage4_radii:
        if rec.get("center"):
            rec["center"] = _shift_xy(rec["center"], ox, oy)

    # Summaries — mark ALL detected tips; valid R100 in per_curve, rest in failed_curves
    measured = [t for t in tips if t.method1]
    accepted = [t for t in tips if t.hard_valid]

    def _vals(getter):
        return [v for v in (getter(t) for t in tips) if v is not None]

    per_curve = []
    failed_curves = []
    for t in tips:
        base = {
            "peak_id": t.tip_id,
            "peak_location": [t.apex_x_px, t.apex_y_px],
            "tip_point": [t.apex_x_px, t.apex_y_px],
            "confidence": t.confidence,
            **(t.method1 or {}),
        }
        base["peak_location"] = [t.apex_x_px, t.apex_y_px]
        if t.method1_valid:
            per_curve.append(base)
        else:
            base["rejection_reason"] = (
                (t.method1 or {}).get("method1_rejection_reason")
                or (t.method1 or {}).get("rejection_reason")
                or t.rejection_reason
                or "invalid"
            )
            base["valid"] = False
            failed_curves.append(base)

    m1_stats = summarize_values(
        _vals(
            lambda t: (t.method1.get("projected_radius_nm") or t.method1.get("radius_nm"))
            if t.method1_valid
            else None
        )
    )
    m2_stats = summarize_values(
        _vals(lambda t: t.method2.get("distance_l_nm") if t.method2_valid else None)
    )
    m3_stats = summarize_values(
        _vals(lambda t: t.method3.get("angle_degrees") if t.method3_valid else None)
    )

    summary = {
        "image_id": image_id,
        "n_detected_arches": n_candidates,
        "n_detected_peaks": n_candidates,
        "n_hard_valid": len(accepted),
        "protocol": proto,
        "nm_per_px": nm_per_px,
        "tip_detection_mode": mode,
        "multi_scale": multi_scale_meta,
        "tilt_note": "Measurements are projected (tilt metadata stored; no blind 2× correction).",
        "fixed_distance_circle": {
            "headline": "median",
            "label": "Method 1 — Fixed distance inscribed circle",
            **m1_stats,
            "median_radius_nm": m1_stats.get("median"),
            "mean_radius_nm": m1_stats.get("mean"),
            "std_radius_nm": m1_stats.get("std"),
            "count": sum(1 for t in tips if t.method1_valid),
            "n_marked": len(tips),
            "per_curve": per_curve,
            "failed_curves": failed_curves,
        },
        "projected_tip_distance": {
            "headline": "median",
            **m2_stats,
            "median_distance_l_nm": m2_stats.get("median"),
            "count": sum(1 for t in tips if t.method2_valid),
            "n_marked": len(tips),
            "per_curve": [
                {"peak_id": t.tip_id, "peak_location": [t.apex_x_px, t.apex_y_px], **(t.method2 or {})}
                for t in tips if t.method2_valid
            ],
            "failed_curves": [
                {
                    "peak_id": t.tip_id,
                    "peak_location": [t.apex_x_px, t.apex_y_px],
                    **(t.method2 or {}),
                    "valid": False,
                    "rejection_reason": (t.method2 or {}).get("rejection_reason")
                    or t.rejection_reason
                    or "invalid",
                }
                for t in tips if not t.method2_valid
            ],
        },
        "inscribed_angle": {
            "headline": "median",
            **m3_stats,
            "count": sum(1 for t in tips if t.method3_valid),
            "n_marked": len(tips),
            "per_curve": [
                {"peak_id": t.tip_id, "peak_location": [t.apex_x_px, t.apex_y_px], **(t.method3 or {})}
                for t in tips if t.method3_valid
            ],
            "failed_curves": [
                {
                    "peak_id": t.tip_id,
                    "peak_location": [t.apex_x_px, t.apex_y_px],
                    **(t.method3 or {}),
                    "valid": False,
                    "rejection_reason": (t.method3 or {}).get("rejection_reason")
                    or t.rejection_reason
                    or "invalid",
                }
                for t in tips if not t.method3_valid
            ],
        },
        "tip_validation": {
            "n_accepted": len(accepted),
            "n_rejected": len(tips) - len(accepted),
            "n_measured": len(measured),
        },
        "diagnostics": _build_diagnostics(tips, proto, nm_per_px),
        "debug_stages": {
            "stage2_detect_points": {
                "status": "ok" if stage2_points or tips else "no_tips",
                "mode": mode,
                "n_candidates": n_candidates,
                "n_tips": len(tips),
                "n_with_r100": sum(1 for t in tips if t.method1_valid),
                "points": stage2_points,
            },
            "stage3_mark_points": {
                "status": "ok",
                "marks": stage3_marks,
            },
            "stage4_proceed_radius": {
                "status": "ok",
                "n_valid_r100": sum(1 for r in stage4_radii if r.get("valid")),
                "radii": stage4_radii,
            },
        },
    }
    m2_vals = _vals(lambda t: t.method2.get("distance_l_nm") if t.method2_valid else None)
    if m2_vals:
        summary["projected_tip_distance"]["median_distance_l_nm"] = float(np.median(m2_vals))

    fdc = summary["fixed_distance_circle"]
    fdc["median_radius_nm"] = fdc.get("median")
    fdc["mean_radius_nm"] = fdc.get("mean")

    # Blade value / whiteboard best-effort (may be empty for ridge mode)
    try:
        summary["blade_value"] = build_blade_value_table(tips)
    except Exception:
        summary["blade_value"] = None

    log.info(
        "[APPROACH 1] peaks=%d marked=%d R100_ok=%d median=%s",
        n_candidates,
        len(tips),
        summary["fixed_distance_circle"]["count"],
        summary["fixed_distance_circle"].get("median"),
    )
    return tips, summary


def _shift_method_dict(d: dict, ox: float, oy: float) -> dict:
    """Shift geometric fields from ROI to full-image coordinates."""
    out = dict(d)
    for key in ("tip_point", "peak_location", "center", "intersection_left", "intersection_right",
                "circle_center", "ultimate_tip", "projected_tip", "convergence_point"):
        if out.get(key):
            out[key] = _shift_xy(out[key], ox, oy)
    for key in ("scan_line", "vertical_l_line", "left_line", "right_line",
                "left_tangent_line", "right_tangent_line", "radius_spoke", "diameter_line"):
        if out.get(key):
            out[key] = _shift_line(out[key], ox, oy)
    for key in ("tip_apex_arc", "curve_points", "arc_points"):
        if out.get(key):
            out[key] = _shift_poly(out[key], ox, oy)
    return out


def tips_to_dataframe(tips: list[TipMeasurement], image_id: str = "") -> pd.DataFrame:
    rows = []
    for t in tips:
        row = t.to_row()
        row["image_id"] = image_id
        rows.append(row)
    cols = [
        "image_id", "tip_id", "apex_x_px", "apex_y_px", "nm_per_px",
        "border_valid", "left_branch_valid", "right_branch_valid", "fit_residual_px",
        "R25_nm", "R50_nm", "R100_nm", "R200_nm",
        "projected_tip_distance_nm", "included_angle_deg", "area_under_curve_nm2",
        "angle_D100_deg",
        "method1_valid", "method2_valid", "method3_valid",
        "confidence", "rejection_reason", "hard_valid", "window_valid",
    ]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols] if len(df) else pd.DataFrame(columns=cols)
