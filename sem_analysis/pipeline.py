"""End-to-end SEM analysis pipeline orchestration."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from sem_analysis.annotation import (
    annotate_approach3_fixed_distance_image,
    annotate_approach3_image,
    annotate_circular_arc_image,
    annotate_image,
    annotate_method1_image,
    annotate_method2_image,
    annotate_method3_image,
    annotate_research_image,
    annotate_validated_tips,
    annotate_whiteboard_image,
    match_fixed_distance_to_tips,
)
from sem_analysis.deduction import FilteredDetection, apply_deduction
from sem_analysis.edge_detection import EdgePeakResult, detect_edges_and_peaks, detect_serration_peaks_global
from sem_analysis.io.image_loader import (
    SEMImage,
    apply_tilt_correction,
    build_calibration_record,
    load_image,
    save_image,
)
from sem_analysis.roi import extract_measurement_roi
from sem_analysis.tip_measurement import measure_all_tips, tips_to_dataframe
from sem_analysis.methods.brainstorming import run_brainstorming_all_peaks, run_brainstorming_methods
from sem_analysis.methods.parabola_approach2 import find_parabola_curves
from sem_analysis.methods.openai_vlm_approach3 import run_openai_vlm_approach
from sem_analysis.research.osculating_tip import measure_all_osculating_tips, osculating_tip_to_dict
from sem_analysis.preprocessing import ProcessedImage, preprocess
from sem_analysis.radius_computation import (
    RadiusResult,
    TipCondition,
    aggregate_radii,
    classify_tip_condition,
    compute_radius,
)
from sem_analysis.shape_detection import DetectedShape, detect_shapes
from sem_analysis.validation import (
    align_predictions,
    compute_error_metrics,
    generate_validation_report,
    load_ground_truth,
)

log = logging.getLogger("sem-api.stages")


# Canonical analysis stages — arch-first protocol (default)
PIPELINE_STAGES = [
    "original_sem_image",
    "per_image_scale_bar_calibration",
    "footer_exclusion_border_margin",
    "clahe_bilateral_preprocess",
    "canny_scharr_edge_probability",
    "complete_arch_detection",
    "branch_validation_measurement_window",
    "resample_smooth_edges",
    "method1_method2_method3_nm",
    "hard_validity_before_confidence",
    "unified_tip_csv",
]

PIPELINE_STAGES_LEGACY = [
    "original_sem_image",
    "crop_metadata_scale_bar",
    "grayscale_clahe_bilateral",
    "multi_algorithm_consensus_edges",
    "morph_close_remove_small_components",
    "contour_filtering_blade_edge_selection",
    "tip_detection",
    "curve_or_circle_fitting",
    "pixel_to_nm_conversion",
]


def _arch_first_enabled(config: dict) -> bool:
    return bool(config.get("pipeline", {}).get("arch_first", True))


def _legacy_peak_enabled(config: dict) -> bool:
    return bool(config.get("pipeline", {}).get("legacy_peak_detection", False))


def load_config(config_path: str | Path | None = None) -> dict:
    """Load YAML configuration with defaults."""
    default_path = Path(__file__).resolve().parents[1] / "config" / "default_config.yaml"
    path = Path(config_path) if config_path else default_path

    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class AnalysisResult:
    """Complete pipeline output for one image."""

    source_path: str
    nm_per_pixel: float
    shapes_detected: int
    shapes_passed: int
    radius_results: list[RadiusResult]
    aggregation: dict
    tip_condition: str | None
    detections: list[FilteredDetection] = field(repr=False)
    edge_results: list[EdgePeakResult] = field(repr=False)
    alternative_methods: dict = field(default_factory=dict)
    brainstorming_methods: dict = field(default_factory=dict)
    primary_method: str = "projected_tip_distance"
    validation: dict | None = None
    annotated_image_path: str | None = None
    annotated_method_paths: dict = field(default_factory=dict)
    research_grade: dict = field(default_factory=dict)
    annotated_research_path: str | None = None
    tilt_correction: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    protocol: dict = field(default_factory=dict)
    pipeline_stages: list[str] = field(default_factory=lambda: list(PIPELINE_STAGES))

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "source_path": self.source_path,
            "nm_per_pixel": self.nm_per_pixel,
            "calibration": self.calibration,
            "protocol": self.protocol or (self.brainstorming_methods or {}).get("protocol", {}),
            "shapes_detected": self.shapes_detected,
            "shapes_passed": self.shapes_passed,
            "radius_results": [
                {
                    "peak_id": r.peak_id,
                    "shape_id": r.shape_id,
                    "radius_px": r.radius_px,
                    "radius_nm": r.radius_nm,
                    "radius_angstrom": r.radius_angstrom,
                    "fit_residual": r.fit_residual,
                    "center": r.center,
                    "method": r.method,
                    "confidence_score": r.confidence_score,
                    "opening_angle_deg": r.opening_angle_deg,
                    "peak_location": r.peak_location,
                }
                for r in self.radius_results
            ],
            "aggregation": self.aggregation,
            "tip_condition": self.tip_condition,
            "primary_method": self.primary_method,
            "brainstorming_methods": self.brainstorming_methods,
            "alternative_methods": self.brainstorming_methods,
            "validation": self.validation,
            "annotated_image_path": self.annotated_image_path,
            "annotated_method_paths": self.annotated_method_paths,
            "research_grade": self.research_grade,
            "annotated_research_path": self.annotated_research_path,
            "tilt_correction": self.tilt_correction,
            "pipeline_stages": self.pipeline_stages or PIPELINE_STAGES,
            "debug_stages": (self.brainstorming_methods or {}).get("debug_stages")
            or (self.alternative_methods or {}).get("debug_stages"),
        }


class SEMAnalysisPipeline:
    """Modular SEM image analysis pipeline."""

    def __init__(self, config: dict | None = None, config_path: str | Path | None = None):
        self.config = config or load_config(config_path)

    def analyze(
        self,
        image_path: str | Path,
        output_dir: str | Path | None = None,
        ground_truth_path: str | Path | None = None,
        run_alternative_methods: bool = True,
    ) -> AnalysisResult:
        """Run full analysis pipeline on a single SEM image.

        Stages:
          1. Original SEM image
          2. Crop metadata and scale-bar area
          3. Noise reduction
          4. Local contrast enhancement
          5. Edge detection
          6. Contour filtering
          7. Blade-edge selection
          8. Tip detection
          9. Curve or circle fitting
         10. Pixel-to-nm conversion (incl. tilt correction)
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir) if output_dir else image_path.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        arch_first = _arch_first_enabled(self.config)
        legacy_peaks = _legacy_peak_enabled(self.config)
        pipeline_stages = PIPELINE_STAGES if arch_first else PIPELINE_STAGES_LEGACY

        log.info("=" * 60)
        log.info("[STAGE 1/4 ANALYZE IMAGE] loading %s", image_path.name)

        # [1] Original SEM image — per-image calibration (no cross-image averaging)
        sem_image = load_image(
            image_path,
            default_nm_per_pixel=self.config.get("calibration", {}).get("nm_per_pixel", 1.0),
            config=self.config,
        )

        # Tilt metadata stored; blind 2× correction off by default
        nm_corrected, tilt_info = apply_tilt_correction(sem_image.nm_per_pixel, self.config)
        sem_image.nm_per_pixel = nm_corrected
        sem_image.tilt_correction = tilt_info

        processed = preprocess(sem_image.data, sem_image.nm_per_pixel, self.config)
        shapes = detect_shapes(processed.data, self.config)

        h, w = processed.data.shape[:2]
        log.info(
            "[STAGE 1/4 ANALYZE IMAGE] size=%dx%d nm/px=%.6f (raw=%.6f) shapes=%d tilt_applied=%s",
            w,
            h,
            processed.nm_per_pixel,
            getattr(sem_image, "nm_per_pixel_raw", sem_image.nm_per_pixel) or sem_image.nm_per_pixel,
            len(shapes),
            bool(tilt_info.get("applied")),
        )
        if abs(float(processed.nm_per_pixel) - 1.0) < 1e-9:
            log.warning(
                "[STAGE 1/4 ANALYZE IMAGE] nm/px is 1.0 — likely missing calibration. "
                "Enter nm per pixel in the UI or Stage 2/4 (100 nm chord) will be wrong."
            )

        # Legacy skyline/Harris peak + Hough path (optional; accepts border peaks)
        global_edge = None
        edge_results: list[EdgePeakResult] = []
        radius_by_shape: dict[int, list[RadiusResult]] = {}
        all_radii: list[RadiusResult] = []
        detections: list[FilteredDetection] = []

        if legacy_peaks:
            global_edge = detect_serration_peaks_global(processed.data, self.config)
            edge_results = [global_edge]
            radius_cfg = self.config.get("radius", {})
            fit_method = radius_cfg.get("primary_method", "hough")

            global_radii: list[RadiusResult] = []
            for i, peak in enumerate(global_edge.peak_locations):
                px, py = float(peak[0]), float(peak[1])
                r = compute_radius(
                    global_edge.edge_points,
                    peak_id=i,
                    shape_id=0,
                    nm_per_pixel=processed.nm_per_pixel,
                    method=fit_method,
                    image=processed.data,
                    peak=(px, py),
                    config=self.config,
                )
                if r:
                    global_radii.append(r)
            radius_by_shape[0] = global_radii

            for shape in shapes:
                if shape.shape_id == 0:
                    continue
                edge = detect_edges_and_peaks(
                    processed.data, shape.shape_id, shape.contour, self.config
                )
                edge_results.append(edge)
                radii: list[RadiusResult] = []
                for i, peak in enumerate(edge.peak_locations):
                    px, py = float(peak[0]), float(peak[1])
                    r = compute_radius(
                        edge.edge_points,
                        peak_id=i,
                        shape_id=shape.shape_id,
                        nm_per_pixel=processed.nm_per_pixel,
                        method=fit_method,
                        image=processed.data,
                        peak=(px, py),
                        config=self.config,
                    )
                    if r:
                        radii.append(r)
                radius_by_shape[shape.shape_id] = radii

            detections = apply_deduction(
                shapes, processed.data.shape, self.config, radius_by_shape
            )
            all_radii = list(global_radii)
            for det in detections:
                if det.passed:
                    shape_radii = radius_by_shape.get(det.shape.shape_id, [])
                    if det.shape.shape_id != 0:
                        det.radius_results = shape_radii
                        for r in shape_radii:
                            r.confidence_score = det.confidence_score
                        all_radii.extend(shape_radii)
            if len(all_radii) > 1:
                kept: list[RadiusResult] = []
                for r in all_radii:
                    if r.peak_location is None:
                        kept.append(r)
                        continue
                    if all(
                        np.linalg.norm(np.array(r.peak_location) - np.array(k.peak_location)) > 8
                        for k in kept if k.peak_location
                    ):
                        kept.append(r)
                all_radii = kept

        # Arch-first protocol: footer ROI → complete arches → Methods 1–3 (same tip IDs)
        brainstorming_methods: dict = {}
        annotated_method_paths: dict = {}
        tip_rows_df = None
        protocol_tips: list = []

        if arch_first or run_alternative_methods:
            border_m = int(self.config.get("preprocessing", {}).get("border_margin_px", 10))
            roi = extract_measurement_roi(sem_image.data, border_margin_px=border_m)
            protocol_tips, brainstorming_methods = measure_all_tips(
                roi,
                processed.nm_per_pixel,
                self.config,
                image_id=image_path.name,
            )
            tip_rows_df = tips_to_dataframe(protocol_tips, image_id=image_path.name)

            # Enrich stage 1 diagnostics for the client console
            stages = brainstorming_methods.setdefault("debug_stages", {})
            stages["stage1_analyze_image"] = {
                "status": "ok",
                "image": image_path.name,
                "width_px": int(w),
                "height_px": int(h),
                "nm_per_pixel": float(processed.nm_per_pixel),
                "n_shapes": len(shapes),
                "calibration_warning": abs(float(processed.nm_per_pixel) - 1.0) < 1e-9,
            }

            ann_base = processed.data
            method1_path = output_dir / f"{image_path.stem}_method1.png"
            method2_path = output_dir / f"{image_path.stem}_method2.png"
            method3_path = output_dir / f"{image_path.stem}_method3.png"

            m1_ok = brainstorming_methods.get("fixed_distance_circle", {}).get("per_curve", [])
            m1_fail = brainstorming_methods.get("fixed_distance_circle", {}).get("failed_curves", [])
            m1_curves = list(m1_ok) + list(m1_fail)
            m2_curves = brainstorming_methods.get("projected_tip_distance", {}).get("per_curve", [])
            m3_curves = brainstorming_methods.get("inscribed_angle", {}).get("per_curve", [])

            annotate_method1_image(
                ann_base, m1_curves, processed.nm_per_pixel, self.config,
                output_path=str(method1_path),
            )
            log.info(
                "[STAGE 3/4 MARK POINTS] wrote %s curves_drawn=%d (ok=%d failed=%d)",
                method1_path.name,
                len(m1_curves),
                len(m1_ok),
                len(m1_fail),
            )
            annotate_method2_image(
                ann_base, m2_curves, processed.nm_per_pixel, self.config,
                output_path=str(method2_path),
            )
            annotate_method3_image(
                ann_base, m3_curves, processed.nm_per_pixel, self.config,
                output_path=str(method3_path),
            )

            # Approach 2: vertex-form parabola curves (purple) + pink vertex
            approach2 = find_parabola_curves(
                ann_base, processed.nm_per_pixel, self.config
            )
            brainstorming_methods["approach2_parabolas"] = approach2
            # Keep legacy key so older UI still finds results
            brainstorming_methods["approach2_circular_arcs"] = approach2
            a2_path = output_dir / f"{image_path.stem}_method1_approach2.png"
            annotate_circular_arc_image(
                ann_base,
                approach2.get("per_curve") or [],
                processed.nm_per_pixel,
                self.config,
                output_path=str(a2_path),
            )
            log.info(
                "[APPROACH 2] wrote %s parabolas=%d median=%s",
                a2_path.name,
                approach2.get("count"),
                approach2.get("median_radius_nm"),
            )

            # Approach 3: OpenAI contour → OpenCV refine → circle fit (tips from Method 1)
            tip_seeds = []
            for c in m1_ok:
                loc = c.get("tip_point") or c.get("peak_location")
                if loc and len(loc) >= 2:
                    tip_seeds.append([float(loc[0]), float(loc[1])])
            try:
                approach3 = run_openai_vlm_approach(
                    ann_base,
                    processed.nm_per_pixel,
                    self.config,
                    tip_seeds=tip_seeds or None,
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("[APPROACH 3] failed (continuing without it): %s", exc)
                approach3 = {
                    "approach": "openai_vlm_circle_fit",
                    "label": "Approach 3 — OpenAI peaks + contour → circle fit",
                    "count": 0,
                    "peak_count": 0,
                    "per_curve": [],
                    "failed_curves": [],
                    "median_radius_nm": None,
                    "mean_radius_nm": None,
                    "std_radius_nm": None,
                    "nm_per_pixel": processed.nm_per_pixel,
                    "openai": {"ok": False, "error": str(exc)},
                }
            brainstorming_methods["approach3_openai_vlm"] = approach3
            a3_curves = list(approach3.get("per_curve") or []) + list(
                approach3.get("failed_curves") or []
            )
            a3_path = output_dir / f"{image_path.stem}_method1_approach3.png"
            a3_fd_path = output_dir / f"{image_path.stem}_method1_approach3_fixed_distance.png"
            try:
                annotate_approach3_image(
                    ann_base,
                    a3_curves,
                    processed.nm_per_pixel,
                    self.config,
                    output_path=str(a3_path),
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("[APPROACH 3] annotate failed: %s", exc)
                a3_path = output_dir / f"{image_path.stem}_method1_approach3.png"
            # Second image: fixed-distance circle construction on OpenAI Vision tips
            a3_fd_curves = match_fixed_distance_to_tips(a3_curves, m1_curves)
            brainstorming_methods["approach3_fixed_distance"] = {
                "approach": "openai_tips_fixed_distance_circle",
                "label": "OpenAI tips + fixed-distance inscribed circle",
                "count": sum(1 for c in a3_fd_curves if c.get("valid") and c.get("radius_nm") is not None),
                "per_curve": a3_fd_curves,
                "median_radius_nm": None,
            }
            fd_radii = [
                float(c["radius_nm"])
                for c in a3_fd_curves
                if c.get("valid") and c.get("radius_nm") is not None
            ]
            if fd_radii:
                brainstorming_methods["approach3_fixed_distance"]["median_radius_nm"] = float(
                    np.median(fd_radii)
                )
                brainstorming_methods["approach3_fixed_distance"]["mean_radius_nm"] = float(
                    np.mean(fd_radii)
                )
            try:
                annotate_approach3_fixed_distance_image(
                    ann_base,
                    a3_fd_curves,
                    processed.nm_per_pixel,
                    self.config,
                    output_path=str(a3_fd_path),
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("[APPROACH 3] fixed-distance annotate failed: %s", exc)
                a3_fd_path = output_dir / f"{image_path.stem}_method1_approach3_fixed_distance.png"
            log.info(
                "[APPROACH 3] wrote %s fitted=%d mean=%s std=%s openai_ok=%s",
                a3_path.name,
                approach3.get("count"),
                approach3.get("mean_radius_nm"),
                approach3.get("std_radius_nm"),
                (approach3.get("openai") or {}).get("ok"),
            )
            log.info(
                "[APPROACH 3] fixed-distance overlay wrote %s tips=%d with_R=%d",
                a3_fd_path.name,
                len(a3_fd_curves),
                brainstorming_methods["approach3_fixed_distance"]["count"],
            )

            annotated_method_paths = {
                "method1": str(method1_path),
                "method1_approach2": str(a2_path),
                "method1_approach3": str(a3_path),
                "method1_approach3_fixed_distance": str(a3_fd_path),
                "method2": str(method2_path),
                "method3": str(method3_path),
            }

            if tip_rows_df is not None and len(tip_rows_df):
                tips_csv = output_dir / f"{image_path.stem}_tips.csv"
                tip_rows_df.to_csv(tips_csv, index=False)
                annotated_method_paths["tips_csv"] = str(tips_csv)

        # Aggregation — protocol medians when arch-first; legacy Hough otherwise
        radius_cfg = self.config.get("radius", {})
        if arch_first and brainstorming_methods:
            m1 = brainstorming_methods.get("fixed_distance_circle", {})
            m2 = brainstorming_methods.get("projected_tip_distance", {})
            aggregation = {
                "count": m1.get("count", 0),
                "mean_radius_nm": m1.get("median_radius_nm") or m1.get("median"),
                "median_radius_nm": m1.get("median_radius_nm") or m1.get("median"),
                "std_radius_nm": m1.get("std"),
                "method2_median_l_nm": m2.get("median_distance_l_nm") or m2.get("median"),
                "n_hard_valid": brainstorming_methods.get("tip_validation", {}).get("n_accepted", 0),
                "n_marked": m1.get("n_marked"),
                "approach2_median_radius_nm": (
                    brainstorming_methods.get("approach2_parabolas")
                    or brainstorming_methods.get("approach2_circular_arcs")
                    or {}
                ).get("median_radius_nm"),
                "approach2_count": (
                    brainstorming_methods.get("approach2_parabolas")
                    or brainstorming_methods.get("approach2_circular_arcs")
                    or {}
                ).get("count"),
                "approach3_median_radius_nm": (
                    brainstorming_methods.get("approach3_openai_vlm") or {}
                ).get("median_radius_nm"),
                "approach3_count": (
                    brainstorming_methods.get("approach3_openai_vlm") or {}
                ).get("count"),
            }
            primary_method = "fixed_distance_circle"
        else:
            aggregation = aggregate_radii(
                all_radii, method=radius_cfg.get("aggregation", "mean")
            )
            primary_method = radius_cfg.get("primary_method", "hough")

        tip_condition = None
        mean_r = aggregation.get("mean_radius_nm") or aggregation.get("median_radius_nm")
        if mean_r is not None:
            tip_condition = classify_tip_condition(mean_r, self.config).value

        # Research-grade osculating (legacy peaks only — disabled by default)
        research_grade: dict = {}
        annotated_research_path: str | None = None
        research_cfg = self.config.get("research_grade", {})
        if (
            legacy_peaks
            and research_cfg.get("enabled", False)
            and global_edge is not None
            and len(global_edge.peak_locations) > 0
        ):
            osc_results, osc_summary = measure_all_osculating_tips(
                processed.data,
                global_edge.peak_locations,
                global_edge.edge_points,
                processed.nm_per_pixel,
                self.config,
            )
            per_curve = [osculating_tip_to_dict(r) for r in osc_results]
            research_grade = {
                "summary": osc_summary,
                "per_curve": per_curve,
                "pipeline": list(pipeline_stages),
            }

            research_path = output_dir / f"{image_path.stem}_research.png"
            annotate_research_image(
                processed.data,
                [c for c in per_curve if not c.get("rejected")],
                processed.nm_per_pixel,
                self.config,
                output_path=str(research_path),
            )
            annotated_research_path = str(research_path)
            annotated_method_paths["research"] = annotated_research_path

        # Overview annotation — whiteboard composite (matches reference SEM sketch)
        annotated_path = output_dir / f"{image_path.stem}_annotated.png"
        if arch_first:
            wb_tips = brainstorming_methods.get("whiteboard", {}).get("per_tip") or []
            if not wb_tips:
                wb_tips = [t.whiteboard for t in protocol_tips if getattr(t, "whiteboard", None)]
            if wb_tips:
                annotate_whiteboard_image(
                    processed.data,
                    wb_tips,
                    processed.nm_per_pixel,
                    self.config,
                    output_path=str(annotated_path),
                )
            else:
                overview_tips = []
                for t in protocol_tips:
                    overview_tips.append({
                        "tip_id": t.tip_id,
                        "apex_x_px": t.apex_x_px,
                        "apex_y_px": t.apex_y_px,
                        "peak_location": [t.apex_x_px, t.apex_y_px],
                        "hard_valid": t.hard_valid,
                    })
                annotate_validated_tips(
                    processed.data,
                    overview_tips,
                    processed.nm_per_pixel,
                    self.config,
                    output_path=str(annotated_path),
                )
            # Also export dedicated whiteboard PNG
            if wb_tips:
                wb_path = output_dir / f"{image_path.stem}_whiteboard.png"
                annotate_whiteboard_image(
                    processed.data,
                    wb_tips,
                    processed.nm_per_pixel,
                    self.config,
                    output_path=str(wb_path),
                )
                annotated_method_paths["whiteboard"] = str(wb_path)
        else:
            annotate_image(
                processed.data,
                detections,
                edge_results,
                processed.nm_per_pixel,
                self.config,
                output_path=str(annotated_path),
                all_radii=all_radii,
            )

        # [8] Validation
        validation_result = None
        if ground_truth_path:
            gt = load_ground_truth(ground_truth_path)
            max_dist = self.config.get("validation", {}).get("alignment_max_distance_px", 50)
            comparison = align_predictions(all_radii, gt, max_dist)
            metrics = compute_error_metrics(comparison)
            validation_result = generate_validation_report(
                comparison, metrics, output_dir, image_path.stem
            )

        calibration = build_calibration_record(sem_image, tilt_info)
        protocol_meta = dict(self.config.get("protocol") or {})
        if brainstorming_methods.get("protocol"):
            protocol_meta = {**protocol_meta, **brainstorming_methods["protocol"]}

        result = AnalysisResult(
            source_path=str(image_path),
            nm_per_pixel=processed.nm_per_pixel,
            shapes_detected=len(shapes),
            shapes_passed=sum(1 for d in detections if d.passed) if legacy_peaks else (
                brainstorming_methods.get("tip_validation", {}).get("n_accepted", 0)
            ),
            radius_results=all_radii,
            aggregation=aggregation,
            tip_condition=tip_condition,
            detections=detections,
            edge_results=edge_results,
            alternative_methods=brainstorming_methods,
            brainstorming_methods=brainstorming_methods,
            primary_method=primary_method,
            validation=validation_result,
            annotated_image_path=str(annotated_path),
            annotated_method_paths=annotated_method_paths,
            research_grade=research_grade,
            annotated_research_path=annotated_research_path,
            tilt_correction=tilt_info,
            calibration=calibration,
            protocol=protocol_meta,
            pipeline_stages=list(pipeline_stages),
        )

        self._export_reports(result, output_dir, image_path.stem)
        return result

    def _export_method_csvs(self, result: AnalysisResult, output_dir: Path, stem: str) -> None:
        """Export per-method per-curve CSV files."""
        bs = result.brainstorming_methods or {}

        m1 = bs.get("fixed_distance_circle", {}).get("per_curve", [])
        if m1:
            rows = []
            for c in m1:
                base = {
                    "peak_id": c.get("peak_id"),
                    "peak_x": c.get("peak_location", [None, None])[0],
                    "peak_y": c.get("peak_location", [None, None])[1],
                    "radius_nm": c.get("radius_nm"),
                    "radius_px": c.get("radius_px"),
                    "distance_l_nm": c.get("distance_l_nm"),
                    "label": c.get("label"),
                    "confidence": c.get("confidence"),
                }
                for label, rd in (c.get("radii_by_l") or {}).items():
                    base[f"{label}_nm"] = rd.get("radius_nm")
                rows.append(base)
            pd.DataFrame(rows).to_csv(output_dir / f"{stem}_method1_radii.csv", index=False)

        a2 = (
            bs.get("approach2_parabolas") or bs.get("approach2_circular_arcs") or {}
        ).get("per_curve", [])
        if a2:
            rows = [
                {
                    "curve_id": c.get("curve_id", c.get("peak_id")),
                    "vertex_x": (c.get("vertex") or c.get("peak_location") or [None, None])[0],
                    "vertex_y": (c.get("vertex") or c.get("peak_location") or [None, None])[1],
                    "a": c.get("a"),
                    "h": c.get("h"),
                    "k": c.get("k"),
                    "radius_px": c.get("radius_px"),
                    "radius_nm": c.get("radius_nm"),
                    "rel_residual": c.get("rel_residual"),
                    "residual_px": c.get("residual_px"),
                    "equation": c.get("equation"),
                }
                for c in a2
            ]
            pd.DataFrame(rows).to_csv(
                output_dir / f"{stem}_method1_approach2_radii.csv", index=False
            )

        a3 = (bs.get("approach3_openai_vlm") or {}).get("per_curve", [])
        if a3:
            rows = [
                {
                    "peak_id": c.get("peak_id"),
                    "peak_x": (c.get("peak_location") or [None, None])[0],
                    "peak_y": (c.get("peak_location") or [None, None])[1],
                    "radius_nm": c.get("radius_nm"),
                    "radius_px": c.get("radius_px"),
                    "center_x": (c.get("center") or [None, None])[0],
                    "center_y": (c.get("center") or [None, None])[1],
                    "fit_method": c.get("fit_method"),
                    "fit_residual_px": c.get("fit_residual_px"),
                    "vlm_confidence": c.get("vlm_confidence"),
                    "source": c.get("source"),
                    "valid": c.get("valid"),
                }
                for c in a3
            ]
            pd.DataFrame(rows).to_csv(
                output_dir / f"{stem}_method1_approach3_radii.csv", index=False
            )

        m2 = bs.get("projected_tip_distance", {}).get("per_curve", [])
        if m2:
            rows = [
                {
                    "peak_id": c.get("peak_id"),
                    "peak_x": c.get("peak_location", [None, None])[0],
                    "peak_y": c.get("peak_location", [None, None])[1],
                    "distance_l_nm": c.get("distance_l_nm"),
                    "distance_l_px": c.get("distance_l_px"),
                    "included_angle_deg": c.get("included_angle_deg"),
                    "area_under_curve_nm2": c.get("area_under_curve_nm2"),
                    "fit_band_nm_lo": (c.get("fit_band_nm") or [None, None])[0],
                    "fit_band_nm_hi": (c.get("fit_band_nm") or [None, None])[1],
                    "confidence": c.get("confidence"),
                }
                for c in m2
            ]
            pd.DataFrame(rows).to_csv(output_dir / f"{stem}_method2_radii.csv", index=False)

        bv = bs.get("blade_value") or {}
        if bv.get("per_tip"):
            pd.DataFrame(bv["per_tip"]).to_csv(output_dir / f"{stem}_blade_value.csv", index=False)
            # Append blade averages as a summary row file
            avg = bv.get("blade_value") or {}
            pd.DataFrame([{**avg, "tip_id": "BLADE_AVG"}]).to_csv(
                output_dir / f"{stem}_blade_value_avg.csv", index=False
            )

        m3 = bs.get("inscribed_angle", {}).get("per_curve", [])
        if m3:
            rows = [
                {
                    "peak_id": c.get("peak_id"),
                    "peak_x": c.get("peak_location", [None, None])[0],
                    "peak_y": c.get("peak_location", [None, None])[1],
                    "angle_degrees": c.get("angle_degrees"),
                    "circle_diameter_nm": c.get("circle_diameter_nm"),
                    "circle_diameter_px": c.get("circle_diameter_px"),
                    "label": c.get("label"),
                    "confidence": c.get("confidence"),
                }
                for c in m3
            ]
            pd.DataFrame(rows).to_csv(output_dir / f"{stem}_method3_radii.csv", index=False)

    def _export_research_csv(self, result: AnalysisResult, output_dir: Path, stem: str) -> None:
        """Export research-grade per-curve CSV."""
        per_curve = (result.research_grade or {}).get("per_curve", [])
        if not per_curve:
            return
        rows = [
            {
                "peak_id": c.get("peak_id"),
                "peak_x": c.get("peak_location", [None, None])[0],
                "peak_y": c.get("peak_location", [None, None])[1],
                "radius_um": c.get("radius_um"),
                "radius_nm": c.get("radius_nm"),
                "included_angle_deg": c.get("included_angle_deg"),
                "distance_l_nm": c.get("distance_l_nm"),
                "confidence_score": c.get("confidence_score"),
                "fit_residual_nm": c.get("fit_residual_nm"),
                "geometric_valid": c.get("geometric_valid"),
                "curvature_kappa": c.get("curvature_kappa"),
                "rejected": c.get("rejected"),
            }
            for c in per_curve
        ]
        pd.DataFrame(rows).to_csv(output_dir / f"{stem}_research_radii.csv", index=False)

    def _export_reports(self, result: AnalysisResult, output_dir: Path, stem: str) -> None:
        """Export JSON and CSV reports."""
        json_path = output_dir / f"{stem}_report.json"
        with open(json_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        self._export_method_csvs(result, output_dir, stem)
        self._export_research_csv(result, output_dir, stem)

        if result.radius_results:
            rows = [
                {
                    "peak_id": r.peak_id,
                    "shape_id": r.shape_id,
                    "radius_px": r.radius_px,
                    "radius_nm": r.radius_nm,
                    "radius_angstrom": r.radius_angstrom,
                    "opening_angle_deg": r.opening_angle_deg,
                    "fit_residual": r.fit_residual,
                    "confidence_score": r.confidence_score,
                    "method": r.method,
                }
                for r in result.radius_results
            ]
            pd.DataFrame(rows).to_csv(output_dir / f"{stem}_radii.csv", index=False)


def np_abs_dist(points: np.ndarray, x: float, y: float) -> np.ndarray:
    """Euclidean distance from points to (x, y)."""
    return np.sqrt((points[:, 0] - x) ** 2 + (points[:, 1] - y) ** 2)


def np_empty_2d() -> np.ndarray:
    return np.empty((0, 2))
