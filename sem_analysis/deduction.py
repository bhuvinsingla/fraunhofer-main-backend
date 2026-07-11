"""Deduction logic: rule-based filters and ML anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import IsolationForest

from sem_analysis.radius_computation import RadiusResult
from sem_analysis.shape_detection import DetectedShape


@dataclass
class FilteredDetection:
    """A detection that passed or failed deduction filters."""

    shape: DetectedShape
    passed: bool
    confidence_score: float
    anomaly_flag: bool
    rejection_reason: str | None = None
    radius_results: list[RadiusResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _rule_based_filter(
    shape: DetectedShape,
    image_shape: tuple[int, int],
    config: dict,
) -> tuple[bool, str | None]:
    """Apply deterministic rejection rules."""
    cfg = config.get("deduction", {})
    h, w = image_shape
    min_circ = cfg.get("min_circularity", 0.4)
    max_aspect = cfg.get("max_aspect_ratio", 5.0)
    min_solidity = cfg.get("min_solidity", 0.5)
    border_margin = cfg.get("border_margin_px", 10)
    min_area = config.get("shape_detection", {}).get("min_area_px", 50)

    if shape.area < min_area:
        return False, "area_below_minimum"

    if shape.circularity < min_circ:
        return False, "low_circularity"

    if shape.solidity < min_solidity:
        return False, "low_solidity"

    aspect = shape.major_axis / max(shape.minor_axis, 1e-6)
    if aspect > max_aspect:
        return False, "excessive_aspect_ratio"

    x, y, bw, bh = shape.bbox
    if (
        x < border_margin
        or y < border_margin
        or x + bw > w - border_margin
        or y + bh > h - border_margin
    ):
        return False, "near_image_border"

    return True, None


def _compute_confidence(shape: DetectedShape, fit_residual: float = 0.0) -> float:
    """Composite confidence score from shape metrics."""
    circ_score = min(shape.circularity, 1.0)
    solidity_score = min(shape.solidity, 1.0)
    residual_penalty = max(0.0, 1.0 - fit_residual / 10.0)
    return float(np.clip(0.4 * circ_score + 0.3 * solidity_score + 0.3 * residual_penalty, 0, 1))


def apply_deduction(
    shapes: list[DetectedShape],
    image_shape: tuple[int, int],
    config: dict,
    radius_results: dict[int, list[RadiusResult]] | None = None,
) -> list[FilteredDetection]:
    """Filter detections using rules and Isolation Forest anomaly detection."""
    cfg = config.get("deduction", {})
    confidence_threshold = cfg.get("confidence_threshold", 0.5)
    radius_results = radius_results or {}

    # Build feature vectors for ML filter
    features = []
    for shape in shapes:
        avg_residual = 0.0
        radii = radius_results.get(shape.shape_id, [])
        if radii:
            avg_residual = np.mean([r.fit_residual for r in radii])
        features.append([
            shape.area,
            shape.circularity,
            shape.eccentricity,
            shape.solidity,
            avg_residual,
        ])

    anomaly_flags = [False] * len(shapes)
    if len(features) >= 3:
        clf = IsolationForest(
            contamination=cfg.get("isolation_forest_contamination", 0.1),
            random_state=42,
        )
        predictions = clf.fit_predict(np.array(features))
        anomaly_flags = [p == -1 for p in predictions]

    results: list[FilteredDetection] = []
    for i, shape in enumerate(shapes):
        passed, reason = _rule_based_filter(shape, image_shape, config)
        radii = radius_results.get(shape.shape_id, [])
        avg_residual = np.mean([r.fit_residual for r in radii]) if radii else 0.0
        confidence = _compute_confidence(shape, avg_residual)

        if anomaly_flags[i]:
            passed = False
            reason = reason or "ml_anomaly"

        if confidence < confidence_threshold:
            passed = False
            reason = reason or "low_confidence"

        results.append(
            FilteredDetection(
                shape=shape,
                passed=passed,
                confidence_score=confidence,
                anomaly_flag=anomaly_flags[i],
                rejection_reason=None if passed else reason,
                radius_results=radii,
            )
        )

    return results
