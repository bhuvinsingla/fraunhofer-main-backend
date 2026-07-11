"""Geometric validation: compare fitted radius vs virtual-apex geometry."""

from __future__ import annotations

import numpy as np


def expected_radius_from_geometry(
    distance_l_px: float,
    included_angle_deg: float,
) -> float | None:
    """
    Expected osculating radius from virtual-apex wedge geometry.

    R ≈ l / (2·sin(α/2)) where l is distance from virtual apex to tip
    and α is the included angle between flank lines.
    """
    half = np.radians(included_angle_deg / 2.0)
    if half < 1e-6 or distance_l_px <= 0:
        return None
    return float(distance_l_px / (2.0 * np.sin(half)))


def validate_geometry(
    measured_radius_px: float,
    distance_l_px: float,
    included_angle_deg: float,
    tolerance_ratio: float = 0.25,
) -> tuple[bool, float | None, float | None]:
    """
    Validate fitted radius against geometric prediction.

    Returns (is_valid, expected_radius_px, relative_error).
    """
    expected = expected_radius_from_geometry(distance_l_px, included_angle_deg)
    if expected is None or measured_radius_px <= 0:
        return False, expected, None

    rel_error = abs(measured_radius_px - expected) / measured_radius_px
    return rel_error <= tolerance_ratio, expected, float(rel_error)


def composite_confidence(
    fit_residual_px: float,
    overlap_score: float,
    inlier_fraction: float,
    geometric_valid: bool,
    geometric_error: float | None,
    weights: dict | None = None,
) -> float:
    """Industrial-style composite confidence score in [0, 1]."""
    w = weights or {
        "residual": 0.30,
        "overlap": 0.30,
        "inliers": 0.20,
        "geometry": 0.20,
    }

    residual_score = max(0.0, 1.0 - fit_residual_px / 5.0)
    geo_score = 1.0 if geometric_valid else max(0.0, 1.0 - (geometric_error or 1.0))
    inlier_score = float(np.clip(inlier_fraction, 0.0, 1.0))

    score = (
        w["residual"] * residual_score
        + w["overlap"] * overlap_score
        + w["inliers"] * inlier_score
        + w["geometry"] * geo_score
    )
    return float(np.clip(score, 0.0, 1.0))
