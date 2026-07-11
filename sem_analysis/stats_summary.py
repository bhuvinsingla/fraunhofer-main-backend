"""Aggregation and confidence helpers for accepted-tip measurements."""

from __future__ import annotations

import math

import numpy as np


def summarize_values(values: list[float], *, headline: str = "median") -> dict:
    """
    Aggregate accepted-tip measurements.

    Headline is median (safer than mean for outlier circle fits).
    Also reports mean, std, IQR, min, max, and approximate 95% CI.
    """
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    n = len(clean)
    if n == 0:
        return {
            "n": 0,
            "median": None,
            "mean": None,
            "std": None,
            "iqr": None,
            "q25": None,
            "q75": None,
            "min": None,
            "max": None,
            "ci95": None,
            "headline": headline,
            "headline_value": None,
        }

    arr = np.asarray(clean, dtype=np.float64)
    q25, med, q75 = np.percentile(arr, [25, 50, 75])
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    # Normal approx 95% CI of the mean
    if n > 1:
        se = std / math.sqrt(n)
        ci = [mean - 1.96 * se, mean + 1.96 * se]
    else:
        ci = [mean, mean]

    headline_value = float(med) if headline == "median" else mean
    return {
        "n": n,
        "median": float(med),
        "mean": mean,
        "std": std,
        "iqr": [float(q25), float(q75)],
        "q25": float(q25),
        "q75": float(q75),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "ci95": [float(ci[0]), float(ci[1])],
        "headline": headline,
        "headline_value": headline_value,
    }


def tip_confidence(
    *,
    edge_score: float = 0.5,
    symmetry_score: float = 0.5,
    fit_score: float = 0.5,
    continuity_score: float = 0.5,
    consensus_score: float = 0.5,
    weights: dict | None = None,
) -> float:
    """
    C = w1·C_edge + w2·C_symmetry + w3·C_fit + w4·C_continuity + w5·C_consensus
    """
    w = weights or {
        "edge": 0.25,
        "symmetry": 0.20,
        "fit": 0.25,
        "continuity": 0.15,
        "consensus": 0.15,
    }
    score = (
        w.get("edge", 0.25) * float(np.clip(edge_score, 0, 1))
        + w.get("symmetry", 0.20) * float(np.clip(symmetry_score, 0, 1))
        + w.get("fit", 0.25) * float(np.clip(fit_score, 0, 1))
        + w.get("continuity", 0.15) * float(np.clip(continuity_score, 0, 1))
        + w.get("consensus", 0.15) * float(np.clip(consensus_score, 0, 1))
    )
    return float(np.clip(score, 0.0, 1.0))


def symmetry_from_branches(left_depth_nm: float, right_depth_nm: float) -> float:
    denom = max(left_depth_nm + right_depth_nm, 1e-6)
    return float(1.0 - abs(left_depth_nm - right_depth_nm) / denom)


def fit_score_from_residual(residual_px: float, scale_px: float = 5.0) -> float:
    return float(max(0.0, 1.0 - residual_px / max(scale_px, 1e-6)))
