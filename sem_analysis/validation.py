"""Validation against 3D topography ground-truth measurements."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy import stats

from sem_analysis.radius_computation import RadiusResult


def load_ground_truth(path: str | Path) -> pd.DataFrame:
    """Load ground-truth radii from CSV or JSON."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        return pd.DataFrame(data.get("measurements", [data]))
    return pd.read_csv(path)


def align_predictions(
    predictions: list[RadiusResult],
    ground_truth: pd.DataFrame,
    max_distance_px: float = 50.0,
) -> pd.DataFrame:
    """Match predicted peaks to ground-truth by nearest centroid."""
    if not predictions or ground_truth.empty:
        return pd.DataFrame()

    pred_centers = np.array([p.center for p in predictions])
    tree = cKDTree(pred_centers)

    gt_x_col = next((c for c in ("x", "centroid_x", "peak_x") if c in ground_truth.columns), None)
    gt_y_col = next((c for c in ("y", "centroid_y", "peak_y") if c in ground_truth.columns), None)
    gt_r_col = next(
        (c for c in ("radius_nm", "measured_radius_nm", "radius") if c in ground_truth.columns),
        None,
    )

    if gt_x_col is None or gt_y_col is None or gt_r_col is None:
        raise ValueError("Ground truth must contain x, y, and radius_nm columns")

    rows = []
    for _, gt_row in ground_truth.iterrows():
        query = np.array([gt_row[gt_x_col], gt_row[gt_y_col]])
        dist, idx = tree.query(query)
        if dist > max_distance_px:
            continue
        pred = predictions[idx]
        rows.append({
            "peak_id": pred.peak_id,
            "predicted_radius_nm": pred.radius_nm,
            "measured_radius_nm": float(gt_row[gt_r_col]),
            "error_nm": pred.radius_nm - float(gt_row[gt_r_col]),
            "relative_error_pct": (
                100.0 * (pred.radius_nm - float(gt_row[gt_r_col])) / float(gt_row[gt_r_col])
                if float(gt_row[gt_r_col]) != 0
                else np.nan
            ),
            "distance_px": float(dist),
        })

    return pd.DataFrame(rows)


def compute_error_metrics(comparison: pd.DataFrame) -> dict:
    """Compute RMSE, MAE, relative error, and Pearson R²."""
    if comparison.empty:
        return {"rmse_nm": None, "mae_nm": None, "mean_relative_error_pct": None, "r_squared": None}

    pred = comparison["predicted_radius_nm"].values
    measured = comparison["measured_radius_nm"].values
    errors = pred - measured

    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    mre = float(np.mean(np.abs(comparison["relative_error_pct"].dropna())))

    if len(pred) > 1:
        r_squared = float(stats.pearsonr(pred, measured)[0] ** 2)
    else:
        r_squared = None

    return {
        "rmse_nm": rmse,
        "mae_nm": mae,
        "mean_relative_error_pct": mre,
        "r_squared": r_squared,
        "n_matched": len(comparison),
    }


def generate_validation_report(
    comparison: pd.DataFrame,
    metrics: dict,
    output_dir: str | Path,
    image_name: str = "validation",
) -> dict:
    """Export CSV comparison and scatter plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{image_name}_validation.csv"
    comparison.to_csv(csv_path, index=False)

    plot_path = None
    if not comparison.empty:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(
            comparison["measured_radius_nm"],
            comparison["predicted_radius_nm"],
            alpha=0.7,
            edgecolors="k",
        )
        lims = [
            min(comparison["measured_radius_nm"].min(), comparison["predicted_radius_nm"].min()),
            max(comparison["measured_radius_nm"].max(), comparison["predicted_radius_nm"].max()),
        ]
        ax.plot(lims, lims, "k--", alpha=0.5, label="Identity")
        ax.set_xlabel("Measured Radius (nm)")
        ax.set_ylabel("Predicted Radius (nm)")
        ax.set_title(f"Validation (R²={metrics.get('r_squared', 'N/A'):.3f})"
                     if metrics.get("r_squared") else "Validation")
        ax.legend()
        ax.set_aspect("equal")
        plot_path = output_dir / f"{image_name}_validation_plot.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    return {
        "csv_path": str(csv_path),
        "plot_path": str(plot_path) if plot_path else None,
        "metrics": metrics,
    }
