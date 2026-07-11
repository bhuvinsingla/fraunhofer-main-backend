"""Generate synthetic SEM-like tip images for testing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def generate_synthetic_tip(
    output_path: str | Path,
    width: int = 512,
    height: int = 512,
    tip_radius_px: float = 15.0,
    nm_per_pixel: float = 2.0,
    noise_level: float = 0.05,
) -> dict:
    """Create a synthetic metal tip SEM image with known radius."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = np.zeros((height, width), dtype=np.float32)
    cx, cy = width // 2, height // 4

    y_grid, x_grid = np.ogrid[:height, :width]
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)

    # Tip arc (upper semicircle)
    arc_mask = (dist <= tip_radius_px * 3) & (y_grid >= cy)
    img[arc_mask] = 0.8

    # Blade body
    blade_width = tip_radius_px * 4
    img[cy:, cx - int(blade_width) : cx + int(blade_width)] = 0.7

    # Tip curvature
    tip_mask = dist <= tip_radius_px
    img[tip_mask] = 1.0

    # Add SEM-like noise
    img += np.random.normal(0, noise_level, img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)

    uint8 = (img * 255).astype(np.uint8)
    Image.fromarray(uint8).save(output_path)

    # Save calibration companion
    cal_path = output_path.parent / f"{output_path.stem}_calibration.json"
    cal_data = {
        "nm_per_pixel": nm_per_pixel,
        "true_radius_nm": tip_radius_px * nm_per_pixel,
        "tip_center": [cx, cy],
    }
    with open(cal_path, "w") as f:
        json.dump(cal_data, f, indent=2)

    # Ground truth for validation
    gt_path = output_path.parent / f"{output_path.stem}_ground_truth.csv"
    with open(gt_path, "w") as f:
        f.write("x,y,radius_nm\n")
        f.write(f"{cx},{cy},{tip_radius_px * nm_per_pixel}\n")

    return {
        "image_path": str(output_path),
        "calibration_path": str(cal_path),
        "ground_truth_path": str(gt_path),
        "true_radius_nm": tip_radius_px * nm_per_pixel,
    }


if __name__ == "__main__":
    info = generate_synthetic_tip("data/sample/synthetic_tip.png")
    print(f"Generated: {info}")
