"""Tests for brainstorming measurement methods."""

import numpy as np
import pytest

from sem_analysis.methods.brainstorming import (
    run_brainstorming_all_peaks,
    run_brainstorming_methods,
    run_brainstorming_per_peak,
)
from sem_analysis.methods.fixed_distance_circle import measure_fixed_distance_circle
from sem_analysis.methods.inscribed_angle import measure_inscribed_angle
from sem_analysis.methods.local_contour import extract_local_contour
from sem_analysis.methods.projected_tip_distance import measure_projected_tip_distance
from sem_analysis.pipeline import load_config
from sem_analysis.utils.sample_generator import generate_synthetic_tip


def _v_contour(cx=80.0, cy=40.0, tip_r=20.0, flank=140.0) -> np.ndarray:
    pts = []
    for t in np.linspace(0, np.pi, 48):
        pts.append([cx + tip_r * np.cos(t), cy + tip_r * (1 - np.sin(t))])
    for y in np.linspace(cy + tip_r, cy + tip_r + flank, 50):
        pts.append([cx - tip_r - 0.4 * (y - cy - tip_r), y])
    for y in np.linspace(cy + tip_r, cy + tip_r + flank, 50):
        pts.append([cx + tip_r + 0.4 * (y - cy - tip_r), y])
    return np.array(pts, dtype=np.float64)


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def synthetic_contour(tmp_path, config):
    img_path = tmp_path / "tip.png"
    generate_synthetic_tip(img_path, tip_radius_px=20.0, nm_per_pixel=2.0)
    from sem_analysis.io.image_loader import load_image
    from sem_analysis.preprocessing import preprocess
    from sem_analysis.shape_detection import detect_shapes

    sem = load_image(img_path)
    processed = preprocess(sem.data, sem.nm_per_pixel, config)
    shapes = detect_shapes(processed.data, config)
    assert len(shapes) >= 1
    return processed.data, shapes[0].contour, processed.nm_per_pixel


@pytest.fixture
def v_tip():
    """Deep V polyline suitable for 50–200 nm physical windows at 2 nm/px."""
    return np.zeros((220, 160), dtype=np.float32), _v_contour(), 2.0


class TestBrainstormingMethods:
    def test_method1_fixed_distance_circle(self, v_tip, config):
        image, contour, nm_per_pixel = v_tip
        result = measure_fixed_distance_circle(
            image, contour, nm_per_pixel, distance_l_nm=100.0
        )
        assert result is not None
        assert result.radius_px > 0
        assert result.radius_nm > 0
        assert abs(result.distance_l_nm - 100.0) < 1e-6
        assert len(result.scan_line) == 4

    def test_method2_projected_tip_distance(self, v_tip):
        _, contour, nm_per_pixel = v_tip
        result = measure_projected_tip_distance(
            contour, nm_per_pixel, fit_band_nm=(50, 200), min_flank_points=5
        )
        assert result is not None
        assert result.distance_px > 0
        assert result.distance_nm > 0
        assert len(result.left_line) == 4
        assert len(result.right_line) == 4
        assert len(result.vertical_l_line) == 4

    def test_method3_inscribed_angle(self, v_tip):
        _, contour, nm_per_pixel = v_tip
        result = measure_inscribed_angle(
            contour, circle_diameter_nm=100.0, nm_per_pixel=nm_per_pixel
        )
        assert result is not None
        assert result.angle_degrees > 0
        assert result.circle_diameter_nm == 100.0
        assert len(result.left_tangent_line) == 4
        assert len(result.right_tangent_line) == 4

    def test_run_all_brainstorming_methods(self, v_tip, config):
        image, contour, nm_per_pixel = v_tip
        serialized, raw = run_brainstorming_methods(image, contour, nm_per_pixel, config)
        assert "projected_tip_distance" in serialized
        assert "fixed_distance_circle" in serialized
        assert serialized["fixed_distance_circle"].get("radius_nm", 0) > 0 or "radii_by_l" in serialized["fixed_distance_circle"]
        assert raw["projected_tip_distance"].distance_nm > 0

    def test_local_contour_extraction(self, v_tip):
        _, contour, _ = v_tip
        tip = contour[np.argmin(contour[:, 1])]
        local = extract_local_contour(contour, tip, window_y_px=80, window_x_px=40)
        assert local is not None
        assert len(local.reshape(-1, 2)) >= 12

    def test_run_brainstorming_per_peak(self, v_tip, config):
        image, contour, nm_per_pixel = v_tip
        tip = contour[np.argmin(contour[:, 1])]
        serialized, raw = run_brainstorming_per_peak(
            image, contour, 0, (float(tip[0]), float(tip[1])), nm_per_pixel, config
        )
        assert "peak_id" in serialized
        assert "projected_tip_distance" in serialized or "fixed_distance_circle" in serialized

    def test_run_brainstorming_all_peaks(self, v_tip, config):
        image, contour, nm_per_pixel = v_tip
        config = dict(config)
        config["measurement_methods"] = {
            **config.get("measurement_methods", {}),
            "min_branch_depth_nm": 20.0,
            "dedup_distance_nm": 5.0,
        }
        tip = contour[np.argmin(contour[:, 1])]
        peaks = np.array([[tip[0], tip[1]]])
        serialized, _ = run_brainstorming_all_peaks(
            image, peaks, contour, nm_per_pixel, config
        )
        assert "fixed_distance_circle" in serialized
        assert "projected_tip_distance" in serialized
        assert "inscribed_angle" in serialized
        assert "per_curve" in serialized["fixed_distance_circle"]
        assert "median" in serialized["fixed_distance_circle"]
        assert "protocol" in serialized
        assert serialized["fixed_distance_circle"].get("headline") == "median"
