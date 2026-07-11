"""Unit tests for SEM tip radius analysis pipeline."""

import json
from pathlib import Path

import numpy as np
import pytest

from sem_analysis.preprocessing import preprocess
from sem_analysis.radius_computation import (
    aggregate_radii,
    classify_tip_condition,
    compute_radius,
    taubin_circle_fit,
    TipCondition,
)
from sem_analysis.shape_detection import detect_shapes
from sem_analysis.pipeline import SEMAnalysisPipeline, load_config
from sem_analysis.utils.sample_generator import generate_synthetic_tip


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def sample_image(tmp_path):
    img_path = tmp_path / "test_tip.png"
    generate_synthetic_tip(img_path, tip_radius_px=20.0, nm_per_pixel=2.0)
    return img_path


class TestRadiusComputation:
    def test_taubin_fit_perfect_circle(self):
        theta = np.linspace(0, 2 * np.pi, 100)
        r_true = 25.0
        cx, cy = 100.0, 100.0
        points = np.column_stack([
            cx + r_true * np.cos(theta),
            cy + r_true * np.sin(theta),
        ])
        center, radius, residual = taubin_circle_fit(points)
        assert abs(radius - r_true) < 0.5
        assert abs(center[0] - cx) < 0.5
        assert abs(center[1] - cy) < 0.5
        assert residual < 0.1

    def test_aggregate_radii(self):
        from sem_analysis.radius_computation import RadiusResult

        results = [
            RadiusResult(0, 0, 10, 20, 200, 0.1, (50, 50), "taubin"),
            RadiusResult(1, 0, 12, 24, 240, 0.2, (50, 50), "taubin"),
        ]
        agg = aggregate_radii(results)
        assert agg["count"] == 2
        assert abs(agg["mean_radius_nm"] - 22.0) < 0.01

    def test_tip_classification(self):
        cfg = {"tip_classification": {"sharp_max_nm": 10, "moderate_max_nm": 50}}
        assert classify_tip_condition(5, cfg) == TipCondition.SHARP
        assert classify_tip_condition(30, cfg) == TipCondition.MODERATE
        assert classify_tip_condition(100, cfg) == TipCondition.BLUNT


class TestPreprocessing:
    def test_preprocess_output_range(self, config):
        image = np.random.rand(128, 128).astype(np.float32)
        result = preprocess(image, nm_per_pixel=1.0, config=config)
        assert result.data.dtype == np.float32
        assert result.data.min() >= 0.0
        assert result.data.max() <= 1.0


class TestShapeDetection:
    def test_detect_shapes_on_synthetic(self, config, sample_image):
        from sem_analysis.io.image_loader import load_image
        from sem_analysis.preprocessing import preprocess

        sem = load_image(sample_image)
        processed = preprocess(sem.data, sem.nm_per_pixel, config)
        shapes = detect_shapes(processed.data, config)
        assert len(shapes) >= 1


class TestPipeline:
    def test_end_to_end(self, sample_image, tmp_path):
        pipeline = SEMAnalysisPipeline()
        gt_path = sample_image.parent / f"{sample_image.stem}_ground_truth.csv"
        result = pipeline.analyze(
            sample_image,
            output_dir=tmp_path / "output",
            ground_truth_path=gt_path,
        )
        assert result.shapes_detected >= 1
        assert result.annotated_image_path is not None
        assert Path(result.annotated_image_path).exists()

        report_path = tmp_path / "output" / f"{sample_image.stem}_report.json"
        assert report_path.exists()
        with open(report_path) as f:
            report = json.load(f)
        assert "aggregation" in report
        assert "brainstorming_methods" in report
        assert "projected_tip_distance" in report["brainstorming_methods"]
        assert report["primary_method"] == "fixed_distance_circle"

        bs = report["brainstorming_methods"]
        assert "fixed_distance_circle" in bs
        assert "projected_tip_distance" in bs
        assert "inscribed_angle" in bs
        assert "per_curve" in bs["fixed_distance_circle"]

        method_paths = result.annotated_method_paths
        assert method_paths.get("method1")
        assert method_paths.get("method2")
        assert method_paths.get("method3")
        assert Path(method_paths["method1"]).exists()
        assert Path(method_paths["method2"]).exists()
        assert Path(method_paths["method3"]).exists()

        assert (tmp_path / "output" / f"{sample_image.stem}_tips.csv").exists()

    def test_brainstorming_methods(self, sample_image, tmp_path):
        pipeline = SEMAnalysisPipeline()
        result = pipeline.analyze(sample_image, output_dir=tmp_path / "output")
        bs = result.brainstorming_methods
        assert "projected_tip_distance" in bs
        ptd = bs["projected_tip_distance"]
        assert ptd.get("count", 0) >= 0
        if ptd.get("per_curve"):
            assert ptd["per_curve"][0].get("distance_l_nm", 0) > 0
