"""Tests for research-grade osculating circle metrology."""

import math

import numpy as np
import pytest

from sem_analysis.io.image_loader import apply_tilt_correction, tilt_scale_factor
from sem_analysis.pipeline import SEMAnalysisPipeline, load_config
from sem_analysis.research.curvature import compute_curvature_profile, find_curvature_tip
from sem_analysis.research.geometric_validation import expected_radius_from_geometry, validate_geometry
from sem_analysis.research.line_fitting import fit_line_ransac, line_intersection
from sem_analysis.research.osculating_tip import measure_osculating_tip
from sem_analysis.utils.sample_generator import generate_synthetic_tip


@pytest.fixture
def config():
    return load_config()


class TestTiltCorrection:
    def test_tilt_scale_factor_60_deg(self):
        assert abs(tilt_scale_factor(60.0) - 2.0) < 1e-9

    def test_apply_tilt_correction_doubles_scale(self, config):
        cfg = {
            **config,
            "calibration": {
                **config.get("calibration", {}),
                "tilt_correction": {"enabled": True, "tilt_angle_deg": 60.0},
            },
        }
        corrected, info = apply_tilt_correction(1.0, cfg)
        assert info["applied"] is True
        assert abs(corrected - 2.0) < 1e-9
        assert abs(info["scale_factor"] - 2.0) < 1e-9

    def test_tilt_disabled(self):
        cfg = {"calibration": {"tilt_correction": {"enabled": False, "tilt_angle_deg": 60}}}
        corrected, info = apply_tilt_correction(1.5, cfg)
        assert corrected == 1.5
        assert info["applied"] is False

    def test_tilt_default_without_config(self):
        """Blind 2× tilt correction is off by default (projected measurements only)."""
        corrected, info = apply_tilt_correction(1.0, {})
        assert info["applied"] is False
        assert corrected == 1.0
        assert abs(info["tilt_angle_deg"] - 60.0) < 1e-9


class TestResearchGrade:
    def test_ransac_line_fit(self):
        x = np.linspace(0, 50, 30)
        y = 0.5 * x + 10 + np.random.default_rng(0).normal(0, 0.3, len(x))
        pts = np.column_stack([x, y])
        result = fit_line_ransac(pts, threshold=2.0)
        assert result is not None
        m, b, inliers, _ = result
        assert abs(m - 0.5) < 0.15
        assert float(np.mean(inliers)) > 0.7

    def test_line_intersection(self):
        pt = line_intersection(1.0, 0.0, -1.0, 10.0)
        assert pt is not None
        assert abs(pt[0] - 5.0) < 0.01

    def test_curvature_on_arc(self):
        theta = np.linspace(-np.pi / 4, np.pi / 4, 50)
        r = 20.0
        pts = np.column_stack([r * np.sin(theta), r * (1 - np.cos(theta))])
        s, kappa, _ = compute_curvature_profile(pts)
        assert len(kappa) > 0
        assert np.max(kappa) > 0.01

    def test_geometric_validation(self):
        r_expected = expected_radius_from_geometry(30.0, 60.0)
        assert r_expected is not None
        valid, _, err = validate_geometry(r_expected, 30.0, 60.0, tolerance_ratio=0.1)
        assert valid
        assert err is not None
        assert err < 0.1

    def test_measure_osculating_tip_synthetic(self, config, tmp_path):
        img_path = tmp_path / "tip.png"
        generate_synthetic_tip(img_path, tip_radius_px=20.0, nm_per_pixel=2.0)
        from sem_analysis.io.image_loader import load_image
        from sem_analysis.preprocessing import preprocess
        from sem_analysis.edge_detection import detect_serration_peaks_global

        sem = load_image(img_path)
        processed = preprocess(sem.data, sem.nm_per_pixel, config)
        edge = detect_serration_peaks_global(processed.data, config)
        assert len(edge.peak_locations) >= 1

        peak = edge.peak_locations[0]
        result = measure_osculating_tip(
            processed.data,
            edge.edge_points,
            0,
            (float(peak[0]), float(peak[1])),
            processed.nm_per_pixel,
            config,
        )
        assert result is not None
        assert result.radius_px > 0
        assert result.confidence_score > 0
        assert result.included_angle_deg > 0

    def test_pipeline_research_output(self, tmp_path, config):
        img_path = tmp_path / "tip.png"
        generate_synthetic_tip(img_path, tip_radius_px=20.0, nm_per_pixel=2.0)
        cfg = {
            **config,
            "pipeline": {**config.get("pipeline", {}), "legacy_peak_detection": True},
            "research_grade": {**config.get("research_grade", {}), "enabled": True},
        }
        pipeline = SEMAnalysisPipeline(config=cfg)
        result = pipeline.analyze(img_path, output_dir=tmp_path / "out")
        assert "summary" in result.research_grade
        assert result.annotated_research_path is not None
        assert (tmp_path / "out" / f"{img_path.stem}_research.png").exists()
        assert (tmp_path / "out" / f"{img_path.stem}_research_radii.csv").exists()
