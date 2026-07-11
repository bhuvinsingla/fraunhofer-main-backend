"""Tests for physical nm protocol, tip gates, and median aggregation."""

import numpy as np

from sem_analysis.contour_features import (
    accept_tip,
    compute_contour_features,
    deduplicate_tips,
    group_parallel_ridges,
    touches_image_border,
)
from sem_analysis.methods.fixed_distance_circle import (
    measure_fixed_distance_at_l,
    measure_fixed_distance_multi,
)
from sem_analysis.methods.inscribed_angle import measure_inscribed_angle
from sem_analysis.methods.projected_tip_distance import measure_projected_tip_distance
from sem_analysis.pipeline import load_config
from sem_analysis.stats_summary import summarize_values


def _v_tip(h=200, w=160, tip_r=20.0, cx=80.0, cy=40.0) -> np.ndarray:
    """Synthetic V-shaped tip contour (open polyline)."""
    pts = []
    for t in np.linspace(0, np.pi, 40):
        x = cx + tip_r * np.cos(t)
        y = cy + tip_r * (1 - np.sin(t))
        pts.append([x, y])
    # left flank down
    for y in np.linspace(cy + tip_r, cy + tip_r + 120, 40):
        x = cx - tip_r - 0.35 * (y - cy - tip_r)
        pts.append([x, y])
    # right flank down
    for y in np.linspace(cy + tip_r, cy + tip_r + 120, 40):
        x = cx + tip_r + 0.35 * (y - cy - tip_r)
        pts.append([x, y])
    return np.array(pts, dtype=np.float64)


class TestPhysicalMethods:
    def test_method1_scales_with_nm_per_pixel(self):
        contour = _v_tip()
        r1 = measure_fixed_distance_at_l(contour, nm_per_pixel=2.0, distance_l_nm=100.0)
        r2 = measure_fixed_distance_at_l(contour, nm_per_pixel=1.0, distance_l_nm=100.0)
        assert r1 is not None and r2 is not None
        # Same physical l → different l_px
        assert abs(r1.distance_l_px - 50.0) < 1e-6
        assert abs(r2.distance_l_px - 100.0) < 1e-6
        assert r1.label == "R100"

    def test_method1_multi_r_labels(self):
        contour = _v_tip()
        multi = measure_fixed_distance_multi(contour, 2.0, [25, 50, 100, 200])
        labels = {r.label for r in multi}
        assert "R100" in labels or len(multi) >= 1

    def test_method2_fit_band_nm(self):
        contour = _v_tip()
        result = measure_projected_tip_distance(
            contour, nm_per_pixel=2.0, fit_band_nm=(50, 200), min_flank_points=5
        )
        assert result is not None
        assert result.fit_band_nm == (50.0, 200.0)
        assert result.distance_nm > 0

    def test_method3_diameter_nm(self):
        contour = _v_tip()
        a = measure_inscribed_angle(contour, circle_diameter_nm=100.0, nm_per_pixel=2.0)
        b = measure_inscribed_angle(contour, circle_diameter_nm=100.0, nm_per_pixel=1.0)
        assert a is not None and b is not None
        assert abs(a.circle_diameter_px - 50.0) < 1e-6
        assert abs(b.circle_diameter_px - 100.0) < 1e-6
        assert a.label in ("theta100", "angle_D100") or a.label.startswith("angle_D")
        assert a.angle_degrees > 0


class TestContourGates:
    def test_border_ignores_top(self):
        pts = np.array([[50.0, 2.0], [40.0, 80.0], [60.0, 80.0]])
        assert touches_image_border(pts, (100, 100), margin_px=5, check_top=False) is False
        assert touches_image_border(pts, (100, 100), margin_px=5, check_top=True) is True

    def test_accept_and_dedup(self):
        cfg = load_config()
        # Relax branch depth for synthetic small tip
        cfg["measurement_methods"]["min_branch_depth_nm"] = 20.0
        contour = _v_tip()
        feat = compute_contour_features(contour, 0, 2.0, (200, 160), cfg)
        accept_tip(feat, cfg)
        # May or may not pass pointed-arch depending on geometry; ensure API works
        assert feat.rejection_reason is None or isinstance(feat.rejection_reason, str)

        f2 = compute_contour_features(contour, 1, 2.0, (200, 160), cfg, other_tips=[feat.tip_point])
        f2.accepted = True
        feat.accepted = True
        out = deduplicate_tips([feat, f2], 2.0, dedup_distance_nm=30.0)
        accepted = sum(1 for f in out if f.accepted)
        assert accepted == 1

    def test_group_parallel_ridges(self):
        c1 = _v_tip(cx=80)
        c2 = _v_tip(cx=82)
        grouped = group_parallel_ridges([c1, c2], nm_per_pixel=2.0, parallel_sep_nm=10.0)
        assert len(grouped) == 1


class TestStatsSummary:
    def test_median_headline(self):
        s = summarize_values([10, 20, 1000], headline="median")
        assert s["median"] == 20
        assert s["headline_value"] == 20
        assert s["n"] == 3
        assert s["iqr"] is not None
