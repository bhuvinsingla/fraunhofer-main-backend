"""Tests for ROI exclusion, measurement window, and corrected geometry helpers."""

import numpy as np

from sem_analysis.blade_value import flank_included_angle_deg
from sem_analysis.edge_geometry import circumcircle_radius, fit_line_tls, intersect_lines, stability_ratio
from sem_analysis.methods.fixed_distance_circle import measure_fixed_distance_at_l, measure_method1_at_l
from sem_analysis.methods.inscribed_angle import measure_inscribed_angle
from sem_analysis.methods.projected_tip_distance import measure_projected_tip_distance
from sem_analysis.roi import detect_footer_row, extract_measurement_roi, has_full_measurement_window


def _v_contour(cx=80.0, cy=80.0, tip_r=20.0, flank=140.0) -> np.ndarray:
    pts = []
    for t in np.linspace(0, np.pi, 48):
        pts.append([cx + tip_r * np.cos(t), cy + tip_r * (1 - np.sin(t))])
    for y in np.linspace(cy + tip_r, cy + tip_r + flank, 50):
        pts.append([cx - tip_r - 0.4 * (y - cy - tip_r), y])
    for y in np.linspace(cy + tip_r, cy + tip_r + flank, 50):
        pts.append([cx + tip_r + 0.4 * (y - cy - tip_r), y])
    return np.array(pts, dtype=np.float64)


class TestROI:
    def test_footer_detection(self):
        img = np.full((600, 400), 180, dtype=np.uint8)
        img[540:, :] = 10  # dark footer
        row = detect_footer_row(img)
        assert 520 <= row <= 560

    def test_border_margin(self):
        img = np.full((200, 200), 128, dtype=np.uint8)
        roi = extract_measurement_roi(img, border_margin_px=10, footer_row=200)
        assert roi.height == 180
        assert roi.width == 180
        assert roi.offset_x == 10

    def test_reject_peak_y_zero(self):
        assert has_full_measurement_window(50, 0, 200, 200, radius_px=100) is False
        assert has_full_measurement_window(100, 100, 200, 200, radius_px=50) is True


class TestGeometry:
    def test_smooth_branch_rejects_short(self):
        from sem_analysis.edge_geometry import smooth_branch

        apex = np.array([50.0, 50.0])
        pts = np.array([[48.0, 51.0], [52.0, 51.0], [50.0, 52.0]])
        assert smooth_branch(pts, apex) is None

    def test_circumcircle(self):
        # Unit circle through (1,0), (0,1), (-1,0) → R=1
        r = circumcircle_radius([1, 0], [0, 1], [-1, 0])
        assert abs(r - 1.0) < 1e-6

    def test_tls_parallel_reject(self):
        c1, d1 = fit_line_tls(np.array([[0, 0], [0, 1], [0, 2]], float))
        c2, d2 = fit_line_tls(np.array([[1, 0], [1, 1], [1, 2]], float))
        try:
            intersect_lines(c1, d1, c2, d2, min_cross=0.15)
            assert False, "expected parallel reject"
        except ValueError:
            pass

    def test_stability_ratio(self):
        assert stability_ratio([10, 10, 10]) == 0.0
        assert stability_ratio([10, 12, 14]) > 0.2

    def test_flank_included_angle(self):
        # 90° between (1,0) and (0,1) after Y-orient → both point +Y-ish
        a = flank_included_angle_deg([1, 1], [-1, 1])
        assert 80 < a < 100

    def test_whiteboard_geometry(self):
        from sem_analysis.whiteboard_geometry import build_whiteboard_geometry

        c = _v_contour(cx=100, cy=60, tip_r=25, flank=160)
        apex = c[np.argmin(c[:, 1])]
        left = c[c[:, 0] < apex[0]]
        right = c[c[:, 0] > apex[0]]
        g = build_whiteboard_geometry(
            tip_id=0,
            apex=apex,
            left=left,
            right=right,
            nm_per_px=2.0,
            fit_band_nm=(50, 200),
        )
        assert g is not None and g.valid
        assert g.included_angle_deg > 1.0
        assert g.circle_radius_px > 0
        assert g.d_px > 0
        # Projected tip should be above ultimate tip (smaller y)
        assert g.projected_tip[1] <= g.ultimate_tip[1] + 5


class TestMethods:
    def test_method2_includes_angle_and_area(self):
        c = _v_contour()
        apex = c[np.argmin(c[:, 1])]
        left = c[c[:, 0] < apex[0]]
        right = c[c[:, 0] > apex[0]]
        r = measure_projected_tip_distance(
            nm_per_pixel=2.0,
            fit_band_nm=(50, 200),
            apex=apex,
            left=left,
            right=right,
            min_flank_points=5,
        )
        assert r is not None and r.valid
        assert r.included_angle_deg is not None and r.included_angle_deg > 1.0
        assert r.area_under_curve_nm2 is not None and r.area_under_curve_nm2 >= 0.0

    def test_method1_physical_l(self):
        c = _v_contour()
        r = measure_fixed_distance_at_l(c, nm_per_pixel=2.0, distance_l_nm=100.0)
        assert r is not None and r.valid
        assert abs(r.distance_l_px - 50.0) < 1e-6
        assert r.projected_radius_nm is not None

    def test_method2_tls(self):
        c = _v_contour()
        apex = c[np.argmin(c[:, 1])]
        left = c[c[:, 0] < apex[0]]
        right = c[c[:, 0] > apex[0]]
        r = measure_projected_tip_distance(
            nm_per_pixel=2.0,
            fit_band_nm=(50, 200),
            apex=apex,
            left=left,
            right=right,
            min_flank_points=5,
        )
        assert r is not None
        assert r.valid
        assert r.distance_nm is not None and r.distance_nm > 0

    def test_method3_interpretation_a(self):
        c = _v_contour()
        apex = c[np.argmin(c[:, 1])]
        left = c[c[:, 0] < apex[0]]
        right = c[c[:, 0] > apex[0]]
        r = measure_inscribed_angle(
            circle_diameter_nm=100.0,
            nm_per_pixel=2.0,
            apex=apex,
            left=left,
            right=right,
        )
        assert r is not None and r.valid
        assert r.definition.startswith("interpretation_A")
        assert r.angle_degrees > 1.0
        assert abs(r.circle_center[0] - apex[0]) < 1e-6
        assert abs(r.circle_center[1] - apex[1]) < 1e-6
