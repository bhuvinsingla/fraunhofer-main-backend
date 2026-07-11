"""Complete-arch tip detection: contours → branches → one apex per arch."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from sem_analysis.edge_geometry import smooth_branch
from sem_analysis.edge_probability import EdgeProbabilityMaps, preprocess_sem
from sem_analysis.roi import has_full_measurement_window


@dataclass
class ValidatedArch:
    """One physical tip from a complete arch."""

    tip_id: int
    apex_x_px: float
    apex_y_px: float
    contour: np.ndarray
    left_raw: np.ndarray
    right_raw: np.ndarray
    left_smooth: np.ndarray | None
    right_smooth: np.ndarray | None
    fit_residual_px: float
    edge_score: float
    border_valid: bool
    left_branch_valid: bool
    right_branch_valid: bool
    window_valid: bool
    valid: bool
    rejection_reason: str | None = None
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("contour", None)
        d.pop("left_raw", None)
        d.pop("right_raw", None)
        d.pop("left_smooth", None)
        d.pop("right_smooth", None)
        return d


def validate_arch(
    contour: np.ndarray,
    image_shape: tuple[int, int],
    min_branch_points: int = 20,
    border_px: int = 10,
) -> bool:
    points = contour.reshape(-1, 2)
    height, width = image_shape
    if len(points) < 2 * min_branch_points:
        return False
    x = points[:, 0]
    y = points[:, 1]
    if (
        np.any(x <= border_px)
        or np.any(x >= width - border_px - 1)
        or np.any(y <= border_px)
        or np.any(y >= height - border_px - 1)
    ):
        return False
    apex_index = int(np.argmin(y))
    apex_x = x[apex_index]
    left_count = int(np.sum(x < apex_x))
    right_count = int(np.sum(x > apex_x))
    return left_count >= min_branch_points and right_count >= min_branch_points


def _branch_depth_ok(
    left: np.ndarray,
    right: np.ndarray,
    apex: np.ndarray,
    min_depth_px: float,
) -> tuple[bool, bool]:
    left_ok = len(left) > 0 and float(left[:, 1].max() - apex[1]) >= min_depth_px
    right_ok = len(right) > 0 and float(right[:, 1].max() - apex[1]) >= min_depth_px
    return left_ok, right_ok


def _edge_score(contour: np.ndarray, gradient: np.ndarray) -> float:
    pts = contour.reshape(-1, 2).astype(int)
    h, w = gradient.shape[:2]
    vals = []
    for x, y in pts:
        if 0 <= x < w and 0 <= y < h:
            vals.append(float(gradient[y, x]))
    if not vals:
        return 0.0
    return float(np.clip(np.mean(vals) / 255.0, 0, 1))


def detect_validated_arches(
    roi_image: np.ndarray,
    nm_per_px: float,
    config: dict,
    edge_maps: EdgeProbabilityMaps | None = None,
) -> list[ValidatedArch]:
    """
    Edge pixels → connected curves → L/R branches → one complete arch → one apex.

    Rejects border-touching / incomplete measurement windows before any circle fit.
    """
    tip_cfg = config.get("tip_validation", {})
    method_cfg = config.get("measurement_methods", {})
    proto = config.get("protocol", {})

    min_branch_points = int(tip_cfg.get("min_branch_points", 20))
    border_px = int(tip_cfg.get("border_px", 10))
    min_branch_depth_nm = float(method_cfg.get("min_branch_depth_nm", 50.0))
    min_depth_px = min_branch_depth_nm / max(nm_per_px, 1e-9)
    # Window for largest requested Method-1 distance
    distances = list(proto.get("method1_distances_nm") or method_cfg.get("fixed_distance_circle", {}).get("distances_nm") or [200])
    max_l_nm = float(max(distances))
    window_radius_px = max_l_nm / max(nm_per_px, 1e-9)
    min_spacing_nm = float(tip_cfg.get("minimum_tip_spacing_nm", 40.0))
    min_spacing_px = min_spacing_nm / max(nm_per_px, 1e-9)
    max_residual_px = float(tip_cfg.get("max_fit_residual_px", 2.0))
    min_contour_len = int(tip_cfg.get("min_contour_length_px", 40))

    if edge_maps is None:
        edge_maps = preprocess_sem(roi_image)

    h, w = edge_maps.canny.shape[:2]
    # Morph close lightly to join broken arch edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edge_maps.canny, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    candidates: list[ValidatedArch] = []

    for i, cnt in enumerate(contours):
        pts = cnt.reshape(-1, 2).astype(np.float64)
        if len(pts) < min_contour_len:
            continue

        if not validate_arch(cnt, (h, w), min_branch_points=min_branch_points, border_px=border_px):
            continue

        apex_idx = int(np.argmin(pts[:, 1]))
        apex = pts[apex_idx]
        ax, ay = float(apex[0]), float(apex[1])

        window_ok = has_full_measurement_window(ax, ay, w, h, window_radius_px, safety_px=5)
        if not window_ok:
            # Still record as rejected for diagnostics
            candidates.append(
                ValidatedArch(
                    tip_id=-1,
                    apex_x_px=ax,
                    apex_y_px=ay,
                    contour=pts,
                    left_raw=np.empty((0, 2)),
                    right_raw=np.empty((0, 2)),
                    left_smooth=None,
                    right_smooth=None,
                    fit_residual_px=0.0,
                    edge_score=_edge_score(cnt, edge_maps.gradient),
                    border_valid=False,
                    left_branch_valid=False,
                    right_branch_valid=False,
                    window_valid=False,
                    valid=False,
                    rejection_reason="incomplete_measurement_window",
                )
            )
            continue

        left = pts[pts[:, 0] < ax]
        right = pts[pts[:, 0] > ax]
        left_ok, right_ok = _branch_depth_ok(left, right, apex, min_depth_px)
        if not left_ok or not right_ok:
            candidates.append(
                ValidatedArch(
                    tip_id=-1,
                    apex_x_px=ax,
                    apex_y_px=ay,
                    contour=pts,
                    left_raw=left,
                    right_raw=right,
                    left_smooth=None,
                    right_smooth=None,
                    fit_residual_px=0.0,
                    edge_score=_edge_score(cnt, edge_maps.gradient),
                    border_valid=True,
                    left_branch_valid=left_ok,
                    right_branch_valid=right_ok,
                    window_valid=True,
                    valid=False,
                    rejection_reason="insufficient_branch_depth",
                )
            )
            continue

        left_fit = smooth_branch(left, apex, max_residual_px=max_residual_px)
        right_fit = smooth_branch(right, apex, max_residual_px=max_residual_px)
        if left_fit is None or right_fit is None:
            candidates.append(
                ValidatedArch(
                    tip_id=-1,
                    apex_x_px=ax,
                    apex_y_px=ay,
                    contour=pts,
                    left_raw=left,
                    right_raw=right,
                    left_smooth=None,
                    right_smooth=None,
                    fit_residual_px=99.0,
                    edge_score=_edge_score(cnt, edge_maps.gradient),
                    border_valid=True,
                    left_branch_valid=True,
                    right_branch_valid=True,
                    window_valid=True,
                    valid=False,
                    rejection_reason="smooth_residual_too_high",
                )
            )
            continue

        residual = 0.5 * (left_fit.residual_px + right_fit.residual_px)
        score = _edge_score(cnt, edge_maps.gradient)
        candidates.append(
            ValidatedArch(
                tip_id=-1,
                apex_x_px=ax,
                apex_y_px=ay,
                contour=pts,
                left_raw=left,
                right_raw=right,
                left_smooth=left_fit.smooth_points,
                right_smooth=right_fit.smooth_points,
                fit_residual_px=residual,
                edge_score=score,
                border_valid=True,
                left_branch_valid=True,
                right_branch_valid=True,
                window_valid=True,
                valid=True,
                rejection_reason=None,
            )
        )

    # NMS: one tip per physical arch
    accepted = [c for c in candidates if c.valid]
    accepted.sort(key=lambda c: (c.edge_score, -c.fit_residual_px), reverse=True)
    kept: list[ValidatedArch] = []
    for c in accepted:
        dup = False
        for k in kept:
            dist = np.hypot(c.apex_x_px - k.apex_x_px, c.apex_y_px - k.apex_y_px)
            if dist < min_spacing_px:
                dup = True
                break
        if dup:
            c.valid = False
            c.rejection_reason = "duplicate_arch"
            c.confidence = 0.0
        else:
            kept.append(c)

    # Assign tip IDs to kept only; rejected keep tip_id=-1
    for i, c in enumerate(kept):
        c.tip_id = i

    # Return kept + rejected diagnostics
    rejected = [c for c in candidates if not c.valid]
    return kept + rejected
