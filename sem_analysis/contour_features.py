"""Per-contour geometric descriptors and tip acceptance gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from sem_analysis.research.curvature import compute_curvature_profile


@dataclass
class ContourFeatures:
    """Geometric descriptors for one contour / local tip curve."""

    contour_id: int
    tip_point: tuple[float, float]
    length_px: float
    length_nm: float
    orientation_deg: float
    kappa_max: float
    kappa_at_apex: float
    nn_distance_nm: float | None
    is_pointed_arch: bool
    pointed_arch_score: float
    touches_border: bool
    has_left_branch: bool
    has_right_branch: bool
    left_branch_depth_nm: float
    right_branch_depth_nm: float
    surface_class: str  # outer | internal_ridge
    accepted: bool = False
    rejection_reason: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tip_point"] = [float(self.tip_point[0]), float(self.tip_point[1])]
        return d


def _as_points(contour: np.ndarray) -> np.ndarray:
    pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    return pts


def contour_length_px(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def contour_orientation_deg(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    centered = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    angle = float(np.degrees(np.arctan2(vt[0, 1], vt[0, 0])))
    return angle


def find_apex(pts: np.ndarray) -> tuple[float, float]:
    idx = int(np.argmin(pts[:, 1]))
    return float(pts[idx, 0]), float(pts[idx, 1])


def branch_depths_nm(
    pts: np.ndarray,
    tip: tuple[float, float],
    nm_per_pixel: float,
) -> tuple[float, float, bool, bool]:
    """Max depth (nm) of points left/right of tip below the apex."""
    tx, ty = tip
    left = pts[pts[:, 0] < tx]
    right = pts[pts[:, 0] > tx]
    left_depth_px = float(left[:, 1].max() - ty) if len(left) else 0.0
    right_depth_px = float(right[:, 1].max() - ty) if len(right) else 0.0
    left_depth_nm = max(0.0, left_depth_px * nm_per_pixel)
    right_depth_nm = max(0.0, right_depth_px * nm_per_pixel)
    return left_depth_nm, right_depth_nm, len(left) >= 3, len(right) >= 3


def touches_image_border(
    pts: np.ndarray,
    image_shape: tuple[int, ...],
    margin_px: float = 10.0,
    *,
    check_top: bool = True,
) -> bool:
    """True if contour contacts any image border within margin_px."""
    h, w = int(image_shape[0]), int(image_shape[1])
    hit = (
        np.any(pts[:, 0] <= margin_px)
        or np.any(pts[:, 0] >= w - 1 - margin_px)
        or np.any(pts[:, 1] >= h - 1 - margin_px)
    )
    if check_top:
        hit = hit or np.any(pts[:, 1] <= margin_px)
    return bool(hit)


def pointed_arch_score(pts: np.ndarray, tip: tuple[float, float]) -> tuple[bool, float]:
    """Score whether contour forms a pointed arch with diverging flanks."""
    tx, ty = tip
    left = pts[pts[:, 0] < tx]
    right = pts[pts[:, 0] > tx]
    if len(left) < 3 or len(right) < 3:
        return False, 0.0

    left_below = left[left[:, 1] > ty]
    right_below = right[right[:, 1] > ty]
    if len(left_below) < 2 or len(right_below) < 2:
        return False, 0.0

    # Width should increase with depth below tip
    depths = np.linspace(ty + 2, min(left_below[:, 1].max(), right_below[:, 1].max()), 5)
    widths = []
    for y in depths:
        lx = left_below[np.argmin(np.abs(left_below[:, 1] - y)), 0]
        rx = right_below[np.argmin(np.abs(right_below[:, 1] - y)), 0]
        widths.append(rx - lx)
    if len(widths) < 2:
        return False, 0.0
    growth = float(widths[-1] - widths[0])
    score = float(np.clip(growth / max(abs(widths[0]), 1.0), 0.0, 1.0))
    return growth > 0, score


def surface_class_for_tip(
    tip: tuple[float, float],
    all_tips: list[tuple[float, float]],
    image_shape: tuple[int, ...],
) -> str:
    """Outer tips lie near the leftmost/rightmost envelope; others are internal ridges."""
    if not all_tips:
        return "outer"
    xs = np.array([t[0] for t in all_tips], dtype=np.float64)
    w = float(image_shape[1])
    # Tips near global min/max x among candidates → outer; central → internal
    span = float(xs.max() - xs.min()) if len(xs) > 1 else w
    if span < 1e-6:
        return "outer"
    rel = abs(tip[0] - float(np.median(xs))) / span
    return "outer" if rel >= 0.25 else "internal_ridge"


def nearest_contour_distance_nm(
    tip: tuple[float, float],
    other_tips: list[tuple[float, float]],
    nm_per_pixel: float,
) -> float | None:
    if not other_tips:
        return None
    dists = [np.hypot(tip[0] - ox, tip[1] - oy) for ox, oy in other_tips]
    return float(min(dists) * nm_per_pixel)


def compute_contour_features(
    contour: np.ndarray,
    contour_id: int,
    nm_per_pixel: float,
    image_shape: tuple[int, ...],
    config: dict,
    other_tips: list[tuple[float, float]] | None = None,
    all_tips: list[tuple[float, float]] | None = None,
) -> ContourFeatures:
    """Compute full geometric descriptor set for one contour."""
    method_cfg = config.get("measurement_methods", {})
    border_margin = float(config.get("deduction", {}).get("border_margin_px", 10))
    min_branch_nm = float(method_cfg.get("min_branch_depth_nm", 50.0))
    smooth = int(config.get("research_grade", {}).get("curvature_smooth_window", 5))

    pts = _as_points(contour)
    tip = find_apex(pts)
    length_px = contour_length_px(pts)
    orientation = contour_orientation_deg(pts)

    kappa_max = 0.0
    kappa_apex = 0.0
    try:
        s, kappa, ordered = compute_curvature_profile(pts, smooth_window=smooth)
        if len(kappa):
            kappa_max = float(np.nanmax(np.abs(kappa)))
            apex_idx = int(np.argmin(ordered[:, 1]))
            kappa_apex = float(abs(kappa[min(apex_idx, len(kappa) - 1)]))
    except Exception:
        pass

    left_nm, right_nm, has_left, has_right = branch_depths_nm(pts, tip, nm_per_pixel)
    # Strengthen branch flags with depth requirement
    has_left = has_left and left_nm >= min_branch_nm * 0.5
    has_right = has_right and right_nm >= min_branch_nm * 0.5

    pointed, pscore = pointed_arch_score(pts, tip)
    border = touches_image_border(pts, image_shape, border_margin)
    tips_for_class = all_tips or ([tip] + (other_tips or []))
    sclass = surface_class_for_tip(tip, tips_for_class, image_shape)
    nn = nearest_contour_distance_nm(tip, other_tips or [], nm_per_pixel)

    return ContourFeatures(
        contour_id=contour_id,
        tip_point=tip,
        length_px=length_px,
        length_nm=length_px * nm_per_pixel,
        orientation_deg=orientation,
        kappa_max=kappa_max,
        kappa_at_apex=kappa_apex,
        nn_distance_nm=nn,
        is_pointed_arch=pointed,
        pointed_arch_score=pscore,
        touches_border=border,
        has_left_branch=has_left,
        has_right_branch=has_right,
        left_branch_depth_nm=left_nm,
        right_branch_depth_nm=right_nm,
        surface_class=sclass,
    )


def accept_tip(
    features: ContourFeatures,
    config: dict,
    *,
    require_outer: bool = False,
    fit_stable: bool = True,
) -> ContourFeatures:
    """Apply hard tip-acceptance gates; mutates accepted / rejection_reason."""
    method_cfg = config.get("measurement_methods", {})
    min_branch_nm = float(method_cfg.get("min_branch_depth_nm", 50.0))

    if features.touches_border:
        features.accepted = False
        features.rejection_reason = "touches_border"
        return features
    if not features.has_left_branch or not features.has_right_branch:
        features.accepted = False
        features.rejection_reason = "missing_branch"
        return features
    if features.left_branch_depth_nm < min_branch_nm or features.right_branch_depth_nm < min_branch_nm:
        features.accepted = False
        features.rejection_reason = "insufficient_branch_depth"
        return features
    if not features.is_pointed_arch and features.kappa_at_apex <= 0 and features.kappa_max <= 0:
        # Still allow if both flanks are deep enough (skyline-style apex)
        if features.left_branch_depth_nm < min_branch_nm or features.right_branch_depth_nm < min_branch_nm:
            features.accepted = False
            features.rejection_reason = "no_local_apex"
            return features
    if require_outer and features.surface_class != "outer":
        features.accepted = False
        features.rejection_reason = "internal_ridge"
        return features
    if not fit_stable:
        features.accepted = False
        features.rejection_reason = "unstable_fit"
        return features

    features.accepted = True
    features.rejection_reason = None
    return features


def deduplicate_tips(
    features_list: list[ContourFeatures],
    nm_per_pixel: float,
    dedup_distance_nm: float = 30.0,
) -> list[ContourFeatures]:
    """Keep highest-scoring tip within dedup distance; mark others rejected."""
    if not features_list:
        return features_list

    dedup_px = dedup_distance_nm / max(nm_per_pixel, 1e-9)
    # Prefer accepted pointed arches with higher kappa
    order = sorted(
        range(len(features_list)),
        key=lambda i: (
            features_list[i].accepted,
            features_list[i].is_pointed_arch,
            features_list[i].kappa_at_apex,
            features_list[i].pointed_arch_score,
        ),
        reverse=True,
    )
    kept_idx: list[int] = []
    for i in order:
        f = features_list[i]
        if not f.accepted:
            continue
        duplicate = False
        for j in kept_idx:
            oj = features_list[j]
            dist = np.hypot(f.tip_point[0] - oj.tip_point[0], f.tip_point[1] - oj.tip_point[1])
            if dist < dedup_px:
                duplicate = True
                break
        if duplicate:
            f.accepted = False
            f.rejection_reason = "duplicate_tip"
        else:
            kept_idx.append(i)
    return features_list


def group_parallel_ridges(
    contours: list[np.ndarray],
    nm_per_pixel: float,
    parallel_sep_nm: float = 10.0,
) -> list[np.ndarray]:
    """
    Group nearby parallel contours; keep the outermost (extreme mean-x) per cluster.
    Simple nearest-neighbour clustering on tip positions.
    """
    if not contours:
        return []
    tips = [find_apex(_as_points(c)) for c in contours]
    eps_px = parallel_sep_nm / max(nm_per_pixel, 1e-9)

    n = len(contours)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(tips[i][0] - tips[j][0], tips[i][1] - tips[j][1])
            if d <= eps_px:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    representatives: list[np.ndarray] = []
    for idxs in clusters.values():
        # Outer = farthest from median tip x among cluster; tie-break on length
        xs = [tips[i][0] for i in idxs]
        med = float(np.median(xs))
        best = max(
            idxs,
            key=lambda i: (abs(tips[i][0] - med), contour_length_px(_as_points(contours[i]))),
        )
        representatives.append(contours[best])
    return representatives
