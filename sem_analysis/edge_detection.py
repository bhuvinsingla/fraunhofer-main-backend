"""Peak detection: multi-algorithm consensus edges + Harris/skyline/1D-ridge peaks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from scipy import signal
from scipy.interpolate import UnivariateSpline

from sem_analysis.consensus_edges import build_consensus_edges, edge_canny


@dataclass
class EdgePeakResult:
    """Edge coordinates and peak locations for a region."""

    shape_id: int
    edge_points: np.ndarray  # Nx2 float (x, y)
    peak_locations: np.ndarray  # Mx2 float (x, y)
    edge_map: np.ndarray | None = None
    hough_lines: list | None = None
    metadata: dict = field(default_factory=dict)


def _build_edge_map(image: np.ndarray, config: dict) -> np.ndarray:
    """
    Edge map from preprocessed image.

    Default: multi-algorithm consensus (vote, do not average).
    Fallback / single-mode: Canny(filtered, 40, 120).
    Then morphological closing + remove small components.
    """
    cfg = config.get("edge_detection", {})
    mode = str(cfg.get("edge_mode", "consensus")).lower()

    if mode == "canny":
        uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        edges = edge_canny(
            uint8,
            low=int(cfg.get("canny_low", 40)),
            high=int(cfg.get("canny_high", 120)),
        )
        consensus_meta = {"mode": "canny_only"}
    else:
        edges, consensus_meta = build_consensus_edges(image, config)
        consensus_meta["mode"] = "consensus"

    # Morphological closing to join broken edge segments
    close_k = cfg.get("morph_close_kernel", 3)
    if close_k and close_k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Remove small connected components
    min_area = cfg.get("min_component_area_px", 20)
    if min_area > 0:
        edges = _remove_small_components(edges, min_area)

    # Stash consensus metadata on the array for callers that inspect edge_map
    edges = np.ascontiguousarray(edges)
    edges.flags.writeable = True
    # Attach via a side channel is awkward; store in a module-level last-meta for pipeline
    _build_edge_map.last_meta = consensus_meta  # type: ignore[attr-defined]
    return edges


_build_edge_map.last_meta = {}  # type: ignore[attr-defined]


def _remove_small_components(edge_map: np.ndarray, min_area: int) -> np.ndarray:
    """Drop connected components smaller than min_area pixels."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edge_map, connectivity=8)
    cleaned = np.zeros_like(edge_map)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def _harris_peaks(edge_map: np.ndarray, config: dict, *, full_blade: bool = False) -> np.ndarray:
    """Harris corner detection on edge map — finds micro-serration peaks."""
    cfg = config.get("edge_detection", {})
    block_size = cfg.get("harris_block_size", 5)
    ksize = cfg.get("harris_ksize", 3)
    k = cfg.get("harris_k", 0.04)
    threshold = cfg.get("harris_threshold", 0.01)
    search_radius = cfg.get("harris_search_radius", 450)

    corners = cv2.cornerHarris(edge_map, block_size, ksize, k)
    corners = cv2.dilate(corners, None)
    peaks = []

    threshold_val = threshold * corners.max() if corners.max() > 0 else threshold
    ys, xs = np.where(corners > threshold_val)
    for x, y in zip(xs, ys):
        if edge_map[y, x] > 0:
            peaks.append([float(x), float(y)])

    if not peaks:
        return np.empty((0, 2), dtype=np.float64)

    peaks_arr = np.array(peaks, dtype=np.float64)
    if full_blade:
        return peaks_arr

    # Restrict to search radius from top edge (blade tip region)
    tip_y = np.min(peaks_arr[:, 1]) if len(peaks_arr) else 0
    mask = peaks_arr[:, 1] <= tip_y + search_radius
    return peaks_arr[mask] if mask.any() else peaks_arr


def _skyline_peaks(edge_map: np.ndarray, config: dict) -> np.ndarray:
    """Column-wise topmost edge pixel skyline + scipy find_peaks."""
    cfg = config.get("edge_detection", {})
    h, w = edge_map.shape
    skyline = np.full(w, np.nan, dtype=np.float64)

    for x in range(w):
        ys = np.where(edge_map[:, x] > 0)[0]
        if len(ys) > 0:
            skyline[x] = float(ys.min())

    valid = ~np.isnan(skyline)
    if valid.sum() < 5:
        return np.empty((0, 2), dtype=np.float64)

    xs_valid = np.where(valid)[0]
    profile = skyline[valid]
    inverted = -profile  # peaks in skyline (smaller y = higher on image) become peaks in -profile

    peak_idx, _ = signal.find_peaks(
        inverted,
        prominence=cfg.get("skyline_prominence", 2.0),
        distance=cfg.get("skyline_min_distance", 5),
    )

    peaks = np.column_stack([
        xs_valid[peak_idx].astype(np.float64),
        profile[peak_idx],
    ])
    return peaks


def _row_skyline_peaks(edge_map: np.ndarray, config: dict) -> np.ndarray:
    """Row-wise edge centerline + find_peaks — detects serrations along full blade height."""
    cfg = config.get("edge_detection", {})
    h, w = edge_map.shape
    centerline_x = np.full(h, np.nan, dtype=np.float64)

    for y in range(h):
        xs = np.where(edge_map[y, :] > 0)[0]
        if len(xs) > 0:
            centerline_x[y] = float(np.median(xs))

    valid = ~np.isnan(centerline_x)
    if valid.sum() < 10:
        return np.empty((0, 2), dtype=np.float64)

    ys_valid = np.where(valid)[0].astype(np.float64)
    profile = centerline_x[valid]
    prominence = cfg.get("row_skyline_prominence", 1.0)
    distance = cfg.get("row_skyline_min_distance", 4)

    peak_idx, _ = signal.find_peaks(profile, prominence=prominence, distance=distance)
    valley_idx, _ = signal.find_peaks(-profile, prominence=prominence, distance=distance)
    all_idx = np.unique(np.concatenate([peak_idx, valley_idx]))

    return np.column_stack([profile[all_idx], ys_valid[all_idx]])


def _extract_skyline_y(edge_map: np.ndarray) -> np.ndarray:
    """Column-wise topmost edge y (NaN where no edge)."""
    h, w = edge_map.shape[:2]
    skyline = np.full(w, np.nan, dtype=np.float64)
    for x in range(w):
        ys = np.where(edge_map[:, x] > 0)[0]
        if len(ys) > 0:
            skyline[x] = float(ys.min())
    return skyline


def _detect_peaks_1d_ridge(
    edge_map: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    1D ridge projection for asperity peak detection.

    1. Extract topmost edge pixels (skyline) per x
    2. Interpolate gaps
    3. Fit heavily smoothed UnivariateSpline (macroscopic ridge)
    4. Relative asperity height = smoothed − raw skyline
       (tips stick upward → smaller y → positive residual)
    5. scipy.signal.find_peaks on that 1D residual
    """
    cfg = config.get("edge_detection", {})
    h, w = edge_map.shape[:2]
    meta: dict[str, Any] = {"method": "1d_ridge_projection"}

    skyline = _extract_skyline_y(edge_map)
    xs = np.arange(w, dtype=np.float64)
    valid = ~np.isnan(skyline)
    n_valid = int(valid.sum())
    meta["n_skyline_valid"] = n_valid
    if n_valid < 10:
        meta["status"] = "too_few_skyline_points"
        return np.empty((0, 2), dtype=np.float64), meta

    # Interpolate gaps across full width
    skyline_filled = skyline.copy()
    skyline_filled[~valid] = np.interp(xs[~valid], xs[valid], skyline[valid])

    # Heavily smoothed spline → macroscopic ridge shape
    # s ~ N * sigma^2; large s = stronger smoothing
    default_s = float(n_valid) * float(cfg.get("spline_smoothing_factor", 50.0))
    spl_s_cfg = cfg.get("spline_smoothing")
    spl_s = default_s if spl_s_cfg is None else float(spl_s_cfg)
    spl_s = max(spl_s, 1.0)
    try:
        spline = UnivariateSpline(xs, skyline_filled, s=spl_s, k=min(3, max(1, n_valid - 1)))
        smooth = np.asarray(spline(xs), dtype=np.float64)
    except Exception as exc:
        meta["status"] = f"spline_failed: {exc}"
        # Fallback: very wide moving average
        k = max(51, (w // 20) | 1)
        kernel = np.ones(k, dtype=np.float64) / k
        smooth = np.convolve(skyline_filled, kernel, mode="same")

    # Relative height of asperities (positive at tips)
    residual = smooth - skyline_filled

    prominence = float(cfg.get("ridge_peak_prominence", cfg.get("skyline_prominence", 2.0)))
    distance = int(cfg.get("ridge_peak_distance", cfg.get("skyline_min_distance", 10)))
    height = cfg.get("ridge_peak_height")  # optional absolute residual threshold

    kwargs: dict[str, Any] = {"prominence": prominence, "distance": max(1, distance)}
    if height is not None:
        kwargs["height"] = float(height)

    peak_idx, props = signal.find_peaks(residual, **kwargs)
    peaks = (
        np.column_stack([xs[peak_idx], skyline_filled[peak_idx]])
        if len(peak_idx)
        else np.empty((0, 2), dtype=np.float64)
    )

    meta.update(
        {
            "status": "ok",
            "spline_smoothing": spl_s,
            "ridge_peak_prominence": prominence,
            "ridge_peak_distance": distance,
            "n_peaks": int(len(peaks)),
            "residual_max": float(np.max(residual)) if len(residual) else None,
            "residual_mean": float(np.mean(residual)) if len(residual) else None,
            "prominences": props.get("prominences", np.array([])).tolist()
            if "prominences" in props
            else [],
        }
    )
    return peaks, meta


def _deduplicate_peaks(peaks: np.ndarray, min_dist: float = 10.0) -> np.ndarray:
    """Merge peaks from Harris and skyline that are within min_dist pixels."""
    if len(peaks) == 0:
        return peaks
    order = np.lexsort((peaks[:, 0], peaks[:, 1]))
    sorted_peaks = peaks[order]
    kept = [sorted_peaks[0]]
    for p in sorted_peaks[1:]:
        if all(np.linalg.norm(p - k) >= min_dist for k in kept):
            kept.append(p)
    return np.array(kept, dtype=np.float64)


def _detect_hough_lines(edge_map: np.ndarray, config: dict) -> list:
    """Probabilistic Hough Line Transform for macroscopic blade arms."""
    cfg = config.get("edge_detection", {})
    lines = cv2.HoughLinesP(
        edge_map,
        rho=1,
        theta=np.pi / 180,
        threshold=cfg.get("hough_line_threshold", 50),
        minLineLength=cfg.get("hough_line_min_length", 30),
        maxLineGap=cfg.get("hough_line_max_gap", 10),
    )
    if lines is None:
        return []
    result = []
    for line in lines:
        coords = line.flatten().tolist()
        if len(coords) >= 4:
            result.append([int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])])
    return result


def _filter_peaks_near_centerline(
    peaks: np.ndarray,
    edge_map: np.ndarray,
    max_dist_px: float = 12.0,
) -> np.ndarray:
    """Keep peaks close to the row-wise blade centerline."""
    if len(peaks) == 0:
        return peaks
    h, w = edge_map.shape
    centerline_x = np.full(h, np.nan, dtype=np.float64)
    for y in range(h):
        xs = np.where(edge_map[y, :] > 0)[0]
        if len(xs) > 0:
            centerline_x[y] = float(np.median(xs))

    kept = []
    for x, y in peaks:
        yi = int(np.clip(round(y), 0, h - 1))
        cx = centerline_x[yi]
        if np.isnan(cx):
            continue
        if abs(x - cx) <= max_dist_px:
            kept.append([x, y])
    return np.array(kept, dtype=np.float64) if kept else np.empty((0, 2), dtype=np.float64)


def detect_serration_peaks_global(
    image: np.ndarray,
    config: dict,
    shape_id: int = 0,
) -> EdgePeakResult:
    """Detect all micro-serration / asperity peaks along the full blade edge.

    Core detector: 1D ridge projection (skyline − smoothed spline → find_peaks).
    Optional supplements: Harris / classic skyline / row-skyline when enabled.
    """
    cfg = config.get("edge_detection", {})
    edge_map = _build_edge_map(image, config)
    consensus_meta = dict(getattr(_build_edge_map, "last_meta", {}) or {})

    # Keep central blade region (ignore side margins)
    h, w = edge_map.shape
    margin = int(w * cfg.get("blade_margin_ratio", 0.05))
    edge_map[:, :margin] = 0
    edge_map[:, w - margin :] = 0

    # Core: 1D ridge projection asperity peaks
    ridge_peaks, ridge_meta = _detect_peaks_1d_ridge(edge_map, config)

    parts: list[np.ndarray] = []
    if len(ridge_peaks) > 0:
        parts.append(ridge_peaks)

    # Optional legacy supplements (off by default when 1D ridge is primary)
    if cfg.get("global_use_harris", False):
        harris = _filter_peaks_near_centerline(
            _harris_peaks(edge_map, config, full_blade=True),
            edge_map,
            max_dist_px=cfg.get("harris_centerline_dist_px", 12.0),
        )
    else:
        harris = np.empty((0, 2), dtype=np.float64)

    if cfg.get("global_use_skyline_supplement", False):
        skyline = _skyline_peaks(edge_map, config)
        row_skyline = _row_skyline_peaks(edge_map, config)
    else:
        skyline = np.empty((0, 2), dtype=np.float64)
        row_skyline = np.empty((0, 2), dtype=np.float64)

    for p in (harris, skyline, row_skyline):
        if len(p) > 0:
            parts.append(p)

    if parts:
        combined = np.vstack(parts)
    else:
        combined = np.empty((0, 2), dtype=np.float64)

    peak_locations = _deduplicate_peaks(
        combined, min_dist=cfg.get("peak_dedup_distance", 5.0)
    )

    # Sort peaks left-to-right along the ridge (then top-to-bottom)
    if len(peak_locations) > 0:
        peak_locations = peak_locations[np.lexsort((peak_locations[:, 1], peak_locations[:, 0]))]

    ys, xs = np.where(edge_map > 0)
    edge_points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    hough_lines = _detect_hough_lines(edge_map, config)

    return EdgePeakResult(
        shape_id=shape_id,
        edge_points=edge_points,
        peak_locations=peak_locations,
        edge_map=edge_map,
        hough_lines=hough_lines,
        metadata={
            "method": "1d_ridge_projection",
            "ridge_count": len(ridge_peaks),
            "ridge": ridge_meta,
            "harris_count": len(harris),
            "skyline_count": len(skyline),
            "row_skyline_count": len(row_skyline),
            "combined_count": len(peak_locations),
            "global": True,
            "edge_consensus": consensus_meta,
        },
    )


def detect_edges_and_peaks(
    image: np.ndarray,
    shape_id: int,
    contour: np.ndarray,
    config: dict,
) -> EdgePeakResult:
    """Detect edges and micro-serration peaks using Harris + skyline combination."""
    cfg = config.get("edge_detection", {})
    edge_map = _build_edge_map(image, config)
    consensus_meta = dict(getattr(_build_edge_map, "last_meta", {}) or {})

    mask = np.zeros_like(edge_map)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    edge_map = cv2.bitwise_and(edge_map, mask)

    ys, xs = np.where(edge_map > 0)
    edge_points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])

    harris = _harris_peaks(edge_map, config)
    skyline = _skyline_peaks(edge_map, config)

    if len(harris) > 0 and len(skyline) > 0:
        combined = np.vstack([harris, skyline])
    elif len(harris) > 0:
        combined = harris
    elif len(skyline) > 0:
        combined = skyline
    else:
        combined = np.empty((0, 2), dtype=np.float64)

    peak_locations = _deduplicate_peaks(
        combined, min_dist=cfg.get("peak_dedup_distance", 10.0)
    )
    hough_lines = _detect_hough_lines(edge_map, config)

    return EdgePeakResult(
        shape_id=shape_id,
        edge_points=edge_points,
        peak_locations=peak_locations,
        edge_map=edge_map,
        hough_lines=hough_lines,
        metadata={
            "harris_count": len(harris),
            "skyline_count": len(skyline),
            "combined_count": len(peak_locations),
            "edge_consensus": consensus_meta,
        },
    )
