"""Multi-scale SEM ridge / asperity extraction for textured serrated blades.

Pipeline:
  1. Macro wedge mask (heavy blur + Otsu) — search only inside blade body
  2. White top-hat + raw brightness gate — isolate bright macroscopic ridge
  3. Bidirectional curve tracking — contiguous wavy ridge polyline
  4. Local peak detection — individual asperities along the tracked curve
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from scipy import ndimage, signal

log = logging.getLogger("sem-api.stages")


@dataclass
class MultiScaleResult:
    """Outputs of the multi-scale asperity pipeline."""

    wedge_mask: np.ndarray
    ridge_mask: np.ndarray
    tracked_curve: np.ndarray  # Nx2 float (x, y)
    peaks: np.ndarray  # Mx2 float (x, y)
    meta: dict[str, Any] = field(default_factory=dict)


def _as_gray_u8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        if arr.dtype != np.uint8:
            f = arr.astype(np.float32)
            mx = float(np.nanmax(f)) if f.size else 1.0
            if mx <= 1.0 + 1e-6:
                arr = (np.clip(f, 0, 1) * 255).astype(np.uint8)
            else:
                arr = np.clip(f, 0, 255).astype(np.uint8)
        if arr.shape[2] == 1:
            return np.ascontiguousarray(arr[:, :, 0], dtype=np.uint8)
        return np.ascontiguousarray(cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    f = arr.astype(np.float32)
    mx = float(np.nanmax(f)) if f.size else 1.0
    if mx <= 1.0 + 1e-6:
        return (np.clip(f, 0, 1) * 255).astype(np.uint8)
    return np.clip(f, 0, 255).astype(np.uint8)


def extract_macro_wedge(
    image_gray: np.ndarray,
    *,
    blur_ksize: int = 51,
    dilate_px: int = 8,
) -> np.ndarray:
    """
    Heavily blur + Otsu → solid macroscopic blade body (wedge/triangle).
    Dilate and return a hard binary mask (255 = inside blade).
    """
    gray = _as_gray_u8(image_gray)
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    k = max(3, k)
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    # Prefer the brighter / darker body that covers a substantial area
    _, thr_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, thr = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Choose the mask whose largest CC is neither tiny nor nearly full-frame
    h, w = gray.shape[:2]
    area = h * w

    def _score(mask: np.ndarray) -> tuple[float, np.ndarray]:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return 0.0, mask
        # Largest non-background component
        areas = stats[1:, cv2.CC_STAT_AREA]
        i = int(np.argmax(areas)) + 1
        a = float(areas[i - 1])
        frac = a / max(area, 1)
        # Prefer mid-sized blade body (~5–85% of frame)
        score = a if 0.05 <= frac <= 0.85 else a * 0.1
        out = np.zeros_like(mask)
        out[labels == i] = 255
        return score, out

    s1, m1 = _score(thr)
    s2, m2 = _score(thr_inv)
    wedge = m1 if s1 >= s2 else m2

    if dilate_px > 0:
        ker = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        wedge = cv2.dilate(wedge, ker, iterations=1)

    # Fill holes so the wedge is a solid polygon
    n, labels, stats, _ = cv2.connectedComponentsWithStats(wedge, connectivity=8)
    if n > 1:
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        solid = np.zeros_like(wedge)
        solid[labels == i] = 255
        # Morphological close to seal small gaps
        solid = cv2.morphologyEx(
            solid, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        )
        wedge = solid

    return wedge


def extract_bright_ridge_otsu(
    image_gray: np.ndarray,
    wedge_mask: np.ndarray | None = None,
    *,
    tophat_ksize: int = 15,
) -> np.ndarray:
    """
    White top-hat + Otsu, gated by absolute raw-brightness Otsu.
    Isolates the bright macroscopic ridge from texture crests.
    """
    gray = _as_gray_u8(image_gray)
    k = tophat_ksize if tophat_ksize % 2 == 1 else tophat_ksize + 1
    k = max(3, k)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    # Otsu on top-hat response
    _, th_hat = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Absolute brightness gate (bright ridge vs dark background / texture)
    _, th_bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ridge = cv2.bitwise_and(th_hat, th_bright)
    if wedge_mask is not None:
        mask = wedge_mask if wedge_mask.dtype == np.uint8 else (wedge_mask > 0).astype(np.uint8) * 255
        if mask.shape[:2] != ridge.shape[:2]:
            mask = cv2.resize(mask, (ridge.shape[1], ridge.shape[0]), interpolation=cv2.INTER_NEAREST)
        ridge = cv2.bitwise_and(ridge, mask)

    # Clean speckles
    ridge = cv2.morphologyEx(
        ridge, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    ridge = cv2.morphologyEx(
        ridge, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    return ridge


def track_ridge_bidirectional(
    ridge_mask: np.ndarray,
    *,
    max_gap: int = 6,
    search_half_width: int = 12,
) -> np.ndarray:
    """
    Trace a contiguous ridge by seeding at the median bright column and
    walking both directions along Y (up and down), snapping to nearby bright pixels.
    Returns Nx2 float array of (x, y) along the wavy curve (sorted by y).
    """
    mask = (ridge_mask > 0).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return np.zeros((0, 2), dtype=float)

    # Seed: median of bright pixels near the vertical mid-band
    y_lo, y_hi = np.percentile(ys, [25, 75]).astype(int)
    band = (ys >= y_lo) & (ys <= y_hi)
    if not np.any(band):
        band = np.ones(len(ys), dtype=bool)
    seed_x = float(np.median(xs[band]))
    seed_y = float(np.median(ys[band]))

    # Build row → list of x for O(1) lookup
    h, w = mask.shape[:2]
    row_xs: list[np.ndarray] = [np.array([], dtype=int) for _ in range(h)]
    for y, x in zip(ys.tolist(), xs.tolist()):
        # collect later via bincount-style
        pass
    # Faster: for each row take argmax of a horizontal profile from the mask
    # Precompute nearest bright x per row via distance transform style
    rows_with = {}
    for y, x in zip(ys, xs):
        rows_with.setdefault(int(y), []).append(int(x))
    for y in list(rows_with.keys()):
        rows_with[y] = np.asarray(sorted(rows_with[y]), dtype=int)

    def _track(direction: int) -> list[list[float]]:
        """direction: -1 up (decreasing y), +1 down."""
        pts: list[list[float]] = []
        x_cur = seed_x
        y_cur = int(round(seed_y))
        gaps = 0
        while 0 <= y_cur < h:
            cands = rows_with.get(y_cur)
            if cands is None or len(cands) == 0:
                gaps += 1
                if gaps > max_gap:
                    break
                y_cur += direction
                continue
            # Prefer candidate nearest to current x within search window
            d = np.abs(cands.astype(float) - x_cur)
            j = int(np.argmin(d))
            if float(d[j]) > search_half_width and gaps > 0:
                # Try expand once
                if float(d[j]) > search_half_width * 2:
                    gaps += 1
                    if gaps > max_gap:
                        break
                    y_cur += direction
                    continue
            x_cur = float(cands[j])
            pts.append([x_cur, float(y_cur)])
            gaps = 0
            y_cur += direction
        return pts

    up = _track(-1)
    down = _track(+1)
    # Merge: up reversed (excluding seed duplicate) + down
    up_sorted = list(reversed(up))
    if up_sorted and down:
        # drop duplicate seed row if both include it
        if abs(up_sorted[-1][1] - down[0][1]) < 0.5:
            curve = up_sorted + down[1:]
        else:
            curve = up_sorted + down
    else:
        curve = up_sorted or down

    if not curve:
        # Fallback: skyline of ridge mask
        sky_x, sky_y = [], []
        for y in range(h):
            cols = np.where(mask[y] > 0)[0]
            if len(cols):
                sky_x.append(float(np.median(cols)))
                sky_y.append(float(y))
        if not sky_x:
            return np.zeros((0, 2), dtype=float)
        return np.column_stack([sky_x, sky_y]).astype(float)

    arr = np.asarray(curve, dtype=float)
    # Sort by y and dedup rows
    order = np.argsort(arr[:, 1])
    arr = arr[order]
    keep = [0]
    for i in range(1, len(arr)):
        if abs(arr[i, 1] - arr[keep[-1], 1]) >= 0.5:
            keep.append(i)
    return arr[keep]


def find_local_peaks(
    tracked_curve: np.ndarray,
    *,
    smooth_sigma: float = 2.0,
    prominence: float = 2.0,
    min_distance: int = 8,
    peak_kind: str = "auto",
) -> np.ndarray:
    """
    Gaussian-smooth the tracked ridge x(y) and find local extrema (asperities).

    For a roughly vertical blade edge, asperities are local maxima/minima of x
    as y sweeps. peak_kind: 'max' | 'min' | 'auto' (choose the stronger set).
    Returns Mx2 (x, y).
    """
    curve = np.asarray(tracked_curve, dtype=float).reshape(-1, 2)
    if len(curve) < max(10, min_distance * 2):
        return np.zeros((0, 2), dtype=float)

    # Resample onto uniform y grid
    y = curve[:, 1]
    x = curve[:, 0]
    y_min, y_max = float(y.min()), float(y.max())
    if y_max - y_min < 5:
        return np.zeros((0, 2), dtype=float)
    ys = np.arange(np.floor(y_min), np.ceil(y_max) + 1, 1.0)
    xs = np.interp(ys, y, x)
    if smooth_sigma > 0:
        xs_s = ndimage.gaussian_filter1d(xs, sigma=smooth_sigma)
    else:
        xs_s = xs

    def _peaks(arr: np.ndarray) -> np.ndarray:
        idx, _ = signal.find_peaks(
            arr,
            prominence=prominence,
            distance=max(1, min_distance),
        )
        return idx

    idx_max = _peaks(xs_s)
    idx_min = _peaks(-xs_s)

    kind = (peak_kind or "auto").lower()
    if kind == "max":
        idx = idx_max
    elif kind == "min":
        idx = idx_min
    else:
        # Prefer the set with more / stronger peaks
        if len(idx_max) >= len(idx_min):
            idx = idx_max
        else:
            idx = idx_min

    if len(idx) == 0:
        return np.zeros((0, 2), dtype=float)

    peaks = np.column_stack([xs_s[idx], ys[idx]]).astype(float)
    return peaks


def run_multi_scale(
    image: np.ndarray,
    config: dict | None = None,
) -> MultiScaleResult:
    """Full multi-scale extraction → wedge, ridge, curve, asperity peaks."""
    cfg = (config or {}).get("multi_scale", {}) or {}
    gray = _as_gray_u8(image)

    wedge = extract_macro_wedge(
        gray,
        blur_ksize=int(cfg.get("wedge_blur_ksize", 51)),
        dilate_px=int(cfg.get("wedge_dilate_px", 8)),
    )
    ridge = extract_bright_ridge_otsu(
        gray,
        wedge,
        tophat_ksize=int(cfg.get("tophat_ksize", 15)),
    )
    curve = track_ridge_bidirectional(
        ridge,
        max_gap=int(cfg.get("track_max_gap", 6)),
        search_half_width=int(cfg.get("track_search_half_width", 12)),
    )
    peaks = find_local_peaks(
        curve,
        smooth_sigma=float(cfg.get("peak_smooth_sigma", 2.0)),
        prominence=float(cfg.get("peak_prominence", 2.0)),
        min_distance=int(cfg.get("peak_min_distance", 8)),
        peak_kind=str(cfg.get("peak_kind", "auto")),
    )

    meta = {
        "n_curve_points": int(len(curve)),
        "n_peaks": int(len(peaks)),
        "wedge_area_px": int(np.count_nonzero(wedge)),
        "ridge_area_px": int(np.count_nonzero(ridge)),
    }
    log.info(
        "[MULTI-SCALE] wedge_area=%d ridge_area=%d curve=%d peaks=%d",
        meta["wedge_area_px"],
        meta["ridge_area_px"],
        meta["n_curve_points"],
        meta["n_peaks"],
    )
    return MultiScaleResult(
        wedge_mask=wedge,
        ridge_mask=ridge,
        tracked_curve=curve,
        peaks=peaks,
        meta=meta,
    )


def refine_ms_peaks_with_cv(
    ms_peaks: np.ndarray,
    cv_peaks: np.ndarray,
    edge_points: np.ndarray | None = None,
    *,
    snap_px: float = 30.0,
) -> np.ndarray:
    """
    Use multi-scale asperity locations as the tip list; snap each to the nearest
    OpenCV ridge peak (or edge point) for sub-pixel accuracy when close enough.
    """
    ms = np.asarray(ms_peaks, dtype=float).reshape(-1, 2)
    cv = np.asarray(cv_peaks, dtype=float).reshape(-1, 2) if cv_peaks is not None else np.empty((0, 2))
    ep = (
        np.asarray(edge_points, dtype=float).reshape(-1, 2)
        if edge_points is not None
        else np.empty((0, 2))
    )
    if len(ms) == 0:
        return ms

    out: list[np.ndarray] = []
    for m in ms:
        chosen = m
        if len(cv) > 0:
            d = np.linalg.norm(cv - m, axis=1)
            j = int(np.argmin(d))
            if float(d[j]) <= snap_px:
                chosen = cv[j]
                out.append(chosen)
                continue
        if len(ep) > 0:
            d = np.linalg.norm(ep - m, axis=1)
            j = int(np.argmin(d))
            if float(d[j]) <= snap_px:
                chosen = ep[j]
        out.append(chosen)

    arr = np.asarray(out, dtype=float).reshape(-1, 2)
    # Dedup
    kept: list[np.ndarray] = []
    for p in arr:
        if all(np.linalg.norm(p - q) >= 3.0 for q in kept):
            kept.append(p)
    return np.asarray(kept, dtype=float).reshape(-1, 2)


def filter_peaks_by_multi_scale_y(
    cv_peaks: np.ndarray,
    ms_peaks: np.ndarray,
    *,
    y_tol_px: float = 12.0,
    x_tol_px: float | None = None,
) -> np.ndarray:
    """
    Keep OpenCV / ridge peaks whose Y (and optionally X) aligns with a
    multi-scale structural asperity. Returns filtered Nx2 array.
    """
    cv = np.asarray(cv_peaks, dtype=float).reshape(-1, 2)
    ms = np.asarray(ms_peaks, dtype=float).reshape(-1, 2)
    if len(cv) == 0:
        return cv
    if len(ms) == 0:
        return cv  # nothing to filter against — keep originals

    kept: list[np.ndarray] = []
    for p in cv:
        dy = np.abs(ms[:, 1] - p[1])
        if x_tol_px is not None:
            dx = np.abs(ms[:, 0] - p[0])
            ok = np.any((dy <= y_tol_px) & (dx <= float(x_tol_px)))
        else:
            ok = np.any(dy <= y_tol_px)
        if ok:
            kept.append(p)

    if not kept:
        return np.zeros((0, 2), dtype=float)

    arr = np.asarray(kept, dtype=float)
    out: list[np.ndarray] = []
    for p in arr:
        if all(np.linalg.norm(p - q) >= 3.0 for q in out):
            out.append(p)
    return np.asarray(out, dtype=float).reshape(-1, 2)
