"""Detect the topmost apex of each parabola-like arch along the central ridge.

The Fraunhofer SEM blade images show a chevron / herringbone structure: bright
lamella lines arc over and converge on a near-vertical spine, forming stacked
∧-shaped arches (parabolas opening downward). The tip we measure is the topmost
point (vertex) of each of those arches.

Detection strategy (classical CV, deterministic):
  1. CLAHE + white top-hat to isolate thin bright ridge lines over texture.
  2. Otsu threshold -> ridge mask; light open/close.
  3. Connected components -> each bright arc fragment.
  4. For every component, take its topmost pixel as a candidate vertex, but keep
     it only if the curve descends on BOTH sides of that point (a genuine ∧
     vertex, not a monotonic flank fragment).
  5. Restrict to the central band and de-duplicate by proximity.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def _fit_spine(
    cands: list[tuple[int, int, int]],
    W: int,
    tol_px: float,
    min_inliers: int,
) -> tuple[np.ndarray | None, float]:
    """Robustly fit the spine line x = a*y + b from ∧-vertex candidates.

    The arch vertices concentrate on the (possibly tilted) central spine, while
    stray vertices scatter over the flanks. We seed from the densest x-cluster,
    then refit on inliers a couple of times (simple IRLS/RANSAC-lite). Returns
    ``(coeffs, spine_x0)`` where coeffs are for np.polyval and spine_x0 is the
    spine x at the median y (used for the sanity band). coeffs is None if the
    fit is unreliable.
    """
    if len(cands) < min_inliers:
        return None, float(W) / 2.0
    xs = np.array([c[0] for c in cands], dtype=float)
    ys = np.array([c[1] for c in cands], dtype=float)

    # Seed: densest x-cluster via a coarse histogram (spine columns dominate).
    hist, edges = np.histogram(xs, bins=max(8, W // 40), range=(0, W))
    peak = int(np.argmax(hist))
    seed_x = 0.5 * (edges[peak] + edges[peak + 1])
    inl = np.abs(xs - seed_x) <= max(tol_px, 40.0)
    if int(inl.sum()) < min_inliers:
        # Fall back to global median column
        seed_x = float(np.median(xs))
        inl = np.abs(xs - seed_x) <= max(tol_px, 60.0)

    coeffs = np.array([0.0, seed_x])
    for _ in range(3):
        if int(inl.sum()) < min_inliers:
            break
        # Fit x as a function of y (spine is near-vertical → x varies slowly)
        try:
            coeffs = np.polyfit(ys[inl], xs[inl], 1)
        except Exception:  # pragma: no cover
            break
        pred = np.polyval(coeffs, ys)
        inl = np.abs(xs - pred) <= tol_px
    if int(inl.sum()) < min_inliers:
        return None, float(np.median(xs))
    spine_x0 = float(np.polyval(coeffs, float(np.median(ys))))
    return coeffs, spine_x0


def _as_gray_u8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        if cv2 is not None:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        else:  # pragma: no cover
            arr = arr.mean(axis=2)
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    return np.ascontiguousarray(arr.astype(np.uint8))


def detect_parabola_apexes(
    image: np.ndarray,
    config: dict | None = None,
) -> list[tuple[float, float]]:
    """Return apex points (x, y) of the central parabola-like arches.

    Coordinates are in the pixel frame of ``image`` (the measurement ROI).
    """
    if cv2 is None:  # pragma: no cover
        return []
    cfg = ((config or {}).get("parabola_apex", {})) or {}

    tophat_ksize = int(cfg.get("tophat_ksize", 13))
    min_area = int(cfg.get("min_area_px", 80))
    min_width = int(cfg.get("min_width_px", 10))
    descend_px = int(cfg.get("descend_px", 5))
    side_px = int(cfg.get("side_px", 5))
    x_lo = float(cfg.get("central_x_lo", 0.28))
    x_hi = float(cfg.get("central_x_hi", 0.74))
    dedup_px = float(cfg.get("dedup_px", 20.0))
    top_margin = int(cfg.get("top_margin_px", 15))
    max_apexes = int(cfg.get("max_apexes", 20))
    auto_spine = bool(cfg.get("auto_spine", True))
    spine_tol_px = float(cfg.get("spine_tol_px", 60.0))
    spine_min_inliers = int(cfg.get("spine_min_inliers", 3))
    sanity_lo = float(cfg.get("sanity_x_lo", 0.12))
    sanity_hi = float(cfg.get("sanity_x_hi", 0.88))

    g = _as_gray_u8(image)
    H, W = g.shape

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(g)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat_ksize, tophat_ksize))
    tophat = cv2.morphologyEx(eq, cv2.MORPH_TOPHAT, k)
    tophat = cv2.GaussianBlur(tophat, (3, 3), 0)

    _, mask = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: list[tuple[int, int, int]] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        if area < min_area or ww < min_width:
            continue
        ys, xs = np.where(lab == i)
        ymin = int(ys.min())
        if ymin < top_margin:
            continue
        band = ys <= ymin + 2
        ax = int(np.median(xs[band]))
        ay = ymin
        below = ys >= ay + descend_px
        if int(below.sum()) < 8:
            continue
        left = bool(np.any(xs[below] < ax - side_px))
        right = bool(np.any(xs[below] > ax + side_px))
        if not (left and right):
            continue  # flank fragment, not a genuine arch vertex
        candidates.append((ax, ay, area))

    candidates.sort(key=lambda t: t[1])  # top-to-bottom

    # Locate the spine adaptively so off-center blade edges still work; the fixed
    # band is only a wide sanity bound to avoid latching onto a flank cluster.
    spine_coeffs = None
    if auto_spine:
        spine_coeffs, _spine_x0 = _fit_spine(
            candidates, W, spine_tol_px, spine_min_inliers
        )

    def _accept(ax: int, ay: int) -> bool:
        if not (sanity_lo * W < ax < sanity_hi * W):
            return False
        if spine_coeffs is not None:
            return abs(ax - float(np.polyval(spine_coeffs, ay))) <= spine_tol_px
        # Fallback: fixed central band when spine could not be fit
        return x_lo * W < ax < x_hi * W

    kept: list[tuple[float, float]] = []
    for ax, ay, _area in candidates:
        if not _accept(ax, ay):
            continue
        if any((ax - kx) ** 2 + (ay - ky) ** 2 < dedup_px ** 2 for kx, ky in kept):
            continue
        kept.append((float(ax), float(ay)))
        if len(kept) >= max_apexes:
            break
    return kept
