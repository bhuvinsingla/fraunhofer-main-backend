"""Brainstorming method visualization — separate per-method annotated outputs."""

from __future__ import annotations

import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

from sem_analysis.deduction import FilteredDetection
from sem_analysis.edge_detection import EdgePeakResult
from sem_analysis.methods.fixed_distance_circle import FixedDistanceCircleResult
from sem_analysis.methods.inscribed_angle import InscribedAngleResult
from sem_analysis.methods.projected_tip_distance import ProjectedTipDistanceResult
from sem_analysis.radius_computation import RadiusResult

# BGR colors
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
GREEN = (0, 255, 0)
MAGENTA = (255, 0, 255)
PURPLE = (180, 0, 180)       # parabola curves (Approach 2)
PINK = (180, 105, 255)       # vertices (h,k) (Approach 2) — light pink in BGR


def _draw_scale_bar(ax, nm_per_pixel: float, scale_bar_nm: float, position: str = "bottom-right") -> None:
    bar_px = scale_bar_nm / nm_per_pixel
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    margin = 20
    x_start = xlim[1] - bar_px - margin if "right" in position else xlim[0] + margin
    y_pos = ylim[0] + margin if "bottom" in position else ylim[1] - margin - 10
    ax.plot([x_start, x_start + bar_px], [y_pos, y_pos], "w-", linewidth=3)
    ax.text(
        x_start + bar_px / 2, y_pos + 5, f"{scale_bar_nm:.0f} nm",
        color="white", ha="center", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.6),
    )


def _save_annotated(img: np.ndarray, output_path: str, nm_per_pixel: float, config: dict) -> None:
    cfg = config.get("annotation", {})
    scale_bar_nm = cfg.get("scale_bar_nm", 50)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    _draw_scale_bar(ax, nm_per_pixel, scale_bar_nm, cfg.get("scale_bar_position", "bottom-right"))
    plt.tight_layout()
    plt.savefig(output_path, dpi=cfg.get("output_dpi", 300), bbox_inches="tight")
    plt.close()


def _base_image(image: np.ndarray) -> np.ndarray:
    uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(uint8, cv2.COLOR_GRAY2BGR)


def _draw_line(img: np.ndarray, coords: list[float], color: tuple, thickness: int = 2) -> None:
    if len(coords) < 4:
        return
    x1, y1, x2, y2 = (int(v) for v in coords[:4])
    cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def _draw_dot(img: np.ndarray, point: tuple[float, float] | list[float], color: tuple, radius: int = 4) -> None:
    cv2.circle(img, (int(point[0]), int(point[1])), radius, color, -1, cv2.LINE_AA)


def _draw_vertical_l(img: np.ndarray, coords: list[float], label: str = "l") -> None:
    """PDF red vertical bracket with end ticks + 'l' label."""
    if len(coords) < 4:
        return
    vals = [float(v) for v in coords[:4]]
    x = int(round(0.5 * (vals[0] + vals[2])))
    y_top = int(round(min(vals[1], vals[3])))
    y_bottom = int(round(max(vals[1], vals[3])))
    if y_bottom - y_top < 2:
        return
    cv2.line(img, (x, y_top), (x, y_bottom), RED, 2, cv2.LINE_AA)
    tick = 8
    cv2.line(img, (x - tick, y_top), (x + tick, y_top), RED, 2, cv2.LINE_AA)
    cv2.line(img, (x - tick, y_bottom), (x + tick, y_bottom), RED, 2, cv2.LINE_AA)
    cv2.putText(
        img,
        label,
        (x + 10, (y_top + y_bottom) // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        RED,
        2,
        cv2.LINE_AA,
    )


def _draw_polyline(img: np.ndarray, points: list, color: tuple, thickness: int = 1) -> None:
    if not points or len(points) < 2:
        return
    pts = np.array([[int(p[0]), int(p[1])] for p in points], dtype=np.int32)
    cv2.polylines(img, [pts], False, color, thickness, cv2.LINE_AA)


def _draw_caret(img: np.ndarray, tip: tuple[float, float] | list[float], color: tuple = BLUE) -> None:
    """Blue V marker at the ultimate tip (as in the reference SEM annotation)."""
    x, y = int(tip[0]), int(tip[1])
    cv2.line(img, (x - 7, y + 10), (x, y), color, 2, cv2.LINE_AA)
    cv2.line(img, (x + 7, y + 10), (x, y), color, 2, cv2.LINE_AA)


def _draw_alpha_arc(img: np.ndarray, arc: dict) -> None:
    if not arc:
        return
    c = arc.get("center")
    r = int(max(8, arc.get("radius", 24)))
    if not c:
        return
    cx, cy = int(c[0]), int(c[1])
    start = float(arc.get("start_deg", 0))
    end = float(arc.get("end_deg", 40))
    cv2.ellipse(img, (cx, cy), (r, r), 0, start, end, RED, 2, cv2.LINE_AA)
    mid = math.radians(0.5 * (start + end))
    lx = int(cx + (r + 10) * math.cos(mid))
    ly = int(cy + (r + 10) * math.sin(mid))
    cv2.putText(img, "a", (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2, cv2.LINE_AA)


def _draw_d_bracket(img: np.ndarray, coords: list[float], label: str = "d") -> None:
    """Red distance bracket from projected tip to ultimate tip (image label d)."""
    if len(coords) < 4:
        return
    x1, y1, x2, y2 = (float(v) for v in coords[:4])
    # Offset bracket slightly to the side so it doesn't cover the tip
    mid_x = 0.5 * (x1 + x2)
    offset = 14.0
    p1 = (int(mid_x + offset), int(y1))
    p2 = (int(mid_x + offset), int(y2))
    cv2.line(img, p1, p2, RED, 2, cv2.LINE_AA)
    tick = 6
    cv2.line(img, (p1[0] - tick, p1[1]), (p1[0] + tick, p1[1]), RED, 2, cv2.LINE_AA)
    cv2.line(img, (p2[0] - tick, p2[1]), (p2[0] + tick, p2[1]), RED, 2, cv2.LINE_AA)
    cv2.putText(
        img, label,
        (p1[0] + 8, int(0.5 * (p1[1] + p2[1]))),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2, cv2.LINE_AA,
    )


def _draw_whiteboard_tip(img: np.ndarray, curve: dict, index: int = 0) -> None:
    """
    Draw one tip exactly like the reference annotation:
      blue edges · yellow flanks · red α · red d · cyan circle · red radius · blue caret
    """
    # 1) Actual blade edges (blue)
    _draw_polyline(img, curve.get("edge_left") or [], BLUE, 1)
    _draw_polyline(img, curve.get("edge_right") or [], BLUE, 1)

    # 2) Yellow projected flanks (V)
    _draw_line(img, curve.get("left_line", []), YELLOW, 2)
    _draw_line(img, curve.get("right_line", []), YELLOW, 2)

    # 3) Projected tip + ultimate tip caret
    projected = curve.get("projected_tip") or curve.get("convergence_point")
    ultimate = curve.get("ultimate_tip") or curve.get("tip_point") or curve.get("peak_location")
    if projected:
        _draw_dot(img, projected, YELLOW, 4)
    if ultimate:
        _draw_caret(img, ultimate, BLUE)

    # 4) Red α arc at projected tip
    _draw_alpha_arc(img, curve.get("alpha_arc") or {})

    # 5) Red d bracket
    _draw_d_bracket(img, curve.get("d_bracket") or curve.get("vertical_l_line") or [], "d")

    # 6) Cyan inscribed circle + red radius / diameter
    center = curve.get("circle_center") or curve.get("center")
    r_px = curve.get("circle_radius_px") or curve.get("radius_px")
    if center and r_px:
        cv2.circle(img, (int(center[0]), int(center[1])), max(3, int(r_px)), CYAN, 2, cv2.LINE_AA)
        _draw_dot(img, center, RED, 3)
        spoke = curve.get("radius_spoke")
        if spoke:
            _draw_line(img, spoke, RED, 2)
        else:
            _draw_line(
                img,
                [center[0], center[1], center[0] + float(r_px), center[1]],
                RED,
                2,
            )
        diam = curve.get("diameter_line")
        if diam:
            _draw_line(img, diam, RED, 1)

    # Labels
    if ultimate and curve.get("radius_nm") is not None:
        px, py = int(ultimate[0]), int(ultimate[1])
        parts = [f"R={curve['radius_nm']:.1f}nm"]
        if curve.get("d_nm") is not None:
            parts.append(f"d={curve['d_nm']:.1f}nm")
        if curve.get("included_angle_deg") is not None:
            parts.append(f"a={curve['included_angle_deg']:.1f}")
        label = "  ".join(parts)
        ly = py - 14 if index % 2 == 0 else py + 18
        cv2.putText(img, label, (px + 10, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.38, YELLOW, 1, cv2.LINE_AA)


def annotate_whiteboard_image(
    image: np.ndarray,
    per_tip: list[dict],
    nm_per_pixel: float,
    config: dict,
    output_path: str | None = None,
) -> np.ndarray:
    """Composite overlay matching the hand-annotated SEM reference."""
    img = _base_image(image)
    for i, tip in enumerate(per_tip):
        _draw_whiteboard_tip(img, tip, i)
    if output_path:
        _save_annotated(img, output_path, nm_per_pixel, config)
    return img


def _draw_method1_curve(img: np.ndarray, curve: dict, index: int) -> None:
    """Method 1 PDF overlay: 3 blue dots, red vertical l, red horizontal chord, cyan circle."""
    tip = curve.get("tip_point") or curve.get("peak_location")
    left = curve.get("intersection_left")
    right = curve.get("intersection_right")
    center = curve.get("center")
    radius_px = curve.get("radius_px")
    scan_line = curve.get("scan_line", [])
    vertical_l = curve.get("vertical_l_line", [])
    rejected = curve.get("valid") is False or bool(
        curve.get("rejection_reason") or curve.get("method1_rejection_reason")
    )
    reason = curve.get("rejection_reason") or curve.get("method1_rejection_reason")

    if tip:
        _draw_dot(img, tip, BLUE)
    if left:
        _draw_dot(img, left, BLUE, 3)
    if right:
        _draw_dot(img, right, BLUE, 3)
    if vertical_l:
        _draw_vertical_l(img, vertical_l, "l")
    elif tip and left and right:
        mid_x = 0.5 * (float(left[0]) + float(right[0]))
        mid_y = 0.5 * (float(left[1]) + float(right[1]))
        _draw_vertical_l(img, [float(tip[0]), float(tip[1]), mid_x, mid_y], "l")
    if scan_line:
        _draw_line(img, scan_line, RED, 2)
    if center and radius_px and not rejected:
        cv2.circle(img, (int(center[0]), int(center[1])), max(3, int(radius_px)), CYAN, 1, cv2.LINE_AA)

    peak = curve.get("peak_location") or tip
    if peak:
        px, py = int(peak[0]), int(peak[1])
        ly = py - 6 if index % 2 == 0 else py + 12
        if rejected:
            label = f"R fail: {reason or 'invalid'}"
            if reason == "no_intersection":
                label = "R fail: no_intersection (check nm/px)"
            cv2.putText(img, label, (px + 6, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.32, RED, 1, cv2.LINE_AA)
        elif curve.get("radius_nm") is not None:
            label = f"R={curve['radius_nm']:.1f}nm"
            cv2.putText(img, label, (px + 6, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.32, YELLOW, 1, cv2.LINE_AA)


def _draw_method2_curve(img: np.ndarray, curve: dict, index: int) -> None:
    """Method 2 PDF overlay: yellow projected edges → convergent tip, blue ultimate tip, red l."""
    _draw_line(img, curve.get("left_line", []), YELLOW, 2)
    _draw_line(img, curve.get("right_line", []), YELLOW, 2)

    projected = curve.get("convergence_point") or curve.get("projected_tip")
    if projected:
        _draw_dot(img, projected, YELLOW, 5)

    arc = curve.get("tip_apex_arc") or []
    if len(arc) >= 2:
        _draw_polyline(img, arc, BLUE, 2)
    tip = curve.get("tip_point") or curve.get("peak_location")
    if tip:
        _draw_dot(img, tip, BLUE, 4)

    vertical_l = curve.get("vertical_l_line", [])
    if vertical_l:
        _draw_vertical_l(img, vertical_l, "l")
    elif projected and tip:
        _draw_vertical_l(
            img,
            [float(projected[0]), float(projected[1]), float(projected[0]), float(tip[1])],
            "l",
        )

    peak = curve.get("peak_location") or tip
    if peak and curve.get("distance_l_nm") is not None:
        px, py = int(peak[0]), int(peak[1])
        label = f"l={curve['distance_l_nm']:.1f}nm"
        ly = py - 6 if index % 2 == 0 else py + 14
        cv2.putText(img, label, (px + 8, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.38, RED, 1, cv2.LINE_AA)


def _draw_method3_curve(img: np.ndarray, curve: dict, index: int) -> None:
    """Method 3 PDF overlay: cyan fixed-D circle, yellow rays tip→edge, θ label."""
    center = curve.get("circle_center")
    radius_px = curve.get("circle_radius_px")
    if center and radius_px:
        cv2.circle(img, (int(center[0]), int(center[1])), max(3, int(radius_px)), CYAN, 1, cv2.LINE_AA)

    tip = curve.get("tip_point")
    if tip:
        _draw_dot(img, tip, BLUE, 4)

    left = curve.get("intersection_left")
    right = curve.get("intersection_right")
    if left:
        _draw_dot(img, left, BLUE, 3)
    if right:
        _draw_dot(img, right, BLUE, 3)

    _draw_line(img, curve.get("left_tangent_line", []), YELLOW, 2)
    _draw_line(img, curve.get("right_tangent_line", []), YELLOW, 2)

    peak = curve.get("peak_location") or tip
    if peak and curve.get("angle_degrees") is not None:
        px, py = int(peak[0]), int(peak[1])
        label = f"θ={curve['angle_degrees']:.1f}°"
        ly = py - 6 if index % 2 == 0 else py + 12
        cv2.putText(img, label, (px + 6, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.32, YELLOW, 1, cv2.LINE_AA)


def _draw_all_serration_curves(img: np.ndarray, radius_results: list[RadiusResult]) -> None:
    """Draw cyan fitted circle + R label at every detected serration peak."""
    for i, r in enumerate(radius_results):
        cx, cy = int(r.center[0]), int(r.center[1])
        r_px = max(3, int(r.radius_px))
        cv2.circle(img, (cx, cy), r_px, CYAN, 1, cv2.LINE_AA)

        if r.peak_location:
            px, py = int(r.peak_location[0]), int(r.peak_location[1])
        else:
            px, py = cx, cy - r_px
        cv2.circle(img, (px, py), 3, BLUE, -1, cv2.LINE_AA)

        if r.tangent_lines:
            for p1, p2 in r.tangent_lines:
                pt1 = (int(p1[0]), int(p1[1]))
                pt2 = (int(p2[0]), int(p2[1]))
                cv2.line(img, pt1, pt2, MAGENTA, 1, cv2.LINE_AA)

        label = f"R={r.radius_nm:.1f}nm"
        if r.opening_angle_deg is not None:
            label += f" A={r.opening_angle_deg:.0f}°"
        lx = px + 6
        ly = py - 4 if i % 2 == 0 else py + 14
        cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.32, YELLOW, 1, cv2.LINE_AA)


def annotate_method1_image(
    image: np.ndarray,
    per_curve: list[dict],
    nm_per_pixel: float,
    config: dict,
    output_path: str | None = None,
) -> np.ndarray:
    """Method 1 annotated image — fixed-distance inscribed circle per curve."""
    img = _base_image(image)
    for i, curve in enumerate(per_curve):
        _draw_method1_curve(img, curve, i)
    cv2.putText(
        img,
        "Method 1: blue=3 points  red=l + horizontal  cyan=circle",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        BLUE,
        2,
        cv2.LINE_AA,
    )
    if output_path:
        _save_annotated(img, output_path, nm_per_pixel, config)
    return img


def _draw_circular_arc_curve(img: np.ndarray, curve: dict, index: int) -> None:
    """Approach 2: purple parabola/arc polyline + pink vertex (or circle center)."""
    arc = curve.get("curve_points") or curve.get("arc_points") or []
    if len(arc) >= 2:
        pts = np.asarray(arc, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, PURPLE, 2, cv2.LINE_AA)
        for p in arc[:: max(1, len(arc) // 12)]:
            _draw_dot(img, p, PURPLE, 2)

    # Pink vertex (parabola) or center (legacy circle)
    vertex = curve.get("vertex") or curve.get("center")
    if vertex:
        _draw_dot(img, vertex, PINK, 5)
        cx, cy = int(vertex[0]), int(vertex[1])
        cv2.drawMarker(img, (cx, cy), PINK, cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)

    # Optional faint circle if radius known (osculating at vertex)
    r_px = curve.get("radius_px")
    if vertex and r_px and curve.get("approach") == "circular_arc":
        cv2.circle(
            img,
            (int(vertex[0]), int(vertex[1])),
            max(3, int(r_px)),
            PURPLE,
            1,
            cv2.LINE_AA,
        )

    tip = curve.get("tip_point") or curve.get("peak_location") or vertex
    if tip and curve.get("radius_nm") is not None:
        px, py = int(tip[0]), int(tip[1])
        ly = py - 8 if index % 2 == 0 else py + 14
        kind = "parab" if curve.get("approach") == "vertex_parabola" else "arc"
        label = f"{kind} R={curve['radius_nm']:.1f}nm"
        cv2.putText(img, label, (px + 6, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.32, PURPLE, 1, cv2.LINE_AA)


def annotate_circular_arc_image(
    image: np.ndarray,
    per_curve: list[dict],
    nm_per_pixel: float,
    config: dict,
    output_path: str | None = None,
) -> np.ndarray:
    """Approach 2 annotated image — purple parabola curves, pink vertices."""
    img = _base_image(image)
    for i, curve in enumerate(per_curve):
        _draw_circular_arc_curve(img, curve, i)
    cv2.putText(
        img,
        "Approach 2: purple=parabola curve  pink=vertex (h,k)",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        PURPLE,
        2,
        cv2.LINE_AA,
    )
    if output_path:
        _save_annotated(img, output_path, nm_per_pixel, config)
    return img


annotate_parabola_image = annotate_circular_arc_image


def _draw_approach3_curve(img: np.ndarray, curve: dict, index: int) -> None:
    """Approach 3: green refined contour + cyan fitted circle + tip marker."""
    contour = curve.get("contour_points") or curve.get("vlm_contour_points") or []
    if len(contour) >= 2:
        pts = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, GREEN, 2, cv2.LINE_AA)

    center = curve.get("center")
    r_px = curve.get("radius_px")
    if center and r_px and float(r_px) > 0:
        cx, cy = int(round(center[0])), int(round(center[1]))
        cv2.circle(img, (cx, cy), max(2, int(round(float(r_px)))), CYAN, 2, cv2.LINE_AA)
        _draw_dot(img, center, CYAN, 3)

    tip = curve.get("tip_point") or curve.get("peak_location")
    if tip:
        _draw_dot(img, tip, YELLOW, 4)
        if curve.get("radius_nm") is not None:
            px, py = int(tip[0]), int(tip[1])
            ly = py - 8 if index % 2 == 0 else py + 14
            conf = curve.get("vlm_confidence")
            conf_s = f" c={conf:.2f}" if isinstance(conf, (int, float)) else ""
            label = f"R={curve['radius_nm']:.1f}nm{conf_s}"
            cv2.putText(
                img, label, (px + 6, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.32, GREEN, 1, cv2.LINE_AA
            )


def annotate_approach3_image(
    image: np.ndarray,
    per_curve: list[dict],
    nm_per_pixel: float,
    config: dict,
    output_path: str | None = None,
) -> np.ndarray:
    """Approach 3 annotated image — OpenAI peaks/contours + fitted circles."""
    img = _base_image(image)
    for i, curve in enumerate(per_curve):
        _draw_approach3_curve(img, curve, i)
    cv2.putText(
        img,
        "Approach 3: yellow=peak  green=contour  cyan=fitted circle (OpenAI+OpenCV)",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        GREEN,
        2,
        cv2.LINE_AA,
    )
    if output_path:
        _save_annotated(img, output_path, nm_per_pixel, config)
    return img


def match_fixed_distance_to_tips(
    tip_curves: list[dict],
    method1_curves: list[dict],
    *,
    max_dist_px: float = 40.0,
) -> list[dict]:
    """Pair OpenAI / Vision tips with nearest fixed-distance Method 1 geometry."""
    return match_geometry_to_tips(
        tip_curves,
        method1_curves,
        max_dist_px=max_dist_px,
        miss_reason="no_fixed_distance_match",
    )


def match_geometry_to_tips(
    tip_curves: list[dict],
    geometry_curves: list[dict],
    *,
    max_dist_px: float = 40.0,
    miss_reason: str = "no_geometry_match",
) -> list[dict]:
    """Pair OpenAI / Vision tips with nearest geometry curves (Method 1/2/3)."""
    matched: list[dict] = []
    for tip_c in tip_curves:
        tip = tip_c.get("tip_point") or tip_c.get("peak_location")
        if not tip or len(tip) < 2:
            continue
        tx, ty = float(tip[0]), float(tip[1])
        best = None
        best_d = float("inf")
        for geo in geometry_curves:
            mt = (
                geo.get("tip_point")
                or geo.get("peak_location")
                or geo.get("ultimate_tip")
            )
            if not mt or len(mt) < 2:
                continue
            d = math.hypot(tx - float(mt[0]), ty - float(mt[1]))
            if d < best_d:
                best_d = d
                best = geo
        if best is not None and best_d <= max_dist_px:
            rec = dict(best)
            rec["peak_id"] = tip_c.get(
                "peak_id", best.get("peak_id") or best.get("tip_id")
            )
            rec["peak_location"] = [tx, ty]
            rec["tip_point"] = [tx, ty]
            rec["openai_tip"] = [tx, ty]
            matched.append(rec)
        else:
            matched.append(
                {
                    "peak_id": tip_c.get("peak_id"),
                    "tip_point": [tx, ty],
                    "peak_location": [tx, ty],
                    "openai_tip": [tx, ty],
                    "valid": False,
                    "rejection_reason": miss_reason,
                }
            )
    return matched


def annotate_approach3_fixed_distance_image(
    image: np.ndarray,
    per_curve: list[dict],
    nm_per_pixel: float,
    config: dict,
    output_path: str | None = None,
) -> np.ndarray:
    """
    Second OpenAI Vision panel: fixed-distance inscribed-circle construction
    (ultimate tip + red l + L/R chord hits + cyan circle) on Vision tip locations.
    Does not modify annotate_approach3_image.
    """
    img = _base_image(image)
    for i, curve in enumerate(per_curve):
        _draw_method1_curve(img, curve, i)
        tip = curve.get("openai_tip") or curve.get("tip_point") or curve.get("peak_location")
        if tip:
            _draw_dot(img, tip, YELLOW, 3)
    cv2.putText(
        img,
        "Fixed-distance circle: blue=tip+L/R hits  red=l+chord  cyan=R  yellow=Vision tip",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        BLUE,
        2,
        cv2.LINE_AA,
    )
    if output_path:
        _save_annotated(img, output_path, nm_per_pixel, config)
    return img


def annotate_approach3_projected_tip_image(
    image: np.ndarray,
    per_curve: list[dict],
    nm_per_pixel: float,
    config: dict,
    output_path: str | None = None,
) -> np.ndarray:
    """
    OpenAI Vision tips + distance from projected tip:
    yellow flank lines → convergent point, red vertical l, blue ultimate tip.
    """
    img = _base_image(image)
    for i, curve in enumerate(per_curve):
        _draw_method2_curve(img, curve, i)
        tip = curve.get("openai_tip") or curve.get("tip_point") or curve.get("peak_location")
        if tip:
            _draw_dot(img, tip, YELLOW, 3)
    cv2.putText(
        img,
        "Projected tip: yellow=edges→converge  red=vertical l  blue=ultimate tip",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        YELLOW,
        2,
        cv2.LINE_AA,
    )
    if output_path:
        _save_annotated(img, output_path, nm_per_pixel, config)
    return img


def annotate_approach3_inscribed_angle_image(
    image: np.ndarray,
    per_curve: list[dict],
    nm_per_pixel: float,
    config: dict,
    output_path: str | None = None,
) -> np.ndarray:
    """
    OpenAI Vision tips + inscribed angle from fixed-diameter circle:
    cyan fixed-D circle, yellow rays from tip through edge hits, θ label.
    """
    img = _base_image(image)
    for i, curve in enumerate(per_curve):
        _draw_method3_curve(img, curve, i)
        tip = curve.get("openai_tip") or curve.get("tip_point") or curve.get("peak_location")
        if tip:
            _draw_dot(img, tip, YELLOW, 3)
    cv2.putText(
        img,
        "Inscribed angle (Interp. A): cyan=D at T  yellow=rays tip→intersections  theta",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        CYAN,
        2,
        cv2.LINE_AA,
    )
    if output_path:
        _save_annotated(img, output_path, nm_per_pixel, config)
    return img


# ── Reference-style d + R composite (WhatsApp / Fraunhofer sketch) ──────────
# Added as a NEW overlay path — does not modify existing drawers above.

REF_ARCH_PURPLE = (200, 60, 220)  # tip arch curve (BGR)


def _nearest_curve(
    tip: list | tuple,
    curves: list[dict],
    *,
    max_dist_px: float = 40.0,
) -> dict | None:
    tx, ty = float(tip[0]), float(tip[1])
    best = None
    best_d = float("inf")
    for c in curves:
        mt = c.get("tip_point") or c.get("peak_location") or c.get("ultimate_tip")
        if not mt or len(mt) < 2:
            continue
        d = math.hypot(tx - float(mt[0]), ty - float(mt[1]))
        if d < best_d:
            best_d = d
            best = c
    if best is None or best_d > max_dist_px:
        return None
    return best


def _synth_alpha_arc(projected, left_line, right_line, d_px: float) -> dict | None:
    """Build α arc dict at projected tip from yellow flank endpoints."""
    if not projected or len(left_line) < 4 or len(right_line) < 4:
        return None
    px, py = float(projected[0]), float(projected[1])

    def _dir_into_body(line):
        x1, y1, x2, y2 = (float(v) for v in line[:4])
        # pick the endpoint farther down (+y) from projected as "into body"
        d1 = (x1 - px, y1 - py)
        d2 = (x2 - px, y2 - py)
        # prefer the direction with larger +y component
        if d1[1] >= d2[1]:
            vx, vy = d1
        else:
            vx, vy = d2
        n = math.hypot(vx, vy) or 1.0
        return vx / n, vy / n

    dl = _dir_into_body(left_line)
    dr = _dir_into_body(right_line)
    a1 = math.degrees(math.atan2(dl[1], dl[0]))
    a2 = math.degrees(math.atan2(dr[1], dr[0]))
    start, end = sorted([a1, a2])
    if end - start > 180:
        start, end = end, start + 360
    arc_r = max(18.0, min(50.0, 0.35 * max(d_px, 8.0) + 12.0))
    return {
        "center": [px, py],
        "radius": float(arc_r),
        "start_deg": float(start),
        "end_deg": float(end),
    }


def build_d_r_reference_curves(
    tip_curves: list[dict],
    projected_curves: list[dict],
    radius_curves: list[dict],
    *,
    max_dist_px: float = 40.0,
) -> list[dict]:
    """
    Merge projected-tip geometry (yellow V + d) with fixed-distance R
    for each Vision tip — data for the reference-style composite PNG.
    """
    out: list[dict] = []
    for tip_c in tip_curves:
        tip = tip_c.get("tip_point") or tip_c.get("peak_location")
        if not tip or len(tip) < 2:
            continue
        tx, ty = float(tip[0]), float(tip[1])
        m2 = _nearest_curve(tip, projected_curves, max_dist_px=max_dist_px)
        m1 = _nearest_curve(tip, radius_curves, max_dist_px=max_dist_px)
        rec: dict = {
            "peak_id": tip_c.get("peak_id"),
            "tip_point": [tx, ty],
            "peak_location": [tx, ty],
            "ultimate_tip": [tx, ty],
            "openai_tip": [tx, ty],
        }
        if m2:
            projected = m2.get("projected_tip") or m2.get("convergence_point")
            rec["projected_tip"] = list(projected) if projected else None
            rec["convergence_point"] = rec["projected_tip"]
            rec["left_line"] = m2.get("left_line") or []
            rec["right_line"] = m2.get("right_line") or []
            rec["tip_apex_arc"] = m2.get("tip_apex_arc") or []
            rec["included_angle_deg"] = m2.get("included_angle_deg")
            d_nm = m2.get("distance_l_nm") or m2.get("d_nm")
            d_px = m2.get("distance_l_px") or m2.get("d_px")
            rec["d_nm"] = float(d_nm) if d_nm is not None else None
            rec["d_px"] = float(d_px) if d_px is not None else None
            rec["distance_l_nm"] = rec["d_nm"]
            if projected:
                rec["d_bracket"] = [
                    float(projected[0]), float(projected[1]), tx, ty,
                ]
                rec["vertical_l_line"] = list(rec["d_bracket"])
            if not rec.get("d_px") and projected:
                rec["d_px"] = abs(ty - float(projected[1]))
            rec["alpha_arc"] = _synth_alpha_arc(
                projected,
                rec["left_line"],
                rec["right_line"],
                float(rec.get("d_px") or 20.0),
            )
        if m1:
            if m1.get("radius_nm") is not None:
                rec["radius_nm"] = float(m1["radius_nm"])
            if m1.get("radius_px") is not None:
                rec["radius_px"] = float(m1["radius_px"])
                rec["circle_radius_px"] = float(m1["radius_px"])
            center = m1.get("center") or m1.get("circle_center")
            if center:
                rec["center"] = [float(center[0]), float(center[1])]
                rec["circle_center"] = list(rec["center"])
                rpx = rec.get("radius_px")
                if rpx:
                    rec["diameter_line"] = [
                        float(center[0]) - float(rpx),
                        float(center[1]),
                        float(center[0]) + float(rpx),
                        float(center[1]),
                    ]
            rec["valid"] = bool(m1.get("valid", True) and rec.get("radius_nm") is not None)
        else:
            rec["valid"] = False
            if not m2:
                rec["rejection_reason"] = "no_projected_or_radius_match"
        out.append(rec)
    return out


def _draw_yellow_alpha_arc(img: np.ndarray, arc: dict) -> None:
    """Yellow α marker at projected tip (matches reference sketch)."""
    if not arc:
        return
    c = arc.get("center")
    r = int(max(8, arc.get("radius", 24)))
    if not c:
        return
    cx, cy = int(c[0]), int(c[1])
    start = float(arc.get("start_deg", 0))
    end = float(arc.get("end_deg", 40))
    cv2.ellipse(img, (cx, cy), (r, r), 0, start, end, YELLOW, 2, cv2.LINE_AA)
    mid = math.radians(0.5 * (start + end))
    lx = int(cx + (r + 8) * math.cos(mid))
    ly = int(cy + (r + 8) * math.sin(mid))
    cv2.putText(img, "a", (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 2, cv2.LINE_AA)


def _draw_d_value_bracket(
    img: np.ndarray,
    projected,
    ultimate,
    d_nm: float | None,
) -> None:
    """Red vertical d from projected tip → ultimate tip with numeric label."""
    if not projected or not ultimate:
        return
    x1, y1 = float(projected[0]), float(projected[1])
    x2, y2 = float(ultimate[0]), float(ultimate[1])
    # Vertical at projected x (as in reference)
    vx = int(round(x1))
    p1 = (vx, int(round(y1)))
    p2 = (vx, int(round(y2)))
    cv2.line(img, p1, p2, RED, 2, cv2.LINE_AA)
    tick = 7
    cv2.line(img, (p1[0] - tick, p1[1]), (p1[0] + tick, p1[1]), RED, 2, cv2.LINE_AA)
    cv2.line(img, (p2[0] - tick, p2[1]), (p2[0] + tick, p2[1]), RED, 2, cv2.LINE_AA)
    # Horizontal tick across ultimate tip
    cv2.line(img, (p2[0] - 12, p2[1]), (p2[0] + 12, p2[1]), RED, 1, cv2.LINE_AA)
    if d_nm is not None:
        label = f"d = {d_nm:.2f} nm"
        lx = p1[0] + 10
        ly = int(0.5 * (p1[1] + p2[1]))
        cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.42, RED, 2, cv2.LINE_AA)


def _draw_d_r_reference_tip(img: np.ndarray, curve: dict, index: int = 0) -> None:
    """
    One tip drawn like the Fraunhofer WhatsApp reference:
      yellow V + α · red d · purple arch · cyan inscribed circle + R
    """
    # Yellow projected flanks
    _draw_line(img, curve.get("left_line", []), YELLOW, 2)
    _draw_line(img, curve.get("right_line", []), YELLOW, 2)

    projected = curve.get("projected_tip") or curve.get("convergence_point")
    ultimate = (
        curve.get("ultimate_tip")
        or curve.get("tip_point")
        or curve.get("peak_location")
    )
    if projected:
        _draw_dot(img, projected, YELLOW, 5)

    # Yellow α at projected tip
    _draw_yellow_alpha_arc(img, curve.get("alpha_arc") or {})

    # Purple physical tip arch
    arc = curve.get("tip_apex_arc") or []
    if len(arc) >= 2:
        _draw_polyline(img, arc, REF_ARCH_PURPLE, 2)

    # Red d
    _draw_d_value_bracket(img, projected, ultimate, curve.get("d_nm"))

    # Cyan inscribed circle + horizontal diameter
    center = curve.get("circle_center") or curve.get("center")
    r_px = curve.get("circle_radius_px") or curve.get("radius_px")
    if center and r_px and float(r_px) > 0:
        cx, cy = int(round(center[0])), int(round(center[1]))
        rr = max(3, int(round(float(r_px))))
        cv2.circle(img, (cx, cy), rr, CYAN, 2, cv2.LINE_AA)
        _draw_dot(img, center, CYAN, 3)
        diam = curve.get("diameter_line")
        if diam and len(diam) >= 4:
            _draw_line(img, diam, CYAN, 1)
        else:
            _draw_line(img, [cx - rr, cy, cx + rr, cy], CYAN, 1)

    # R label (cyan), placed to the side of the tip
    if ultimate and curve.get("radius_nm") is not None:
        px, py = int(ultimate[0]), int(ultimate[1])
        ly = py + 16 if index % 2 == 0 else py + 28
        label = f"R = {curve['radius_nm']:.2f} nm"
        cv2.putText(img, label, (px + 12, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.42, CYAN, 2, cv2.LINE_AA)


def annotate_approach3_d_r_reference_image(
    image: np.ndarray,
    per_curve: list[dict],
    nm_per_pixel: float,
    config: dict,
    output_path: str | None = None,
) -> np.ndarray:
    """
    NEW OpenAI Vision panel matching the annotated SEM reference:
    yellow projected flanks + α, red d, purple tip arch, cyan R circle.
    Does not modify existing annotate_* functions.
    """
    img = _base_image(image)
    for i, curve in enumerate(per_curve):
        _draw_d_r_reference_tip(img, curve, i)
    cv2.putText(
        img,
        "Reference: yellow=V+a  red=d  purple=arch  cyan=R (projected tip + inscribed circle)",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        YELLOW,
        2,
        cv2.LINE_AA,
    )
    if output_path:
        _save_annotated(img, output_path, nm_per_pixel, config)
    return img


def annotate_method2_image(
    image: np.ndarray,
    per_curve: list[dict],
    nm_per_pixel: float,
    config: dict,
    output_path: str | None = None,
) -> np.ndarray:
    """Method 2 annotated image — projected tip distance per curve (PDF slide 3)."""
    img = _base_image(image)
    for i, curve in enumerate(per_curve):
        _draw_method2_curve(img, curve, i)
    cv2.putText(
        img,
        "Method 2: yellow=projected edges → convergent tip  red=vertical l  blue=ultimate tip",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        YELLOW,
        2,
        cv2.LINE_AA,
    )
    if output_path:
        _save_annotated(img, output_path, nm_per_pixel, config)
    return img


def annotate_method3_image(
    image: np.ndarray,
    per_curve: list[dict],
    nm_per_pixel: float,
    config: dict,
    output_path: str | None = None,
) -> np.ndarray:
    """Method 3 annotated image — inscribed angle per curve (PDF slide 4)."""
    img = _base_image(image)
    for i, curve in enumerate(per_curve):
        _draw_method3_curve(img, curve, i)
    cv2.putText(
        img,
        "Method 3: cyan=fixed-D at tip T  yellow=rays T→P_L/P_R  theta=included angle (Interp. A)",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        CYAN,
        2,
        cv2.LINE_AA,
    )
    if output_path:
        _save_annotated(img, output_path, nm_per_pixel, config)
    return img


def annotate_research_image(
    image: np.ndarray,
    per_curve: list[dict],
    nm_per_pixel: float,
    config: dict,
    output_path: str | None = None,
) -> np.ndarray:
    """Research-grade osculating circle annotated image."""
    img = _base_image(image)
    for i, curve in enumerate(per_curve):
        if curve.get("rejected"):
            continue
        _draw_line(img, curve.get("left_line", []), YELLOW, 2)
        _draw_line(img, curve.get("right_line", []), YELLOW, 2)
        _draw_vertical_l(img, curve.get("vertical_l_line", []), "l")

        va = curve.get("virtual_apex")
        if va:
            _draw_dot(img, va, MAGENTA, 4)

        tip = curve.get("physical_tip")
        if tip:
            _draw_dot(img, tip, BLUE, 5)

        center = curve.get("center")
        r_px = curve.get("radius_px")
        if center and r_px:
            cv2.circle(img, (int(center[0]), int(center[1])), max(3, int(r_px)), CYAN, 1, cv2.LINE_AA)

        peak = curve.get("peak_location") or tip
        if peak and curve.get("radius_um") is not None:
            px, py = int(peak[0]), int(peak[1])
            conf = curve.get("confidence_score", 0)
            label = f"R={curve['radius_um']:.2f}um C={conf*100:.0f}%"
            ly = py - 6 if i % 2 == 0 else py + 12
            cv2.putText(img, label, (px + 6, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.3, YELLOW, 1, cv2.LINE_AA)

    if output_path:
        _save_annotated(img, output_path, nm_per_pixel, config)
    return img


def annotate_validated_tips(
    image: np.ndarray,
    tips: list[dict],
    nm_per_pixel: float,
    config: dict,
    output_path: str | None = None,
) -> np.ndarray:
    """Draw hard-valid protocol tip apexes (arch-first pipeline overview)."""
    img = _base_image(image)
    for tip in tips:
        loc = tip.get("peak_location") or [tip.get("apex_x_px"), tip.get("apex_y_px")]
        if not loc or loc[0] is None or loc[1] is None:
            continue
        px, py = float(loc[0]), float(loc[1])
        valid = tip.get("hard_valid", True)
        color = CYAN if valid else RED
        _draw_dot(img, (px, py), color, radius=5)
        tip_id = tip.get("tip_id", tip.get("peak_id", "?"))
        cv2.putText(
            img,
            f"tip {tip_id}",
            (int(px) + 8, int(py) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    if output_path:
        _save_annotated(img, output_path, nm_per_pixel, config)
    return img


def annotate_image(
    image: np.ndarray,
    detections: list[FilteredDetection],
    edge_results: list[EdgePeakResult],
    nm_per_pixel: float,
    config: dict,
    output_path: str | None = None,
    brainstorming_raw: dict | None = None,
    all_radii: list[RadiusResult] | None = None,
    show_secondary_methods: bool = False,
) -> np.ndarray:
    """Render Hough bulk serration curve annotation (cyan circles)."""
    annotated = _base_image(image)

    for edge in edge_results:
        if edge.hough_lines:
            for line in edge.hough_lines:
                if line and len(line) >= 4:
                    x1, y1, x2, y2 = (int(v) for v in line[:4])
                    cv2.line(annotated, (x1, y1), (x2, y2), GREEN, 1, cv2.LINE_AA)

    radii = all_radii or []
    if not radii:
        for det in detections:
            if det.passed:
                radii.extend(det.radius_results)
    _draw_all_serration_curves(annotated, radii)

    if output_path:
        _save_annotated(annotated, output_path, nm_per_pixel, config)

    return annotated
