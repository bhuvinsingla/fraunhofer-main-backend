"""Shape detection, segmentation, and geometric classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np
from skimage import measure, morphology, segmentation
from skimage.feature import peak_local_max


class ShapeType(str, Enum):
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    POLYGON = "polygon"
    FREEFORM = "freeform"


@dataclass
class DetectedShape:
    """A detected geometric region in the SEM image."""

    shape_id: int
    shape_type: ShapeType
    contour: np.ndarray
    area: float
    perimeter: float
    centroid: tuple[float, float]
    circularity: float
    eccentricity: float
    solidity: float
    major_axis: float
    minor_axis: float
    bbox: tuple[int, int, int, int]  # x, y, w, h
    fitted_center: tuple[float, float] | None = None
    fitted_radius: float | None = None
    metadata: dict = field(default_factory=dict)


def _classify_shape(circularity: float, num_vertices: int, threshold: float) -> ShapeType:
    if circularity >= threshold:
        return ShapeType.CIRCLE
    if num_vertices <= 6 and circularity > 0.5:
        return ShapeType.POLYGON
    if circularity > 0.4:
        return ShapeType.ELLIPSE
    return ShapeType.FREEFORM


def detect_shapes(image: np.ndarray, config: dict) -> list[DetectedShape]:
    """Detect and classify shapes via adaptive thresholding and contour analysis."""
    cfg = config.get("shape_detection", {})
    block_size = cfg.get("adaptive_block_size", 11)
    c = cfg.get("adaptive_c", 2)
    min_area = cfg.get("min_area_px", 50)
    circ_threshold = cfg.get("circularity_threshold", 0.6)

    uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    binary = cv2.adaptiveThreshold(
        uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c
    )

    # Watershed for touching objects
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    local_max = peak_local_max(distance, min_distance=10, labels=binary)
    markers = np.zeros_like(binary, dtype=np.int32)
    for i, (y, x) in enumerate(local_max, start=1):
        markers[y, x] = i
    if local_max.size == 0:
        markers = measure.label(binary)
    else:
        markers = segmentation.watershed(-distance, markers, mask=binary)

    labeled = measure.label(markers)
    regions = measure.regionprops(labeled, intensity_image=image)

    shapes: list[DetectedShape] = []
    shape_id = 0

    for region in regions:
        if region.area < min_area:
            continue

        contour_mask = (labeled == region.label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * region.area / (perimeter**2)
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        shape_type = _classify_shape(circularity, len(approx), circ_threshold)

        y1, x1, y2, x2 = region.bbox
        bbox = (x1, y1, x2 - x1, y2 - y1)

        fitted_center = None
        fitted_radius = None
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            fitted_center = (ellipse[0][0], ellipse[0][1])
            fitted_radius = (ellipse[1][0] + ellipse[1][1]) / 4.0

        shapes.append(
            DetectedShape(
                shape_id=shape_id,
                shape_type=shape_type,
                contour=contour,
                area=float(region.area),
                perimeter=float(perimeter),
                centroid=(region.centroid[1], region.centroid[0]),
                circularity=float(circularity),
                eccentricity=float(region.eccentricity),
                solidity=float(region.solidity),
                major_axis=float(region.major_axis_length),
                minor_axis=float(region.minor_axis_length),
                bbox=bbox,
                fitted_center=fitted_center,
                fitted_radius=fitted_radius,
            )
        )
        shape_id += 1

    # Hough circle validation / supplemental detection
    circles = cv2.HoughCircles(
        uint8,
        cv2.HOUGH_GRADIENT,
        dp=cfg.get("hough_dp", 1.2),
        minDist=cfg.get("hough_min_dist", 20),
        param1=cfg.get("hough_param1", 50),
        param2=cfg.get("hough_param2", 30),
        minRadius=cfg.get("hough_min_radius", 5),
        maxRadius=cfg.get("hough_max_radius", 200),
    )

    if circles is not None:
        for circle in circles[0]:
            cx, cy, r = circle
            # Skip if overlaps existing detection
            overlap = any(
                np.hypot(cx - s.centroid[0], cy - s.centroid[1]) < r * 0.5 for s in shapes
            )
            if overlap:
                continue
            theta = np.linspace(0, 2 * np.pi, 64)
            contour_pts = np.stack(
                [cx + r * np.cos(theta), cy + r * np.sin(theta)], axis=1
            ).astype(np.int32).reshape(-1, 1, 2)
            shapes.append(
                DetectedShape(
                    shape_id=shape_id,
                    shape_type=ShapeType.CIRCLE,
                    contour=contour_pts,
                    area=float(np.pi * r**2),
                    perimeter=float(2 * np.pi * r),
                    centroid=(float(cx), float(cy)),
                    circularity=1.0,
                    eccentricity=0.0,
                    solidity=1.0,
                    major_axis=float(2 * r),
                    minor_axis=float(2 * r),
                    bbox=(int(cx - r), int(cy - r), int(2 * r), int(2 * r)),
                    fitted_center=(float(cx), float(cy)),
                    fitted_radius=float(r),
                    metadata={"source": "hough"},
                )
            )
            shape_id += 1

    return shapes
