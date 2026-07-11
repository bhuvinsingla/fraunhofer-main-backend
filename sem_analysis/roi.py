"""Measurement ROI: footer detection and border exclusion."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MeasurementROI:
    """Cropped measurement region with offsets into the original image."""

    image: np.ndarray
    offset_x: int
    offset_y: int
    footer_row: int
    border_margin_px: int
    original_shape: tuple[int, int]

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


def detect_footer_row(image: np.ndarray, min_dark_ratio: float = 0.55) -> int:
    """
    Detect SEM info-bar / footer start row by finding a sustained dark band at the bottom.

    Returns the first row of the footer (exclusive end of measurement ROI),
    or image height if no footer is found.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    h = gray.shape[0]
    if h < 40:
        return h

    # Mean intensity per row (bottom third only)
    start = int(h * 0.55)
    row_means = gray[start:, :].mean(axis=1)
    # Dark footer: mean well below image median
    med = float(np.median(gray))
    thr = med * 0.45 if med > 20 else 40.0

    # Find longest trailing run of dark rows
    dark = row_means < thr
    footer_start_local = len(dark)
    run = 0
    for i in range(len(dark) - 1, -1, -1):
        if dark[i]:
            run += 1
            footer_start_local = i
        else:
            if run >= max(8, int(0.02 * h)):
                break
            run = 0
            footer_start_local = len(dark)

    if run < max(8, int(0.02 * h)):
        # Fallback: fixed ratio crop (~8%)
        return max(1, h - int(h * 0.08))

    return start + footer_start_local


def extract_measurement_roi(
    image: np.ndarray,
    border_margin_px: int = 10,
    footer_row: int | None = None,
) -> MeasurementROI:
    """
    Remove SEM footer and border margin before edge detection.

    Footer is detected when footer_row is None (do not hardcode 548).
    """
    h, w = image.shape[:2]
    if footer_row is None:
        footer_row = detect_footer_row(image)
    footer_row = int(np.clip(footer_row, 1, h))

    m = int(max(0, border_margin_px))
    y0, y1 = m, footer_row - m
    x0, x1 = m, w - m
    if y1 <= y0 + 20 or x1 <= x0 + 20:
        # Degenerate — return almost-full image with tiny margin
        y0, y1, x0, x1 = 1, max(2, footer_row - 1), 1, max(2, w - 1)

    cropped = image[y0:y1, x0:x1].copy()
    return MeasurementROI(
        image=cropped,
        offset_x=x0,
        offset_y=y0,
        footer_row=footer_row,
        border_margin_px=m,
        original_shape=(h, w),
    )


def has_full_measurement_window(
    x: float,
    y: float,
    image_width: int,
    image_height: int,
    radius_px: float,
    safety_px: int = 5,
) -> bool:
    """Reject tips whose required measurement window touches any ROI boundary."""
    required = int(np.ceil(radius_px)) + int(safety_px)
    return (
        x >= required
        and x < image_width - required
        and y >= required
        and y < image_height - required
    )
