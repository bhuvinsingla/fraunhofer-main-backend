"""Validate 1D ridge asperity peak detection on a SEM grating / blade image.

Usage:
  python test_grating.py path/to/PFAS_K05_MP1_02.tif
  python test_grating.py   # searches common locations

Writes annotated outputs under ./test_output/
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sem_analysis.edge_detection import detect_serration_peaks_global
from sem_analysis.io.image_loader import load_image
from sem_analysis.pipeline import SEMAnalysisPipeline, load_config
from sem_analysis.preprocessing import preprocess


def _find_default_image() -> Path | None:
    candidates = [
        ROOT / "PFAS_K05_MP1_02.tif",
        ROOT / "data" / "PFAS_K05_MP1_02.tif",
        ROOT / "data" / "sample" / "PFAS_K05_MP1_02.tif",
        ROOT / "uploads" / "PFAS_K05_MP1_02.tif",
    ]
    # Recent uploads under uploads/<job>/
    uploads = ROOT / "uploads"
    if uploads.is_dir():
        candidates.extend(sorted(uploads.rglob("PFAS_K05_MP1_02.tif"), reverse=True)[:5])
        candidates.extend(sorted(uploads.rglob("*.tif"), reverse=True)[:5])
    for p in candidates:
        if p.is_file():
            return p
    return None


def _annotate_peaks(image: np.ndarray, peaks: np.ndarray, out_path: Path) -> None:
    if image.dtype != np.uint8:
        vis = (np.clip(image, 0, 1) * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    else:
        vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    for i, (x, y) in enumerate(peaks):
        cv2.circle(vis, (int(round(x)), int(round(y))), 4, (0, 255, 0), -1)
        if i % 5 == 0:
            cv2.putText(
                vis,
                str(i),
                (int(round(x)) + 5, int(round(y)) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )
    cv2.putText(
        vis,
        f"1D ridge peaks: {len(peaks)}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (50, 220, 50),
        2,
        cv2.LINE_AA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    image_path = Path(argv[0]) if argv else _find_default_image()
    if image_path is None or not image_path.is_file():
        print("Usage: python test_grating.py <image.tif>")
        print("No PFAS_K05_MP1_02.tif found. Pass the path explicitly.")
        return 1

    out_dir = ROOT / "test_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    # Ensure legacy global peak path is available for pipeline annotations
    config.setdefault("pipeline", {})["legacy_peak_detection"] = True

    print(f"Image: {image_path}")
    sem = load_image(image_path, config=config)
    processed = preprocess(sem.data, sem.nm_per_pixel, config)

    edge = detect_serration_peaks_global(processed.data, config)
    n = len(edge.peak_locations)
    print(f"1D ridge peaks: {n}")
    print(f"  ridge meta: {edge.metadata.get('ridge')}")
    print(f"  combined_count: {edge.metadata.get('combined_count')}")

    ann_path = out_dir / f"{image_path.stem}_annotated.png"
    _annotate_peaks(processed.data, edge.peak_locations, ann_path)
    print(f"Wrote {ann_path}")

    # Full pipeline → method1–3 overlays
    pipeline = SEMAnalysisPipeline(config)
    result = pipeline.analyze(image_path, output_dir=out_dir)
    print(f"Pipeline tip_condition={result.tip_condition} nm/px={result.nm_per_pixel}")
    for key, path in (result.annotated_method_paths or {}).items():
        if path:
            print(f"  {key}: {path}")

    if n < 5:
        print("WARNING: few peaks detected — tune ridge_peak_prominence / ridge_peak_distance / spline_smoothing")
        return 2
    print("OK — 1D ridge asperity detection completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
