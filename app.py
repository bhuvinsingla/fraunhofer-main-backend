"""SEM tip radius API backend (port 8000)."""

from __future__ import annotations

import logging
import os
import shutil
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from sem_analysis.io.image_loader import ResizedImageError
from sem_analysis.pipeline import SEMAnalysisPipeline
from sem_analysis.protocol import apply_protocol_overrides, get_protocol, parse_protocol_form

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sem-api")

BACKEND_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = BACKEND_ROOT / "uploads"
SAMPLE_IMAGE = BACKEND_ROOT / "data" / "sample" / "synthetic_tip.png"
SAMPLE_GROUND_TRUTH = BACKEND_ROOT / "data" / "sample" / "synthetic_tip_ground_truth.csv"
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

app = FastAPI(title="SEM Tip Radius API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = SEMAnalysisPipeline()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    log.info("→ %s %s", request.method, request.url.path)
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    log.info("← %s %s %d (%.0fms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


def _validate_extension(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        log.warning("Rejected file type: %s", ext)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Use PNG, JPG, or TIFF.",
        )


def _job_dir(job_id: str) -> Path:
    path = UPLOAD_DIR / job_id
    path.mkdir(parents=True, exist_ok=True)
    log.info("Job %s → %s", job_id, path)
    return path


def _serialize_result(result, job_id: str) -> dict:
    data = result.to_dict()
    stem = Path(result.source_path).stem
    base = f"{API_BASE_URL}/api/jobs/{job_id}/files"

    data["job_id"] = job_id
    data["files"] = {
        "annotated": f"{base}/{stem}_annotated.png",
        "report": f"{base}/{stem}_report.json",
        "radii": f"{base}/{stem}_radii.csv",
        "tips": f"{base}/{stem}_tips.csv",
        "method1": f"{base}/{stem}_method1.png",
        "method2": f"{base}/{stem}_method2.png",
        "method3": f"{base}/{stem}_method3.png",
        "whiteboard": f"{base}/{stem}_whiteboard.png",
        "method1_csv": f"{base}/{stem}_method1_radii.csv",
        "method2_csv": f"{base}/{stem}_method2_radii.csv",
        "method3_csv": f"{base}/{stem}_method3_radii.csv",
        "blade_value_csv": f"{base}/{stem}_blade_value.csv",
        "research": f"{base}/{stem}_research.png",
        "research_csv": f"{base}/{stem}_research_radii.csv",
    }
    if result.validation:
        data["files"]["validation"] = f"{base}/{stem}_validation.csv"
        data["files"]["validation_plot"] = f"{base}/{stem}_validation_plot.png"
    return data


def _log_result_summary(job_id: str, result) -> None:
    agg = result.aggregation or {}
    mean_r = agg.get("mean_radius_nm")
    log.info(
        "Job %s complete | shapes %d/%d | peaks %d | mean radius %s nm | tip %s",
        job_id,
        result.shapes_passed,
        result.shapes_detected,
        len(result.radius_results),
        f"{mean_r:.2f}" if mean_r is not None else "n/a",
        result.tip_condition or "n/a",
    )


def _run_analysis(
    image_path: Path,
    output_dir: Path,
    ground_truth_path: Path | None = None,
    protocol_overrides: dict | None = None,
) -> dict:
    job_id = output_dir.name
    log.info("Job %s | starting pipeline on %s", job_id, image_path.name)
    if ground_truth_path:
        log.info("Job %s | ground truth: %s", job_id, ground_truth_path.name)

    cfg = apply_protocol_overrides(pipeline.config, protocol_overrides)
    if protocol_overrides:
        log.info("Job %s | protocol overrides: %s", job_id, protocol_overrides)
    job_pipeline = SEMAnalysisPipeline(config=cfg)

    t0 = time.perf_counter()
    result = job_pipeline.analyze(
        image_path,
        output_dir=output_dir,
        ground_truth_path=ground_truth_path,
    )
    elapsed = time.perf_counter() - t0
    log.info("Job %s | pipeline finished in %.1fs", job_id, elapsed)
    _log_result_summary(job_id, result)
    return _serialize_result(result, job_id)


@app.on_event("startup")
def on_startup() -> None:
    log.info("SEM Tip Radius API starting on port %s", os.getenv("API_PORT", "8000"))
    log.info("Upload dir: %s", UPLOAD_DIR)
    log.info("Sample image: %s (%s)", SAMPLE_IMAGE, "found" if SAMPLE_IMAGE.exists() else "missing")


@app.get("/api/health")
def health() -> dict:
    log.info("Health check OK")
    return {"status": "ok", "service": "sem-tip-radius-api", "port": 8000}


@app.get("/api/protocol")
def protocol_defaults() -> dict:
    """Proposed measurement protocol (pending client approval of l, fit band, D)."""
    return {
        "protocol": get_protocol(pipeline.config),
        "message": (
            "Approve Method 1 fixed distances (l), Method 2 fitting range, "
            "and Method 3 circle diameter before treating results as a lab standard."
        ),
        "docs": "docs/CALCULATION_BASIS.md",
    }


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    ground_truth: UploadFile | None = File(None),
    calibration: UploadFile | None = File(None),
    method1_distances_nm: str | None = None,
    method1_primary_nm: float | None = None,
    method2_fit_lo_nm: float | None = None,
    method2_fit_hi_nm: float | None = None,
    method3_circle_diameter_nm: float | None = None,
    protocol_approved: bool | None = None,
    protocol_approved_by: str | None = None,
) -> dict:
    if not image.filename:
        log.warning("Analyze rejected: no image provided")
        raise HTTPException(status_code=400, detail="Image file is required.")

    _validate_extension(image.filename)

    job_id = uuid.uuid4().hex[:12]
    job_path = _job_dir(job_id)

    image_path = job_path / image.filename
    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)
    log.info("Job %s | saved image: %s (%.1f KB)", job_id, image.filename, image_path.stat().st_size / 1024)

    if calibration and calibration.filename:
        cal_ext = Path(calibration.filename).suffix.lower()
        if cal_ext not in {".json", ".csv"}:
            raise HTTPException(status_code=400, detail="Calibration must be JSON or CSV.")
        cal_path = job_path / f"{image_path.stem}_calibration{cal_ext}"
        with open(cal_path, "wb") as f:
            shutil.copyfileobj(calibration.file, f)
        log.info("Job %s | saved calibration: %s", job_id, calibration.filename)

    gt_path = None
    if ground_truth and ground_truth.filename:
        gt_ext = Path(ground_truth.filename).suffix.lower()
        if gt_ext not in {".json", ".csv"}:
            raise HTTPException(status_code=400, detail="Ground truth must be JSON or CSV.")
        gt_path = job_path / ground_truth.filename
        with open(gt_path, "wb") as f:
            shutil.copyfileobj(ground_truth.file, f)
        log.info("Job %s | saved ground truth: %s", job_id, ground_truth.filename)

    try:
        overrides = parse_protocol_form(
            method1_distances_nm=method1_distances_nm,
            method1_primary_nm=method1_primary_nm,
            method2_fit_lo_nm=method2_fit_lo_nm,
            method2_fit_hi_nm=method2_fit_hi_nm,
            method3_circle_diameter_nm=method3_circle_diameter_nm,
            protocol_approved=protocol_approved,
            protocol_approved_by=protocol_approved_by,
        )
        return _run_analysis(image_path, job_path, gt_path, overrides or None)
    except ResizedImageError as exc:
        log.warning("Job %s | resized image rejected: %s", job_id, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Job %s | analysis failed: %s", job_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/analyze/sample")
def analyze_sample() -> dict:
    log.info("Sample analysis requested")
    if not SAMPLE_IMAGE.exists():
        log.error("Sample image missing at %s", SAMPLE_IMAGE)
        raise HTTPException(
            status_code=404,
            detail="Sample image not found. Run: python -m sem_analysis.utils.sample_generator",
        )

    job_id = uuid.uuid4().hex[:12]
    job_path = _job_dir(job_id)
    image_path = job_path / SAMPLE_IMAGE.name
    shutil.copy2(SAMPLE_IMAGE, image_path)
    log.info("Job %s | copied sample image", job_id)

    cal_src = SAMPLE_IMAGE.parent / f"{SAMPLE_IMAGE.stem}_calibration.json"
    if cal_src.exists():
        shutil.copy2(cal_src, job_path / cal_src.name)
        log.info("Job %s | copied sample calibration", job_id)

    gt_path = SAMPLE_GROUND_TRUTH if SAMPLE_GROUND_TRUTH.exists() else None
    if gt_path:
        shutil.copy2(gt_path, job_path / gt_path.name)
        log.info("Job %s | copied sample ground truth", job_id)

    try:
        return _run_analysis(image_path, job_path, gt_path)
    except Exception as exc:
        log.exception("Job %s | sample analysis failed: %s", job_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/files/{filename}")
def get_job_file(job_id: str, filename: str) -> FileResponse:
    file_path = UPLOAD_DIR / job_id / filename
    if not file_path.exists() or not file_path.is_file():
        log.warning("File not found: job=%s file=%s", job_id, filename)
        raise HTTPException(status_code=404, detail="File not found.")
    log.info("Serving file: job=%s file=%s (%.1f KB)", job_id, filename, file_path.stat().st_size / 1024)
    return FileResponse(file_path)


def main() -> None:
    import uvicorn

    log.info("Launching uvicorn…")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
        reload_dirs=[str(BACKEND_ROOT)],
        log_level="info",
    )


if __name__ == "__main__":
    main()
