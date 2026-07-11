"""Iterate until analyze returns tip points usable by Results dashboard."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=False)

from sem_analysis.pipeline import SEMAnalysisPipeline


def load_cfg() -> dict:
    with open(ROOT / "config" / "default_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("openai_vlm_approach", {})
    cfg["openai_vlm_approach"]["enabled"] = True
    cfg["openai_vlm_approach"]["max_peaks"] = 8
    cfg["openai_vlm_approach"]["timeout_s"] = 90
    cfg["openai_vlm_approach"]["fit_window_px"] = 28.0
    cfg["openai_vlm_approach"]["max_radius_px"] = 80.0
    cfg["openai_vlm_approach"]["max_radius_nm"] = 150.0
    cfg["openai_vlm_approach"]["roi_half_size_px"] = 48
    return cfg


def summarize(result) -> dict:
    bs = result.brainstorming_methods or {}
    m1 = bs.get("fixed_distance_circle") or {}
    m2 = bs.get("projected_tip_distance") or {}
    m3 = bs.get("inscribed_angle") or {}
    a3 = bs.get("approach3_openai_vlm") or {}
    return {
        "ok": True,
        "m1_count": m1.get("count") or len(m1.get("per_curve") or []),
        "m1_median": m1.get("median_radius_nm") or m1.get("median"),
        "m2_count": m2.get("count") or len(m2.get("per_curve") or []),
        "m3_count": m3.get("count") or len(m3.get("per_curve") or []),
        "a3_count": a3.get("count") or len(a3.get("per_curve") or []),
        "a3_peaks": a3.get("peak_count"),
        "a3_mean": a3.get("mean_radius_nm") or a3.get("mean"),
        "a3_openai": (a3.get("openai") or {}),
        "a3_failed": len(a3.get("failed_curves") or []),
        "a3_sample": [
            {
                "peak_id": c.get("peak_id"),
                "xy": c.get("peak_location"),
                "R_nm": c.get("radius_nm"),
                "fit": c.get("fit_method"),
            }
            for c in (a3.get("per_curve") or [])
        ],
        "m1_sample": [
            {
                "peak_id": c.get("peak_id"),
                "xy": c.get("peak_location") or c.get("tip_point"),
                "R_nm": c.get("radius_nm"),
            }
            for c in (m1.get("per_curve") or [])
        ],
        "files": {
            k: bool(v)
            for k, v in (result.annotated_method_paths or {}).items()
            if "method" in k or "approach" in k
        },
    }


def _xy_list(samples: list[dict]) -> list[tuple[float, float]]:
    pts = []
    for s in samples:
        xy = s.get("xy") or [None, None]
        if xy[0] is not None and xy[1] is not None:
            pts.append((float(xy[0]), float(xy[1])))
    return pts


def _tips_colocated(m1: list[tuple[float, float]], a3: list[tuple[float, float]],
                    tol: float = 120.0) -> bool:
    """Every Approach-3 tip should have a nearby Method-1 tip (nearest-neighbor match)."""
    if not m1 or not a3:
        return False
    for ax, ay in a3:
        nearest = min(((ax - mx) ** 2 + (ay - my) ** 2) ** 0.5 for mx, my in m1)
        if nearest > tol:
            print(f"SPATIAL MISS: a3=({ax:.0f},{ay:.0f}) nearest_m1={nearest:.0f}px", flush=True)
            return False
    return True


def success(s: dict) -> bool:
    """Method 1 + Approach 3 points present, co-located, and R in a plausible tip range."""
    if int(s.get("m1_count") or 0) < 1 or int(s.get("a3_count") or 0) < 1:
        return False
    src = ((s.get("a3_openai") or {}).get("peak_source") or "")
    if src not in (
        "method1_tips",
        "openai_snapped_cv",
        "openai_snapped_cv+cv_fill",
        "cv_ridge_fallback",
        "openai",
    ):
        return False
    m1_pts = _xy_list(s.get("m1_sample") or [])
    a3_pts = _xy_list(s.get("a3_sample") or [])
    if not m1_pts or not a3_pts:
        return False
    if not _tips_colocated(m1_pts, a3_pts):
        return False
    m1r = s.get("m1_median")
    a3r = s.get("a3_mean")
    if m1r and a3r:
        # Approach 3 R should be tip-scale, not flank-scale (was ~1000 nm)
        if float(a3r) > 150 or float(a3r) < 1:
            print(f"RADIUS MISS: a3_mean={a3r} m1_median={m1r}", flush=True)
            return False
        if float(a3r) > max(150.0, float(m1r) * 6.0):
            print(f"RADIUS MISS vs M1: a3_mean={a3r} m1_median={m1r}", flush=True)
            return False
    return True


def main() -> int:
    image = ROOT / "uploads" / "01a1920fbdc1" / "PFAS_K05_MP1_02.tif"
    if not image.exists():
        image = ROOT / "data" / "sample" / "synthetic_tip.png"
    out = ROOT / "uploads" / "_loop_test"
    out.mkdir(parents=True, exist_ok=True)

    # Require two consecutive successes so flaky VLM empty-peak runs don't pass by luck
    max_attempts = 4
    streak = 0
    last = None
    for attempt in range(1, max_attempts + 1):
        print(f"\n===== ATTEMPT {attempt}/{max_attempts} image={image.name} =====", flush=True)
        cfg = load_cfg()
        pipe = SEMAnalysisPipeline(cfg)
        try:
            result = pipe.analyze(str(image), str(out))
            s = summarize(result)
            last = s
            print(json.dumps(s, indent=2, default=str), flush=True)
            if success(s):
                streak += 1
                print(f"OK ({streak}/2 consecutive)", flush=True)
                if streak >= 2:
                    print("SUCCESS: Method 1 + Approach 3 have points (2x)", flush=True)
                    return 0
            else:
                streak = 0
                print("INCOMPLETE: need m1_count>=1 and a3_count>=1", flush=True)
        except Exception as exc:
            streak = 0
            last = {"ok": False, "error": str(exc)}
            print("FAILED:", exc, flush=True)
            traceback.print_exc()
            (out / "last_error.txt").write_text(str(exc), encoding="utf-8")

    print("\nGIVING UP after loop. last=", json.dumps(last, indent=2, default=str), flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
