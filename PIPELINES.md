# SEM Tip Radius — Pipelines & Approaches

This file is the reference map for what runs on each analyze job, what each stage is for, and which code owns it.

---

## End-to-end flow

```
SEM image
   │
   ├─ Load + scale (nm/px) + optional footer OCR
   ├─ Preprocess (CLAHE, bilateral denoise)
   │
   ├─ Tip detection (find apexes)  ← tip_detection.mode
   │      ridge | qwen | arch
   │
   ├─ Measurement methods (PDF brainstorming) — same tip IDs
   │      Method 1 → tip radius R
   │      Method 2 → projected tip distance l
   │      Method 3 → included angle θ
   │
   └─ Extra radius approaches (parallel)
          Approach 1 = Method 1 (CV tips)
          Approach 2 = parabola osculating R
          Approach 3 = OpenAI peaks → contour → circle fit
```

Config: `config/default_config.yaml`  
Orchestration: `sem_analysis/pipeline.py`  
API: `app.py` → `/api/analyze`

---

## Tip detection (find apexes)

| Mode | Config | Uses | Task | Code |
|------|--------|------|------|------|
| **ridge** (default) | `tip_detection.mode: ridge` | Skyline → spline residual → `find_peaks` | Locate all V-apex candidates along the blade | `edge_detection.detect_serration_peaks_global` |
| **qwen** | `tip_detection.mode: qwen` | Local Qwen2.5-VL → snap to CV peaks | Same apex task, VLM-assisted | `qwen_peaks.py` + CV ridge |
| **arch** | `tip_detection.mode: arch` | Complete-arch validation | Stricter tip windows only | `arch_detection.py` |

---

## Measurement methods (PDF brainstorming)

Source: *Tip Radius Measurement Brainstorming* (Mammoth).  
Run on every accepted tip from tip detection. Different geometric questions — not three ways to compute the same R.

### Method 1 — Fixed distance inscribed circle

| | |
|--|--|
| **Task** | Tip radius |
| **Procedure** | Apex (top blue) → drop fixed vertical **l** → horizontal red chord → left/right blue hits → circumcircle through 3 points → **R** |
| **Default l** | `protocol.method1_primary_nm` (50 nm) |
| **Output** | `radius_nm` (small R = sharper) |
| **Image / CSV** | `*_method1.png`, `*_method1_radii.csv` |
| **API key** | `brainstorming_methods.fixed_distance_circle` |
| **Code** | `methods/fixed_distance_circle.py` |

### Method 2 — Distance from projected tip

| | |
|--|--|
| **Task** | How far the real tip sits below the virtual intersection of the flanks |
| **Procedure** | Fit yellow flank lines → convergent point **above** tip → vertical red **l** down to ultimate tip (blue curve) → length of red line |
| **Fit band** | `protocol.method2_fit_band_nm` [50, 200] nm (with fallback bands) |
| **Output** | `distance_l_nm` (vertical image distance × nm/px) |
| **Image / CSV** | `*_method2.png`, `*_method2_radii.csv` |
| **API key** | `brainstorming_methods.projected_tip_distance` |
| **Code** | `methods/projected_tip_distance.py` |

### Method 3 — Inscribed angle from fixed diameter circle

| | |
|--|--|
| **Task** | Opening angle of the tip |
| **Procedure** | Cyan circle of diameter **D** at apex → yellow rays tip→edge intersections → included angle **θ** |
| **Default D** | `protocol.method3_circle_diameter_nm` (100 nm) |
| **Output** | `angle_degrees` |
| **Image / CSV** | `*_method3.png`, `*_method3_radii.csv` |
| **API key** | `brainstorming_methods.inscribed_angle` |
| **Code** | `methods/inscribed_angle.py` |

---

## Extra approaches (tip-radius strategies)

Approaches answer: **“What is the tip radius?”** with different tip-finding or fitting strategies.

### Approach 1 — CV Method 1 (primary)

| | |
|--|--|
| **Task** | Primary tip radius (PDF Method 1) |
| **Uses** | Ridge (or qwen/arch) apexes + fixed-distance inscribed circle |
| **Same as** | Measurement Method 1 |
| **API key** | `brainstorming_methods.fixed_distance_circle` |
| **UI** | Method 1 panel |

### Approach 2 — Vertex-form parabola

| | |
|--|--|
| **Task** | Osculating radius at tip from parabola fit (not PDF Method 1) |
| **Uses** | Sliding-window fit `y = a(x − h)² + k` on skyline; `R = 1/(2|a|)` |
| **Config** | `parabola_approach.*` |
| **Image / CSV** | `*_method1_approach2.png`, `*_method1_approach2_radii.csv` |
| **API key** | `brainstorming_methods.approach2_parabolas` |
| **Code** | `methods/parabola_approach2.py` |

### Approach 3 — OpenAI Vision + circle fit

| | |
|--|--|
| **Task** | Tip radius via VLM peaks/contours + classical circle fit (OpenAI replaces Gemini) |
| **Flow** | Preprocess → OpenAI peaks → ROI crop → OpenAI apex contour → OpenCV refine → Pratt/Taubin/LS circle → nm → stats |
| **Config** | `openai_vlm_approach.*` (`circle_fit`: `pratt` \| `taubin` \| `least_squares`) |
| **Env** | `OPENAI_API_KEY`, optional `OPENAI_VLM_MODEL` |
| **Image / CSV** | `*_method1_approach3.png`, `*_method1_approach3_radii.csv` |
| **API key** | `brainstorming_methods.approach3_openai_vlm` |
| **Code** | `methods/openai_vlm_approach3.py` |
| **Stats** | Mean / std / peak count (headline = mean) |

---

## What to use for which decision

| Question | Use |
|----------|-----|
| Tip radius as in the PDF diagram | **Method 1 / Approach 1** |
| Tip radius with AI peaks + contour fit | **Approach 3** (OpenAI) |
| Tip curvature from parabola | **Approach 2** |
| How far tip sits from projected intersection | **Method 2** |
| Opening angle at fixed D | **Method 3** |
| Footer / scale text from SEM bar | OpenAI OCR (`io/openai_ocr.py`) — not geometry |

---

## UI mapping (`ResultsDashboard.jsx`)

| Panel | Backend key | File image key |
|-------|-------------|----------------|
| Fixed distance inscribed circle | `fixed_distance_circle` | `method1` |
| Distance from projected tip | `projected_tip_distance` | `method2` |
| Inscribed angle | `inscribed_angle` | `method3` |
| Approach 3 — OpenAI peaks + contour → circle fit | `approach3_openai_vlm` | `method1_approach3` |

Approach 2 (parabola) still runs in the backend; it is not on the primary dashboard panels by default.

---

## Related modules

| Area | Path |
|------|------|
| Tip orchestration | `sem_analysis/tip_measurement.py` |
| Overlays | `sem_analysis/annotation.py` |
| Protocol constants (l, D, fit band) | `sem_analysis/protocol.py` |
| Scale / Zeiss / OCR | `sem_analysis/io/` |
| Default knobs | `config/default_config.yaml` |
