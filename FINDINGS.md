# Verified Findings

Facts measured directly from the data and source references. Nothing here is assumed.
Date: 2026-09-30

---

## 1. Eye — ADEM_TEST (measured from disk)

Scanned all 240 gaze files (20 subjects x 12 tasks).

| Property | Measured value |
|---|---|
| Subjects | 10 AD + 10 NC = 20 |
| Files | 240 (12 tasks per subject) |
| File format | 5 CSV columns: `timestamp_ms, x_left, y_left, x_right, y_right` |
| Samples per file | min 8 / median 107 / max 341 |
| Recording span per task | min 157 ms / **median 297 ms** / max 434 ms |
| Inter-sample interval | min 1.0 ms / **median 3.0 ms** / max 8.5 ms |
| **Implied sampling rate** | **~333 Hz** |
| Total gaze per subject | **~3.6 seconds** across all 12 tasks |

**Timestamp integrity:** 4 files have one non-increasing timestamp step each, all in
`NC/220808_zjr_m` (`antisaccade1`, `antisaccade2`, `presaccade1`, `presaccade2`).
Loader must handle this explicitly.

**Not stated by the source:** the Zenodo record does not name the eye-tracker model or
its specified sampling rate. The ~333 Hz above is derived from timestamps, not quoted
from documentation.

### Compatibility verdict: webcam deployment

| | Research data | Our deployment |
|---|---|---|
| Sensor | Binocular eye tracker | Webcam + MediaPipe + WebGazer.js |
| Rate | ~333 Hz | ~30 Hz |

Gap is ~11x. At 30 Hz a 297 ms task yields roughly **9 gaze points**. A saccade lasts
30-80 ms, so a webcam captures 1-2 samples inside one.

**Saccade-level features are not reproducible on webcam.** This is a sensor limit, not
an implementation problem. Coarse features (dwell location, dwell duration, rough scan
path) may transfer and must be validated separately before any claim is made.

---

## 2. Motor — Koppelmans et al. 2023 (read from the paper)

Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC10578230/
Reference implementation: https://github.com/vnckppl/FingerTappingTask (PsychoPy)

| Property | Value as published |
|---|---|
| Sample | 100 total: 47 cognitively normal, 27 aMCI, 26 AD |
| Mean age | 74.5 +/- 5.9; 58% female; 98% white |
| Conditions | 4: left index, right index, simultaneous dual, alternating |
| Trials | 3 per condition, **10 s each** |
| Input device | Keyboard **shift keys** |
| Cue | "Get Ready!" for 3 s, then "Start Pressing!" |
| Features | 20 continuous metrics + tapping-gap binary measures |
| Model | Balanced Random Forest |

Named features: initial reaction time, taps per second, taps-per-second variance,
unpaired taps rate (dual), tap pair duration and variance (dual), error rate and
transition metrics (alternating), and tapping gaps (pauses >= 1 s).

**Reported performance — the ceiling we would be reproducing:**
- Overall accuracy **70%**, multi-class AUC **0.76**
- Precision: controls 76%, MCI 100%, **AD 45%**

AD precision of 45% means the published model is wrong more often than right on AD.
No implementation of ours can exceed this by reproducing their method.

**Their stated limitations:** sample size modest for machine learning; 98% white
sample limits generalizability; no tau biomarker data; ~30-week gap between imaging
and behavioural assessment; the finger tapping test itself is unvalidated and
keyboard-only, with no pressure measurement.

---

## 3. Clock — NHATS (verified on disk)

11 zip archives, 507 MB total, in `datasets/NHATS/`. **Images only — no label file
is present in any archive.** This confirms the flagged deviation in PROJECT_OUTLINE.md:
training labels must come from our own annotation, not from the paper's source.

---

## 4. Open items

- Eye-tracker hardware model: unknown, not published.
- Clock labels: to be produced by single-rater annotation. Inter-rater agreement
  (Cohen's kappa) **cannot be computed** with one rater and must not be claimed.
- Motor labels: none. Module reports features only until labelled data exists.
