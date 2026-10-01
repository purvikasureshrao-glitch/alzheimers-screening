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

11 zip archives, 507 MB total, in `datasets/NHATS/`. The image zips hold no labels,
but the **public NHATS Sample Person (SP) survey files do** — checked 2026-10-01 with
`scripts/check_nhats_labels.py` (aggregate output only).

| Check | Result |
|---|---|
| Total clock images, rounds 1-11 | **54,027 — exact match to Bone et al.** |
| Image filename = SPID | 54,026 plain 8-digit; 1 with a `_V<n>` suffix |
| `cg<r>dclkdraw` (clock 0-5) | Present in SP file, rounds 1-10 |
| `hc<r>disescn9` (dementia) | Present in SP file, rounds 1-10 |
| Image to survey join on SPID | 99.8-100% per round (24 unmatched images in total, all in R1-R3) |
| Round 11 survey | **Not yet downloaded** — 3,091 R11 images unlabelled until it is |

Value codes seen: clock 0-5 plus negative special codes (-1, -2, -3, -4, -7, -9);
dementia 1 = yes, 2 = no, 7 = previously reported (from R2 on), plus negative codes.
Per Bone et al., target = 1 for codes 1 and 7, else 0. Rows with negative codes are
excluded; exact meanings to be confirmed against the NHATS codebook.

Dementia prevalence (codes 1 + 7) is roughly 6-8% of SP rows per round: heavy class
imbalance, which supports balanced accuracy as the headline metric.

**Hand annotation is not needed.** The single-rater / no-kappa limitation is withdrawn.

---

## 4. Open items

- Eye-tracker hardware model: unknown, not published.
- Clock labels: resolved — public NHATS SP files (see section 3). Round 11 survey
  still to download.
- Clock split: the same SPID appears in many rounds, so train/validation must be
  split **by SPID**, not by image, or the same person leaks into both sides.
- Motor labels: none. Module reports features only until labelled data exists.
