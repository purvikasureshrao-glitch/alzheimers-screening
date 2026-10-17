# Eye modality — end to end

Status: **pilot run, inconclusive, no model file saved, not validated for webcam.**
Method is our own [OURS] (PLAN.md B2): no usable paper exists (FINDINGS 6).

## 1. What it does

AD vs NC (2 classes) from short gaze recordings made with a research eye tracker.
Output is **flagged / not flagged** only. No stage, no severity (CLAUDE.md rule 3).

## 2. Data

ADEM_TEST: 20 subjects (10 AD, 10 NC), 12 tasks each, ~333 Hz binocular
(FINDINGS 1). Each task file has 5 columns: `timestamp_ms, x_left, y_left, x_right, y_right`.
Each recording lasts only ~0.3 s (median 297 ms). Only 3 tasks are used, fixed before
any result was seen: `antisaccade1`, `visual_attention1`, `visual_search1`.

## 3. Pipeline (code: `features.py`, `cv.py`)

1. **Load** (`load_gaze`): `-100` is treated as missing; rows whose timestamp does not
   increase are dropped (4 files in `NC/220808_zjr_m`).
2. **Cyclopean position**: mean of the eyes that are valid at each sample.
3. **Velocity** (px/ms) between consecutive valid samples. Screen geometry is
   unpublished, so there are no degrees.
4. **Event detection**: adaptive velocity threshold after Nystrom & Holmqvist (2010),
   mean + 6 SD of the velocities below the threshold, iterated to convergence.
   Above threshold = saccade; below = fixation (kept only if >= 40 ms).
   [OURS] deviation: iteration starts at the maximum velocity, not 100 deg/s.
5. **10 features per task** (30 in total): fixation count, mean fixation duration,
   saccade count, mean saccade amplitude, peak saccade velocity, time to first saccade,
   L/R disconjugacy x and y, gaze spread x and y.
   Tracker dropout (`missing_frac`) is reported but **not** given to the model, because
   it differs by group and is a data-quality confound.
6. **Classifier** (`make_model`): median imputer, standard scaler, logistic regression
   (sklearn defaults, no tuning), cut-off 0.5. Imputer and scaler are fitted on training
   subjects only.
7. **Evaluation** (`loso`): leave-one-subject-out, 20 folds, train 19 / test 1.
   95% CI by bootstrap over subjects, 2000 resamples (`common/metrics.py`).

Run it with `notebooks/eye.ipynb` on Colab (uploads: `eye_code.tar`, `eye_data.tar`).

## 4. Results (Colab, 2026-10-12; FINDINGS 8b)

n = 20, LOSO, logistic regression, cut-off 0.5.

| Input | Balanced accuracy (95% CI) | AUC (95% CI) |
|---|---|---|
| all three tasks | 0.300 (0.101-0.505) | 0.240 (0.000-0.510) |
| antisaccade1 | 0.300 (0.107-0.521) | 0.310 (0.090-0.593) |
| visual_attention1 | 0.450 (0.227-0.667) | 0.300 (0.081-0.582) |
| visual_search1 | 0.400 (0.192-0.625) | 0.340 (0.101-0.615) |

**Read this as: no detectable signal.** Every CI includes 0.5 (chance). It does not mean
"30% accurate", and the AUC below 0.5 is the known leave-one-out artefact on tiny balanced
groups, not a reversed signal. Do not compare with other papers' figures (CLAUDE.md rule 6).

## 5. Why the result is weak (measured)

- The adaptive threshold exceeds a recording's own maximum velocity in 28 of 60
  recordings (47%), so those have 0 saccades by construction. Cause: the published
  rule applied to ~0.3 s recordings. Saccade features are then imputed.
- Timestamps are integer ms at ~3 ms spacing, so velocity has about +/-30% quantisation noise.
- Tracker dropout is higher in AD (antisaccade1: 34% vs 15% of samples).
- L/R disconjugacy (150-360 px) is mostly tracker calibration offset.
- Antisaccade direction errors cannot be computed: target side is not in the files.
- n = 20 and 30 features.

Decision (2026-10-12): detector not changed and nothing tuned on the same 20 subjects.

## 6. Storage and deployment

- **No model is stored.** The notebook only evaluates (LOSO). It never fits a final model
  on all 20 subjects and never writes a file.
- If a model is wanted later: fit `make_model()` on all 20, export with skl2onnx, save to
  `D:\Alzheimers\models\`. It is a 30-feature logistic regression, so the file is tiny.
- **Not patient-ready** (CLAUDE.md rule 7). The live app would use a ~30 Hz webcam
  (MediaPipe + WebGazer), about 11x slower than the research tracker, i.e. ~9 gaze points
  per task. Saccade features cannot be reproduced at that rate (FINDINGS 1). Coarse
  features would have to be validated separately on webcam data we do not have.
