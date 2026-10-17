# Verified Findings

Facts measured directly from the data and source references. Nothing here is assumed.
Date: 2026-09-30 (last updated 2026-10-08)

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

**Classifier details (read 2026-10-07):** "Balanced random forest classification from the
Imbalanced Learn (0.9.1) package, implemented in Python 3.8.10"; GridSearchCV gave max
depth 5, 50 trees, gini; five-fold CV on a 70% train / 30% test split. Target is
**3-class**: control / amnestic MCI / AD. Data: "available on request from the
corresponding author."

**Lima 2024 checked 2026-10-17 (search result of the Scientific Data paper, PMC11582313; the figshare page returned 403):**
smartphone **touchscreen** tapping, 30 s, central screen area, one or both hands, 176 healthy adults 18-74. It is **not**
the keyboard shift-key, 10 s, 4-condition task of Koppelmans, so it cannot serve as a healthy reference range for our
features (different device, duration and conditions, younger ages). Earlier idea of an "atypical vs healthy reference"
flag from this dataset is therefore withdrawn.

**Public labelled tapping data — searched 2026-10-07, none found:**

| Dataset | Groups | Access | Usable? |
|---|---|---|---|
| Koppelmans et al. 2023 | 47 CN / 27 aMCI / 26 AD, shift keys | On request | Best match — requires email |
| Kutz et al. 2022 (J Pers Med 12:286) | 79 healthy / 80 possible MCI / 66 MCI, age 79-92 | On request | Force transducer, not keyboard; LDA 50% accuracy |
| Lima et al. 2024 (Sci Data, figshare 10.6084/m9.figshare.26940823) | 176 **healthy only**, 18-74 | Open, CC-BY | No patient group → cannot train a classifier |

No published, validated cut-off exists for this task, so no threshold is used.

**Re-searched 2026-10-12 (web, standard + extended):** still no open keyboard tapping
dataset with AD/MCI labels. Found only Parkinson's sets (HUBU-FIS video, Zenodo
10.5281/zenodo.17738775, CC-BY; Kasemsap figshare 53 PD / 47 HC; MIT-CS1PD PhysioNet
typing) and light-diode / smartphone AD studies without public data. Not usable for a
CN/aMCI/AD classifier.

**Reference task code read 2026-10-12** (github.com/vnckppl/FingerTappingTask, AGPL-3.0,
PsychoPy; `lastrun.py`): key-down only (`event.getKeys`), keys `lshift`/`rshift`, 4 tasks x 3
repetitions x 10 s, times in seconds from the task screen flip, saved with `saveAsWideText`
(fields `Lk/Rk/Bk/Ak.keys` and `.rt`). No release times, no sample data.

**Paper's metric definitions (read 2026-10-12, PMC10578230):** 20 continuous metrics = 3
dominant + 3 non-dominant + 7 dual + 7 alternating; the paper gives names and one-line
descriptions but **no formulas**; "onset delay" is not defined; gap = period >= 1 s with no
key; trials combined by median (Table 4 footnote) but "maximum" for gap measures (Table 3
footnote). Dominant / non-dominant come from the Dutch Handedness Questionnaire.

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
| `cg<r>dclkdraw` (clock 0-5) | Present in SP file, rounds 1-11 |
| `hc<r>disescn9` (dementia) | Present in SP file, rounds 1-11 |
| Image to survey join on SPID | 99.8-100% per round, rounds 1-11 (24 unmatched images in total, all in R1-R3) |
| Round 11 survey | Both columns found; 3,091 of 3,091 images joined (2026-10-01) |

Value codes seen: clock 0-5 plus negative special codes (-1, -2, -3, -4, -7, -9);
dementia 1 = yes, 2 = no, 7 = previously reported (from R2 on), plus negative codes.
Per Bone et al. (2.4.1), dementia target = 1 for codes 1 and 7, **0 otherwise** (negative
codes included as 0). Negative clock codes have no 0-5 score and are excluded [OURS].

Dementia prevalence (codes 1 + 7) is roughly 6-8% of SP rows per round: heavy class
imbalance, which supports balanced accuracy as the headline metric.

**Hand annotation is not needed.** The single-rater / no-kappa limitation is withdrawn.

### 3a. Bone et al. Supplementary Methods (read 2026-10-01)

Source: `Paper/dad270171-sup-0001-suppmat.docx`.

| Item | What the supplement says |
|---|---|
| NHATS preprocessing | Already line-filtered by NHATS; only **inverted**. Black name-censoring rectangles removed by a filter — "see accompanying code", **which is not published**. |
| Crop-Net targets | **No hand labelling.** Rough crop (96% of pixel sum, +50 px), then the clock is rotated (≤180°), resized (1 to 1/8), placed on a 512×512 black canvas with added noise, lines, writing and rectangles. Target = 2×512×512: clock area = 1, and clock area = score/5. |
| Crop-Net data split | 52,527 train / 500 val / 1,000 test |
| Crop-Net training | MSE; Adam lr 0.00002, betas (0.5, 0.999); batch 50; LR ÷5 on plateau; early stopping |
| Multi-step crop | ×3: heatmap → 84% of heatmap sum +80 px → 96% of pixel sum +30 px → pad square → 512×512 |
| Manual review | Lead author eyeballed **all 54,027 crops**: 48,985 good / 5,042 bad (9.3% rejected). Crop-Net retrained from scratch on good crops (48,485 train / 500 val). |
| "Writing" augmentation | "Publicly available sources on the internet" — **source not named** |
| Crop-Net layers | Only in Figure 1 of the main paper (raster image). **Read 2026-10-08** by extracting the embedded image from the PDF — full layer list in PLAN.md H4 |

**Crop-Net output discrepancy (resolved by choice 2026-10-09, PLAN H4):** Figure 1 ends in Conv 3x3 → **1x64x64** → sigmoid
("clock location heatmap"). The supplement describes the target as **2x512x512** (channel
0 = clock area, channel 1 = clock area x score/5). Figure 2A shows single-channel
"score-weighted targets". These do not agree. Re-read 2026-10-09: the supplement is
self-consistent ("The 512x512 heatmap was reshaped to the original image size", para on
multi-step cropping); only Figure 1 differs. Authors' code is unpublished, so no source
settles it. **Choice [OURS]:** Figure 1 layers with a 2-channel final conv (2x64x64);
2x512x512 target area-averaged to 64x64; channel 0 used for cropping.

**Correction:** the ViT's 47,485 + 1,000 = 48,485 images are the **successfully cropped**
images, not "images with valid labels". The ~5,500 gap from 54,027 is crop rejection.

### 3c. NHATS Conditions of Use (read 2026-10-09)

Source: https://www.nhats.org/researcher/data-access, heading "Conditions of Use for NHATS
Public Use Data" (accepted at registration). Relevant clauses, verbatim:

- "will NOT upload NHATS or NSOC data to a public large language model or AI platforms
  including but not limited to ChatGPT, Claude, Gemini, Copilot, and other similar systems
  that process data externally and do not guarantee data confidentiality or non-retention."
- "will NOT transfer NHATS Public Use data to anyone other than staff or students that you
  have direct responsibility for."
- "will use the data only for scientific research purposes and only for aggregate
  statistical reporting."
- Cite NHATS; notify NHATSdata@Westat.com of publications; destroy data if requested.

Kaggle / Colab are not named. **User's decision:** Google Drive + Colab allowed; Kaggle
(including private datasets) not used. Colab's Gemini / any AI assistant must not be used
in a notebook with NHATS data loaded. Optional confirmation: NHATSdata@Westat.com.

### 3b. NHATS negative codes (read 2026-10-07)

Read from the **value labels** stored in the SP `.dta` files (metadata only, no respondent
rows), rounds 1, 2, 5, 8, 11 — identical across rounds.

| Code | `cg<r>dclkdraw` (clock) | `hc<r>disescn9` (dementia) |
|---|---|---|
| 0-5 | 0 not recognizable … 5 accurate | — |
| 1 / 2 / 7 | — | YES / NO / PREVIOUSLY REPORTED (7 from R2) |
| -1 | Inapplicable | Inapplicable |
| -2 | Proxy says cannot ask SP | — |
| -3 | Proxy says can ask SP but SP unable to answer | — |
| -4 | SP did not attempt to draw clock | — |
| -7 | SP refused to draw clock | RF (refused) |
| -8 | DK | DK |
| -9 | Missing | Missing |

R11 dementia label text reads "…INCLUDES EVER". Consequences: no negative clock code is a
score → excluded. Dementia negatives → 0 per the paper; refused/DK are genuinely unknown,
so `dementia_raw` is kept for a sensitivity analysis.

---

## 4. Open items

- Eye-tracker hardware model: unknown, not published.
- Clock labels: resolved — public NHATS SP files, all 11 rounds (section 3).
- Clock split: by SPID (same person recurs across rounds).
- Crop-Net output shape: resolved by choice 2026-10-09 (2x64x64) — section 3a, PLAN H4.
- NHATS Conditions of Use: read 2026-10-09 (section 3c). Drive + Colab only; no Kaggle.
- Motor labels: none public. Data request emailed to Koppelmans et al. (sent by user, by 2026-10-09); awaiting reply.
- ADReSS test labels: blank on our copy; obtainable from DementiaBank (optional).

---

## 5. Speech — Sarawgi et al. official code (read 2026-10-08)

Paper: *Multimodal Inductive Transfer Learning for Detection of Alzheimer's Dementia and
its Severity*, Interspeech 2020. Code: github.com/wazeerzulfikar/alzheimers-dementia (MIT),
TensorFlow/Keras.

| Item | From the code |
|---|---|
| 11 disfluency features | intervention rate; counts of `&uh`-type fillers, `<…>` retracing, `[/+]`, `(.)` pauses, `+...` trailing off, `mhm/hm`, `[:word]`, `wor(d)`, `w:ord`; word rate (÷50). All ÷ (audio seconds/10) |
| Interventions input | Speaker per line INV `[0,0,1]` / PAR `[0,1,0]` / pad `[1,0,0]`, first→last PAR line, length 32 |
| ComParE | `SMILExtract -C IS13_ComParE.conf` per full-wave file (6,373 features) |
| Scaler + PCA(21) | Fitted per fold on the training part only |
| Disfluency model (`create_pause_model`, re-read 2026-10-09) | BN → [Dense ReLU → BN → Dropout] × 4 (16, 16, 24, 24; dropout 0.2/0.2/0.1/0.1) → Dense 2 softmax with kernel l2(0.01) + activity l1(0.01). Hidden layers unregularised |
| Acoustic model (`create_compare_model`) | Dense 24 ReLU (kernel l2 0.01 + activity l1 0.01) → BN → dropout 0.2 → Dense 2 softmax (kernel l2 0.01 + activity l1 0.01). Activity l1 on a softmax output is constant (sums to 1) → no effect |
| Interventions model | LSTM 16 → BN → dropout 0.2 → Dense 16 ReLU → BN → Dense 2 softmax. No regularisers. Sequence length: `config.py` 32 (the `dataset.py` default is 40) |
| Code hparams | lr 0.00125 / batch 24 (disfluency, interventions); lr 0.01 / batch 16 (acoustic); 2000 epochs; ModelCheckpoint on validation loss |
| Paper hparams | batch 8, Adam lr 0.01 for all — **paper text is followed** (decision 2026-10-07) |

Paper's evaluation saves "the best models … by monitoring the validation loss in each
fold" — the same fold it reports on, which is optimistic. We checkpoint on an inner split.

**Quirks of the official feature code (reproduced as published, 2026-10-09):** regexes run over
the whole `.cha` text including `@` headers and `%mor`/`%gra` lines; a line counts as
INV/PAR if it merely contains the string (so the `@Participants`/`@ID` header lines count).
`models.py` also contains an unused 800-step Conv1D "silences" model — not part of the paper's three.
**[OURS] ComParE:** `opensmile-python` `ComParE_2016` replaces `SMILExtract -C IS13_ComParE.conf`
(both 6,373 features; equivalence not verified). **[OURS] PyTorch port** of the Keras models (ONNX export).

Data check on disk: train 54 cc (S001…) + 54 cd (S079…) full-wave `.wav`, `.cha`, meta
with MMSE; test 48 (S160…) with `test_results.txt` empty — matches the paper's split.

---

## 6. Eye — search for the method paper (2026-10-07)

- Zenodo 10.5281/zenodo.18796183: title "Deep Learning-Driven 3D Eye Tracking as a
  Biomarker for Alzheimer's Disease Detection and Assessment", author Zhang, Wenlin
  (Tianjin University), CC-BY-4.0. No journal / DOI / code given. **No published paper with
  this title found.**
- Same group: MC-CNN (arXiv 2303.06868) and DISCN (arXiv 2403.10124) use stimulus RGB and
  depth maps with heatmaps. ADEM_TEST has no stimulus images → neither is reproducible.
- Conclusion: the eye method is our own, flagged (PLAN.md B2).


---

## 7. TalkBank / DementiaBank Ground Rules (read 2026-10-10)

Source: https://talkbank.org/0share/rules.html (fetched via a summarising tool, so quotes
are fragments; re-read the page before relying on exact wording). ADReSS is listed under
"Challenges" on https://talkbank.org/dementia/ with **no terms of its own**; the general
DementiaBank (password-protected clinical bank) rules therefore apply as far as we can tell.

- "Password-protected data cannot be posted on other websites or servers or shared with
  anyone" without password access.
- "Users cannot upload TalkBank data to web-based systems unless those systems include
  assurance" that data is not kept; the page names OpenAI/Anthropic/Google/Rev-AI with
  opt-outs. Google Drive, Colab and Kaggle are **not mentioned either way**.
- Downloaded identifiable data "can only be stored on a local device for analysis and
  cannot be circulated further".
- Access to clinical banks is restricted to faculty; students go through their faculty
  advisor, who is responsible for the student's compliance.
- Non-commercial use; cite sources in publications.

**Consequence:** a Kaggle private dataset of ADReSS is "posted on another server" — likely
not permitted. Not settled; see SESSION_LOG 2026-10-10.

---

## 8. Eye — features measured on ADEM_TEST (2026-10-10)

- Sentinel `-100` = no data in a gaze column; treated as missing.
- Recordings are ~0.3 s: with the adaptive threshold the three fixed tasks average **~0.5-1.3
  saccades per recording**; 9-11 of 20 subjects have no detected saccade on a given task, so
  saccade features are missing for ~half the subjects and are imputed (training fold only).
- Tracker dropout differs by group (antisaccade1: AD 34% vs NC 15% of samples; visual_search1:
  30% vs 3%) → a data-quality confound; reported, not used as a feature.
- L/R disconjugacy is 150-360 px: dominated by eye-tracker calibration offset, not a
  validated gaze sign. Kept because PLAN B2 lists it; interpret cautiously.
- Antisaccade direction errors cannot be computed: the target side is not in the files.

### 8b. Eye pilot result and detector check (run on Colab 2026-10-12)

Measured, ADEM_TEST, n=20 (10 AD / 10 NC), leave-one-subject-out, logistic regression, cut-off 0.5:

| Input | Balanced accuracy (95% CI) | AUC (95% CI) |
|---|---|---|
| all three tasks | 0.300 (0.101-0.505) | 0.240 (0.000-0.510) |
| antisaccade1 | 0.300 (0.107-0.521) | 0.310 (0.090-0.593) |
| visual_attention1 | 0.450 (0.227-0.667) | 0.300 (0.081-0.582) |
| visual_search1 | 0.400 (0.192-0.625) | 0.340 (0.101-0.615) |

Every CI includes 0.5 → **no detectable signal; inconclusive, not "30% accurate".** AUC below 0.5 is
the known LOSO artifact on tiny balanced groups, not a reversed signal.

Detector check (60 recordings): the adaptive threshold (mean + 6 SD) exceeds the recording's own
maximum velocity in **28/60 (47%)**, giving 0 saccades by construction; the same 28 recordings
have n_sac = 0. The cause is the published rule on ~0.3 s recordings, not the start value (any
start >= max gives the same first step). Timestamps are integer ms with ~3 ms spacing, so
velocity carries ~±30% quantisation noise. `231120_lfr_f` antisaccade1 has 0 valid samples.
Decision: no detector change (an invented threshold or tuning on these 20 subjects is not allowed).

## 9. Speech — ADReSS train result (run on Colab 2026-10-17)

Measured, ADReSS train, n=108 (54 cc / 54 cd), 5-fold CV, stratified, seed 0, subject-level bootstrap
95% CI, threshold 0.5. Shape checks passed (features (108,11), (108,32,3), ComParE (108,6373), no NaN/inf).

| Network | Balanced accuracy (95% CI) | F1 (95% CI) | AUC (95% CI) |
|---|---|---|---|
| Disfluency | 0.759 (0.669-0.841) | 0.764 (0.667-0.845) | 0.819 (0.731-0.896) |
| Acoustic (ComParE) | 0.583 (0.494-0.678) | 0.554 (0.431-0.667) | 0.643 (0.537-0.749) |
| Interventions | 0.704 (0.613-0.784) | 0.698 (0.586-0.784) | 0.739 (0.638-0.822) |
| Ensemble (hard vote) | 0.759 (0.675-0.834) | 0.750 (0.645-0.832) | not defined for a vote |

Reading: Disfluency, Interventions and the ensemble have CIs entirely above 0.5. Acoustic's balanced
accuracy CI includes 0.5 (lower bound 0.494), so no reliable signal from ComParE alone here.
The ensemble does not beat Disfluency alone. This is cross-validation on one dataset (the ADReSS
training set, matched for age/gender); no held-out test and no domain-shift check yet (rule 7), so
nothing here supports a patient-facing claim. Not compared to the paper's figures (different
evaluation, rule 6). `speech_oof.csv` (out-of-fold predictions) was produced on Colab.

### 9b. Speech — Whisper transcript check (run on Colab 2026-10-17)

Whisper `small`, word timestamps, ADReSS train n=108 (54/54), prompt only discourages dropping fillers.
Five features recoverable from words + timing (`speech/whisper_features.py`; pause = gap >= 0.5 s, fixed in advance):

| Feature | Spearman vs human | Human mean | Whisper mean | Zero in human / Whisper |
|---|---|---|---|---|
| fillers | 0.269 | 1.030 | 0.145 | 3% / 51% |
| mhm_hm | 0.320 | 0.200 | 0.029 | 50% / 85% |
| repetition | 0.241 | 0.429 | 0.102 | 18% / 61% |
| pause | 0.020 | 0.185 | 1.535 | 41% / 2% |
| word_rate | 0.852 | 1.245 | 0.319 | 0% / 0% |

Median words per recording: 369 (human features) vs 100 (Whisper). The human word count is inflated by the
official code's quirk of counting `@header`/`%mor` text (FINDINGS 5), so word-rate means are not on one scale;
only the rank correlation is meaningful.

Network on the 5 Whisper features (Disfluency architecture, 5 inputs), 5-fold CV, n=108, stratified, seed 0:
balanced accuracy **0.639 (0.546-0.731)**, F1 0.629 (0.510-0.730), AUC 0.643 (0.531-0.749).

Reading: Whisper `small` loses most disfluencies (fillers 14% of human count, 51% of recordings with none);
pause counts do not track the human `(.)` marks at all (rho 0.02, different definition). Only word rate
carries over. The Whisper-input network is above chance but clearly below the human-transcript Disfluency
(0.759, 0.669-0.841; the CIs overlap but the point estimate drops 0.12). The 11-feature ONNX
(`speech_disfluency.onnx`) cannot be fed from Whisper. **Live Speech is therefore weak:** nothing here
supports a patient-facing claim. One model size, one prompt, one pause threshold were tried.

### 9c. Speech — Whisper `medium` (run on Colab 2026-10-17, same protocol as 9b)

| Feature | Spearman vs human | Human mean | Whisper mean | Zero in human / Whisper |
|---|---|---|---|---|
| fillers | 0.260 | 1.030 | 0.136 | 3% / 54% |
| mhm_hm | 0.501 | 0.200 | 0.039 | 50% / 82% |
| repetition | 0.307 | 0.429 | 0.092 | 18% / 63% |
| pause | -0.017 | 0.185 | 1.489 | 41% / 1% |
| word_rate | 0.825 | 1.245 | 0.319 | 0% / 0% |

5-feature network, 5-fold CV, n=108: balanced accuracy **0.556 (0.460-0.650)**, F1 0.529 (0.400-0.642),
AUC 0.572 (0.462-0.683). The CI includes 0.5 → no detectable signal.
Reading: the larger model did not recover more disfluencies (fillers 0.136 vs 0.145 with `small`), and the CV
result is not better (0.556 vs 0.639; CIs overlap, so this is noise-level difference, not "medium is worse").
**Conclusion:** with Whisper transcripts, the published disfluency approach does not carry over (two model
sizes, one prompt, one pause threshold). Live Speech via Whisper is **research-only, no patient-facing claim**
(rules 3 and 7). Not tried: a verbatim-trained ASR.

## 10. Motor — healthy reference from Lima (computed locally 2026-10-17)

`motor/reference.py` on `datasets/Lima.csv` (49,929 taps; 176 subjects, 304 trials; 176 dominant + 128 non-dominant;
ages 18-74, median 30; decades: 10s 15, 20s 70, 30s 48, 40s 28, 50s 10, 60s 4, 70s 1). Dominant-hand trial, first
10 s, three features also defined in our keyboard task:

| Feature | Healthy reference cut-off [OURS: 5th / 95th percentile, fixed in advance] |
|---|---|
| taps/s | below 3.85 |
| variance of 1-s counts | above 3.71 |
| time to first tap | above 1.37 s |

Flag = at least 2 of the 3 outside. Flagged in the reference sample itself: 10/176 (6%). **By age: 5/15 (33%)
of those aged 50+ and 2/5 aged 60+ are flagged** (median taps/s 5.80 overall vs 4.90 aged 50+). So healthy older
adults fall outside a reference made mostly of 20-40 year-olds: the flag would over-call in the target age group.
This is the domain shift, measured. It is a descriptive "outside the healthy range" flag, **not** an Alzheimer's
detector: no AD-labelled data, no accuracy, unvalidated. Small n at older ages (15 aged 50+) makes a
sensible age-matched range impossible from this dataset.

### 3d. Clock Part A result (run on Colab 2026-10-17, aggregate counts only)

- Images indexed **54,027** (matches Bone et al.). Kept **53,914**; excluded 89 `clock_code_not_a_score` + 24 `no_survey_match`.
- 10,887 unique SPIDs. Clock score counts (kept): 0: 494, 1: 1,837, 2: 6,089, 3: 11,700, 4: 20,057, 5: 13,737.
- Dementia = 1: 2,568 of 53,914 (4.8%). Per round 1-11 kept images: 6,918 / 5,504 / 4,459 / 3,671 / 7,076 / 5,997 / 5,256 / 4,658 / 4,186 / 3,098 / 3,091.
- Split by SPID (seed 0): train 47,614 images / 9,596 SPIDs / 4.8% dementia; val 956 / 202 / 4.4%; test 5,344 / 1,089 / 4.8%;
  **SPIDs in more than one split: 0.** (Val is 956, not the requested ~1,000, because whole SPIDs are assigned.)
- Dementia prevalence of 4.8% means balanced accuracy, not accuracy, is the headline (rule 6).

### 3e. Clock build: deviations from Bone et al. (code written 2026-10-17, not yet run)

| Item | Paper | Ours [OURS] | Why |
|---|---|---|---|
| Image scale | full 2560x3312 | area-averaged 4x on load; name-box size and crop margins divided by 4 | 54k TIFFs would take hours on Colab's 2 CPUs |
| ViT | ViT-L/16 SWAG, 512 px | ViT-B/16 (torchvision ImageNet-1k), 224 px, no position-embedding interpolation | free-Colab GPU time, no Drive for checkpoints |
| ViT head | 3 x FC(1024) + FC 2 | FC(768->1024), FC(1024->1024) x2, FC(1024->2), sigmoid | B/16 has 768-d features |
| Crop-Net output | Figure 1 shows 1x64x64 | 2x64x64 (area, area x score/5) | supplement's two-channel target (PLAN H4) |
| Crop-Net training | until early stopping, writing source unspecified | 6,000 iterations max, 90 min cap, random-scribble "writing", synthetic val 500 | compute budget |
| Crop QC (H6) | clock-by-clock rejection 9.3%, retrain on passing crops | descriptive stats only, no rejection, no retrain | needs the user's ~300 grades |
| Epochs | not specified | fixed 12,000 balanced draws per epoch, patience 4 epochs | |
| Threshold | not stated | chosen on natural validation set (max balanced accuracy) | |
Task heads: output 0 = clock score / 5 (x5 at use), output 1 = dementia probability; MSE on both. Reported on test: balanced
accuracy, F1, AUC (95% CI, bootstrap over SPIDs), clock-score MSE. NHATS-only evidence; photo/webcam domain shift unvalidated.

### 10b. Motor: Lima protocol with age-adjusted reference (2026-10-18; supersedes the 10 s keyboard version above)

User decision: Koppelmans data/protocol are unobtainable, so the Motor test follows Lima et al.: one hand, tap as fast as
possible for 30 s (dominant; non-dominant optional). `motor/reference.py` (rewritten): features taps/s, interval CV, fatigue
(last 10 s / first 10 s taps); each regressed on age (OLS, 176 dominant-hand trials, age clamped to 18-74); cut-offs = 5th / 95th
percentile of residuals (conventions, fixed in advance); flag = at least 2 of 3 outside.
Age trends in the reference: taps/s falls 0.19 per decade (r = -0.28); interval CV rises 0.03 per decade (r = 0.16);
fatigue ratio rises 0.10 per decade (r = 0.22).
In-sample flag counts: 2/176 overall, 1/43 aged 40+, 0/15 aged 50+, 0/5 aged 60+. The 10 s keyboard version flagged 33% of the
50+ group; age adjustment removes that over-calling, but the flag is now conservative and its **sensitivity to impairment is
unknown** (no patient data; no accuracy). Still: touchscreen vs the app's device, 15 adults aged 50+, extrapolation to ~75 limited
by clamping to 74. It means "outside the healthy range for this age", not Alzheimer's.
