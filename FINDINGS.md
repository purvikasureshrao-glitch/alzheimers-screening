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

**Public labelled tapping data — searched 2026-10-07, none found:**

| Dataset | Groups | Access | Usable? |
|---|---|---|---|
| Koppelmans et al. 2023 | 47 CN / 27 aMCI / 26 AD, shift keys | On request | Best match — requires email |
| Kutz et al. 2022 (J Pers Med 12:286) | 79 healthy / 80 possible MCI / 66 MCI, age 79-92 | On request | Force transducer, not keyboard; LDA 50% accuracy |
| Lima et al. 2024 (Sci Data, figshare 10.6084/m9.figshare.26940823) | 176 **healthy only**, 18-74 | Open, CC-BY | No patient group → cannot train a classifier |

No published, validated cut-off exists for this task, so no threshold is used.

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
