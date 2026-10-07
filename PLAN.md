# Implementation Plan (evidence-based)

Every design decision below cites the paper, code or measurement it came from.
Status: **plan only — no code written yet.** Last updated 2026-10-08.
Anything marked **[OURS]** is not in the source and is a flagged deviation.

**How we work:** read the paper → compare its data with ours → write the code from the
paper → the user runs it on **Kaggle or Colab** (no ML on the local PC) → results come back
with 95% CI, n and protocol.

**Where the finished test runs:** a **laptop only**. Every trained model is exported to
**ONNX** (parity-checked against the original) and served by one local app with
onnxruntime. The browser page captures webcam, microphone and keyboard.

---

## A. Corrections to the original outline (clock)

### A1. The clock labels are PUBLIC — verified 2026-10-01
Bone et al. 2.1.2: NHATS rounds 1-11 "from the NHATS website", 54,027 samples.
Section 2.4.1 names the targets:

| Variable | Meaning | Range |
|---|---|---|
| `cg<round>dclkdraw` | Clock score, assigned by the NHATS interviewer | 0-5 |
| `hc<round>disescn9` | Self-reported dementia/AD ("yes" and "previously reported" = 1, else 0) | 0/1 |

No hand annotation. Full verification in FINDINGS.md section 3.

### A2. Input resolution is 512x512, not 224x224
Bone et al. 2.3: "requiring a 512 x 512 resolution".

### A3. Three stages; stage 3 cannot be built
| Stage | What it does | Trained on | Can we? |
|---|---|---|---|
| 1. Crop-Net | CNN heatmap locating the clock; applied 3x to refine the crop | NHATS | Yes |
| 2. ViT | Fine-tuned to predict `[clock_score, dementia]` | NHATS | Yes |
| 3. PCA + Lasso | On ViT features; outputs dementia | TDRA (n=862, private) | **No** |

The 76.5% headline is stage 3 on TDRA: **not reproducible, never claimed.**

---

## B. Method card per modality

### B1. Speech — Sarawgi et al. 2020 (Interspeech), ADReSS
Source: `Paper/Speech.pdf` + official code github.com/wazeerzulfikar/alzheimers-dementia
(MIT licence; read 2026-10-08, see FINDINGS section 5).

| | |
|---|---|
| Paper's data | ADReSS 2020: 108 train + 48 test (also DementiaBank Pitt — we don't have it) |
| Our data | Same ADReSS. Train 54 cc + 54 cd (full-wave audio, `.cha`, MMSE). Test 48, **labels blank** |
| ① Disfluency | 11 features: intervention rate + 9 CHAT-marker counts + word rate, each ÷ (audio length/10). MLP BN→16→16→24→24→softmax 2 (ReLU, BN, dropout 0.2/0.2/0.1/0.1) |
| ② Acoustic | openSMILE `IS13_ComParE.conf`, 6,373 features → StandardScaler + PCA(21), **fitted on the training fold only** → Dense 24 (L2/L1 reg) → BN → dropout 0.2 → softmax 2 |
| ③ Interventions | Speaker sequence INV/PAR/pad one-hot (3-dim), first→last PAR line, padded/truncated to 32 → LSTM 16 → BN → dropout 0.2 → Dense 16 → BN → softmax 2 |
| Ensemble | Hard (majority vote) for research evaluation |
| Training | **Paper text:** batch 8, Adam lr 0.01, categorical cross-entropy. Layer sizes from code (paper shows them only in an image). Code's own hparams differ — recorded, not used |
| Evaluation | 5-fold CV on the 108 labelled subjects |
| **[OURS]** checkpointing | Paper saved the best model on the fold it reported on (optimistic). We use an inner validation split |
| **[OURS]** live use | Laptop mic → **Whisper** (local) transcript with word timings, pauses from gaps. Live verdict = ① + ② (mean of probabilities). ③ needs an interviewer → research evaluation only. Whisper validated against the human ADReSS transcripts |
| Excluded | MMSE regression (section E); Pitt evaluation |
| Optional | ADReSS test labels from DementiaBank → paper's own 48-subject test evaluation |

### B2. Eye — ADEM_TEST, method [OURS]
| | |
|---|---|
| Paper | **None usable.** "Deep Learning-Driven 3D Eye Tracking as a Biomarker…" is the title of the Zenodo dataset record, not a published paper. Related Tianjin papers (MC-CNN arXiv 2303.06868, DISCN arXiv 2403.10124) need stimulus RGB/depth images not in ADEM_TEST |
| Our data | 20 subjects (10 AD, 10 NC), 12 tasks, ~333 Hz binocular, heatmaps, MMSE/MoCA |
| Tasks used | **3, one per kind, fixed before any results:** `antisaccade1`, `visual_attention1`, `visual_search1` (first of each kind; antisaccade chosen over prosaccade because antisaccade errors are the best-documented AD eye sign) |
| Event detection | Adaptive velocity threshold (Nyström & Holmqvist 2010): threshold computed from each recording's own noise — no hand-set number. Speeds in px/ms (screen geometry unpublished) |
| Features | Fixation count/duration, saccade latency/amplitude/peak velocity, antisaccade direction errors, L/R disconjugacy, heatmap spread |
| Classifier | Logistic regression; cut-off 0.5 (classes exactly balanced; no tuning) |
| Evaluation | LOSO (train 19, test 1, x20). Reported as a **pilot** |
| Not used | Deep learning (n=20 too small) |
| Live | Laptop webcam, MediaPipe + WebGazer, ~30 Hz → saccade features may not survive; validated separately |

### B3. Clock — Bone et al. 2025
Source: `Paper/Clock_drawing.pdf` + `Paper/dad270171-sup-0001-suppmat.docx`.

- ViT weights (paper): `ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1`. **Ours: ViT-B/16 at 512px**
  baseline; L/16 only if justified [OURS].
- Head: 3 x FC(1024→1024) + LeakyReLU(0.2), then FC(1024→2) + sigmoid; clock output x5.
- Loss MSE; Adam lr 0.00002, betas (0.5, 0.999); LR ÷5 on plateau; early stopping.
- Two phases: head only (batch 50), then full body (batch 4, grad accumulation 10).
- Balance: replicate under-represented clock scores so 0-5 are equal.
- Augmentation: scale 0.7-1.0, translate 0-0.5, rotate ±180°, shear ±10°, uniform.
- NHATS images 2560x3312 single-channel, pre-filtered → invert; remove name boxes.
- Crop-Net: section H4 (read from Figure 1, 2026-10-08).
- Live: patient draws an 11:10 clock on paper → laptop webcam photo → same cleaning/crop → ONNX.

### B4. Motor — Koppelmans et al. 2023
Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC10578230/ ; task code
github.com/vnckppl/FingerTappingTask.

| | |
|---|---|
| Paper's data | 100 (47 CN, 27 aMCI, 26 AD). "Available on request from the corresponding author" |
| Our data | **None.** No public labelled tapping dataset found (FINDINGS section 2) |
| Task | Shift keys; "Get Ready!" 3 s, then "Start Pressing!"; 3 trials x 10 s per condition |
| Conditions used | **3 of 4 [OURS]:** right index, simultaneous, alternating. Left index dropped |
| Features | Paper's 20 minus the 3 left-hand ones = 17, plus tapping-gap count (onset delay or gap > 1 s) |
| Classifier | Balanced Random Forest (imbalanced-learn), 50 trees, max depth 5, gini (paper's GridSearchCV result) |
| Output classes | The dataset's own groups: CN / aMCI / AD (3 classes) |
| Threshold | No published validated cut-off exists → none invented |
| Route | User emails Koppelmans et al. for the data. Meanwhile: task + features + classifier code. Until data arrives: **features only, no verdict** |

---

## C. Build order (training)

| # | Step | Note |
|---|---|---|
| 0 | ✅ NHATS survey labels verified (2026-10-01) | |
| 1 | Repo scaffold: layout, one notebook per modality, `common/metrics.py`, Python **3.13** locally | Local PC = editing + running the app only |
| 2 | **Speech** end to end | Data ready, method fully specified |
| 3 | **Eye** | |
| 4 | **Clock** Crop-Net → crop → QC → ViT | **Blocked:** NHATS Conditions of Use check (upload scope) |
| 5 | **Motor** task + features (+ classifier when data arrives) | |
| 6 | ONNX export + parity check for every model | |
| 7 | Local laptop app: one endpoint per test + "see what was captured" views | |
| 8 | LLM summary (local 7-8B via Ollama), schema-validated | |

**Display order** in the app and summary: **Clock → Motor → Eye → Speech**.

---

## D. Settled decisions

| Decision | Choice |
|---|---|
| Where ML runs | All preprocessing + training on **Kaggle (primary) / Colab (fallback)**. Nothing ML on the local PC (2026-10-07) |
| Deployment | Laptop only; ONNX models via onnxruntime in one local app |
| NHATS labels | Verified. Public SP files carry both targets; SPID join 99.8-100%. No hand annotation |
| NHATS negative codes | Read from Stata value labels (FINDINGS 3b). Clock negatives excluded [OURS]; dementia negatives → 0 per paper, `dementia_raw` kept for sensitivity analysis |
| NHATS upload | **Pending:** user pastes the NHATS Conditions of Use; then decide Kaggle private dataset vs Drive+Colab vs local label step only |
| Data handling | NHATS data never goes to Claude. The user runs NHATS code; only aggregate counts come back |
| Clock crop QC | Option A: automatic rules validated against ~300 user-graded crops [OURS] |
| Name-box filter | Our own [OURS] (paper's code unpublished) |
| Crop-Net 'writing' augmentation | Our own source [OURS] |
| Clock split | SPID-grouped train / validation / test [OURS: test split added] |
| Clock backbone | ViT-B/16 at 512px [OURS]; L/16 only if justified |
| Speech hparams | Paper text (batch 8, lr 0.01) + layer sizes from official code |
| Speech live | Whisper + ① Disfluency + ② Acoustic; ③ research only [OURS] |
| Eye method | Our own classical features + logistic regression, 3 fixed tasks [OURS] |
| Motor | 3 of 4 conditions [OURS]; email authors for data; 3-class output if data arrives |
| Output classes | Each modality outputs the classes its dataset provides (speech/eye/clock 2, motor 3). No invented severity or staging |
| Local Python | 3.13 (Smart App Control blocks new native wheels on 3.11) |
| Speech MMSE regression | Out of scope (section E) |

---

## E. Severity and staging — excluded

No mild / moderate / severe, no MMSE estimate. Sarawgi et al. report RMSE 4.60 on MMSE
regression, close to the width of a whole stage band (~5-6 points), so a predicted stage
would often be wrong by a band.

**Clarification (2026-10-07):** a modality may output its dataset's own diagnostic groups
(motor: CN / aMCI / AD). Those are labels assigned by the study's clinicians, not a
severity scale we construct.

---

## F. Metrics reported per modality

| Metric | Why |
|---|---|
| **Balanced accuracy** | Headline. Classes are uneven (NHATS ~6-8% dementia) |
| Precision, recall, F1 | Sarawgi reports these |
| AUC | Threshold-independent; Koppelmans reports it (multi-class for motor) |
| **95% CI** | Mandatory. Bootstrap over subjects for CV; Wilson for single held-out proportions |
| n and protocol | Inline with every number |

```
MODALITY   TASK                  BAL ACC  F1   AUC  95% CI  PROTOCOL
Clock      dementia flag           ?      ?     ?    ?-?    SPID-grouped test split
Motor      CN/aMCI/AD (if data)    -      -     -    -      features only until data
Eye        AD vs NC                ?      ?     ?    ?-?    LOSO, n=20 (pilot)
Speech     AD vs control           ?      ?     ?    ?-?    5-fold CV, n=108
```

No fifth row, no average. NHATS, ADReSS and ADEM share zero participants, so no combined
accuracy can be computed.

---

## G. LLM summary layer

Input: the four structured results (label, probability, metrics). Output: prose for a
clinician, in the order Clock → Motor → Eye → Speech.

One stated rule, not learned: **2 or more of 4 flagged → recommend clinical follow-up.**
No accuracy claim. Constraints enforced in code: runs after all four; no weighting /
averaging / fusing; no diagnosis, no invented stage or threshold; schema-validated output,
violations rejected. Model: local 7-8B via Ollama.

---

## H. Clock build spec — step by step

Sources: Bone et al. main text + Supplementary Methods. All steps run on Kaggle/Colab once
the NHATS upload question is settled (D). NHATS scripts print aggregate counts only.

### H1. Label table
- Input: clock zips + SP `.dta` inside each Stata zip (R8 Stata sits in `survey/SAS/`).
- Image stem = SPID (one `_V<n>` suffix: strip). Columns:
  `spid, round, zip_name, member_path, clock_score, dementia, dementia_raw`.
- `clock_score` = `cg<r>dclkdraw` if 0-5; **negative codes excluded [OURS]** (none is a score).
- `dementia` = 1 if `hc<r>disescn9` in {1, 7}, else 0 (paper's rule, negatives included as 0).
- Unmatched images (24, R1-R3) excluded. Print counts by reason and prevalence.
- If the NHATS terms forbid uploading the survey files, H1 runs locally (seconds, no ML)
  and only `labels.csv` is uploaded.

### H2. SPID-grouped split [OURS]
Each SPID in exactly one of train / validation (~1,000 images) / test (10% of SPIDs);
fixed seed; stratified on "ever dementia = 1". Output `split.csv`.

### H3. Preprocess
Invert; name-box filter [OURS] (solid rectangles → 0, validated on a sample the user views);
rough crop = smallest box holding 96% of pixel sum, +50 px.

### H4. Crop-Net (read from Figure 1, 2026-10-08)
- Input 1x512x512.
- **7 "convolution downsample" blocks.** Each: Conv 3x3 s1 p1 → LeakyReLU(0.2) → Conv 4x4
  s2 p1 → LeakyReLU(0.2), then concatenate the original image downsampled to the same size
  (max-pool and bilinear = +2 channels). Channels increase at the second conv, except
  block 1 (1→25 at the first conv, 25→48 at the second).
  Outputs: (48+2)x256², (98+2)x128², (198+2)x64², (398+2)x32², (423+2)x16², (448+2)x8²,
  (498+2)x4².
- **2 "convolution" blocks:** Conv 3x3 s1 p1 → LeakyReLU(0.2); 500x4x4 each.
- **4 "convolution upsample" blocks:** ConvTranspose 4x4 s2 p1 → LeakyReLU(0.2);
  250x8², 150x16², 100x32², 50x64².
- Final Conv 3x3 s1 p1 → **1x64x64** → sigmoid → clock location heatmap.
- **Open discrepancy:** the supplement describes a 2x512x512 target (clock mask, and mask
  x score/5); the figure outputs 1x64x64. To resolve before coding (FINDINGS 3a).
- Training samples built on the fly: rotate ≤180°, resize 1 to 1/8, place on a 512 black
  canvas ([OURS] interpretation: first fit the rough crop to 512 on its longest side), add
  noise, single/parallel lines (lined paper in 2 of 50 per batch), writing [OURS source],
  rectangles; noise/lines may overlap the clock, writing/rectangles may not.
- MSE; Adam lr 0.00002, betas (0.5, 0.999); batch 50; LR ÷5 on plateau; early stopping.
  Paper split 52,527 / 500 / 1,000.

### H5. Multi-step crop (x3)
Resize to 512 → heatmap → resize back → crop to 84% of heatmap sum +80 px → crop to 96% of
pixel sum +30 px → pad square → 512x512.

### H6. Crop quality check — Option A [OURS]
User grades ~300 random crops; automatic rules (ink fraction, ink touching border, crop
size) fitted to them; report agreement. Retrain Crop-Net on passing crops and re-run H5, as
the paper did. Report rejection rate beside the paper's 9.3% (5,042 / 54,027).

### H7. ViT
Spec B3. Position embeddings interpolated to 512. Grey → 3 channels. Balancing and
augmentation in train **and validation** (paper); none on test. **Decision threshold
[OURS]:** chosen on validation (max balanced accuracy), applied unchanged to test. Report
on test: balanced accuracy, F1, AUC, 95% CI (bootstrap over SPIDs), clock-score MSE.

### File map
| Step | File | Main functions |
|---|---|---|
| H1 | `clock/labels.py` | `index_images()`, `read_sp_file(round)`, `build_label_table()` |
| H2 | `clock/split.py` | `split_by_spid()` |
| H3 | `clock/preprocess.py` | `load_tif()`, `invert()`, `remove_name_boxes()`, `crop_to_mass()`, `pad_square()` |
| H4 | `clock/cropnet/model.py`, `clock/cropnet/synth.py`, `notebooks/clock_cropnet_train.ipynb` | `CropNet`, `make_sample()` |
| H5 | `clock/crop.py` | `multi_step_crop(img, model, iterations=3)` |
| H6 | `clock/crop_qc/viewer.py`, `clock/crop_qc/rules.py` | `grade()`, `fit_rules()`, `apply_rules()` |
| H7 | `clock/vit/model.py`, `notebooks/clock_vit_train.ipynb` | `ClockViT` |
| all | `common/metrics.py`, `common/export_onnx.py` | `balanced_acc_ci()`, `auc_ci()`, `pick_threshold()`, `export()`, `verify_parity()` |

---

## I. "See what was captured" views (laptop app)

| Test | Shown |
|---|---|
| Clock | original → cleaned → Crop-Net heatmap → final crop → ViT attention map [OURS] → score + verdict |
| Motor | tap timeline per key, inter-tap gaps, pauses ≥ 1 s highlighted, feature table |
| Eye | gaze path with fixations/saccades (both eyes) + heatmap; live webcam with MediaPipe landmarks and WebGazer gaze dot, then replay |
| Speech | waveform, Whisper transcript with pauses/fillers/repeats highlighted, each network's probability |

---

## J. Things only the user can do
1. Paste the NHATS Conditions of Use → decide upload scope (blocks clock).
2. Email Koppelmans et al. for the tapping data (Claude drafts it).
3. Optional: ADReSS test labels from DementiaBank.
