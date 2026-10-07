# Project Outline — Alzheimer's Multimodal Screening

Four **independent** screening tests. No shared label space, no numerical fusion, no
invented thresholds. Rule for each: **Paper → Dataset → Published protocol → Published
method → Our implementation**; every deviation is flagged **[OURS]**. Full detail:
`PLAN.md` (method cards in section B). Last updated 2026-10-08.

- **Training:** all preprocessing and training on **Kaggle (primary) / Colab (fallback)**.
  Nothing ML runs on the local PC.
- **Deployment:** a **laptop only**. Models exported to **ONNX**, served by one local app
  (onnxruntime). The browser captures webcam, microphone and keyboard.
- **Output classes:** each test outputs the classes its own dataset provides. No severity
  scale or staging invented by us.
- **Order shown in the app and summary:** Clock → Motor → Eye → Speech.

---

## 1. Clock Drawing

| | |
|---|---|
| **Paper** | Bone et al. (2025), *A vision transformer approach for fully automated and scalable dementia screening using clock drawing test images* (+ Supplementary Methods) |
| **Dataset** | NHATS clock drawings, **54,027 `.tif`, exact match to the paper**, 11 round zips, named by SPID. Labels from public NHATS SP files — verified for all 11 rounds |
| **Protocol** | Draw an analog clock showing **11:10**; scanned drawing is the only input |
| **Method (paper)** | Invert + name-box removal → **Crop-Net** heatmap crop x3 → 512x512 → crop QC → **ViT** predicting [clock score, dementia]. Stage 3 (PCA + Lasso on private TDRA) **cannot be reproduced**; its 76.5% is never claimed |
| **Labels** | Clock 0-5 (negative codes excluded [OURS]); dementia = 1 for codes 1 and 7, else 0 (paper) |
| **Deviations [OURS]** | ViT-B/16 instead of L/16; own name-box filter; own handwriting source; crop QC on ~300 user-graded crops; SPID-grouped test split; threshold chosen on validation |
| **Result** | Dementia yes/no on the SPID-grouped test split |
| **Live** | Patient draws on paper → laptop webcam photo → same preprocessing → `models/clock/vit_b16_clock.onnx` |
| **Blocked by** | NHATS Conditions of Use check (what may be uploaded to Kaggle/Drive) |
| **Open** | Crop-Net output shape: Figure 1 (1x64x64) vs supplement (2x512x512) |

---

## 2. Motor / Finger Tapping

| | |
|---|---|
| **Paper** | Koppelmans et al. (2023), *Delayed and More Variable Unimanual and Bimanual Finger Tapping in Alzheimer's Disease* |
| **Dataset** | **None.** No public labelled tapping dataset exists. Paper's data (47 CN / 27 aMCI / 26 AD) is "available on request" — the user will email the authors |
| **Protocol** | Shift keys; "Get Ready!" 3 s → "Start Pressing!"; 3 trials x 10 s per condition. **Ours: 3 of 4 conditions** — right index, simultaneous, alternating [OURS] |
| **Method (paper)** | 17 tapping features (paper's 20 minus left-hand) + tapping-gap count → **Balanced Random Forest** (50 trees, depth 5, gini) |
| **Result** | If data arrives: **CN / aMCI / AD** (the dataset's 3 groups). Until then: **features only, no verdict**. No threshold invented |
| **Live** | Browser task on the laptop keyboard (matches the paper's device); every tap timestamped; timing jitter measured per session |

---

## 3. Eye Movement

| | |
|---|---|
| **Paper** | **None usable.** The title we had belongs to the Zenodo dataset record, not a paper; related Tianjin papers need stimulus images that the dataset lacks |
| **Dataset** | ADEM_TEST (Zenodo 10.5281/zenodo.18796183): 10 AD + 10 NC, 12 tasks, ~333 Hz binocular, heatmaps, MMSE/MoCA |
| **Tasks used** | 3, one per kind, fixed in advance: `antisaccade1`, `visual_attention1`, `visual_search1` [OURS] |
| **Method [OURS]** | Adaptive velocity threshold (Nyström & Holmqvist 2010) for fixations/saccades → classic eye features → logistic regression (cut-off 0.5, balanced classes) |
| **Result** | AD vs NC, leave-one-subject-out, n=20 — a **pilot** with wide CIs |
| **Live** | Laptop webcam, MediaPipe Face Mesh + WebGazer.js, ~30 Hz (~11x slower) → validated separately |

Data quirks the loader must handle: `gazeheat/` names `visual_search1..3` as
`sensitivity1..3`; `calil.pkl`/`calir.pkl` fail to unpickle; `NC/220808_zjr_m` has
non-monotonic timestamps in its 4 saccade files. MMSE groups overlap, so MMSE is not a
proxy label.

---

## 4. Speech

| | |
|---|---|
| **Paper** | Sarawgi et al. (2020), *Multimodal Inductive Transfer Learning for Detection of Alzheimer's Dementia and its Severity* (Interspeech). Code: github.com/wazeerzulfikar/alzheimers-dementia |
| **Dataset** | ADReSS-IS2020 — the paper's own dataset. Train 54 cc + 54 cd (audio, `.cha`, MMSE). Test 48 with **labels blank** |
| **Protocol** | Cookie Theft picture description |
| **Method (paper)** | ① Disfluency MLP (11 transcript features) ② Acoustic MLP (ComParE 6,373 → PCA 21) ③ Interventions LSTM (speaker sequence, 32 steps) → majority vote. Batch 8, Adam lr 0.01; layer sizes from the official code |
| **Result** | AD vs control, 5-fold CV on 108. Not comparable to the paper's 83.3% test figure |
| **Deviations [OURS]** | Inner validation split for checkpointing; live use = **Whisper** transcript + ① + ② (③ needs an interviewer → research only); MMSE regression excluded |
| **Live** | Laptop microphone → Whisper (local) → features → ONNX models |

---

## Domain shift — per modality

| Modality | Research setup | Our deployment | Consequence |
|---|---|---|---|
| Clock | Scanned NHATS forms | Laptop webcam photo of paper | Lighting/perspective; capture augmentation + validation needed |
| Motor | PsychoPy, fixed hardware | Browser on laptop keyboard | Same device type; timing jitter measured |
| Eye | Binocular tracker ~333 Hz | Webcam ~30 Hz | Saccade measures may not survive; validated separately |
| Speech | Enhanced Pitt recordings + human transcripts | Laptop mic + Whisper | Audio quality and ASR errors; Whisper checked against ADReSS transcripts |

No modality makes a patient-facing claim until its own domain-shift check has run.

---

## Training, checkpoints, deployment

```
Kaggle/Colab GPU  →  trained model  →  ONNX export + parity check  →  D:\Alzheimers\models\  →  local laptop app
```

| Modality | Artifact |
|---|---|
| Clock | `models/clock/cropnet.onnx`, `models/clock/vit_b16_clock.onnx` |
| Motor | `models/motor/brf_motor.onnx` (only if data arrives) |
| Eye | `models/eye/eye_logreg.onnx` |
| Speech | `models/speech/{disfluency,acoustic,interventions}.onnx` + scaler/PCA |

Each artifact ships a model card: target, training population, protocol, metrics with
CIs, domain-shift statement. The app has one endpoint per test and a **"see what was
captured"** view for each (PLAN.md section I). It never computes a combined score.

---

## LLM summary

Local open-source 7-8B via Ollama (hosted API only as fallback). Receives the four
structured results, writes clinician-facing prose in the order Clock → Motor → Eye →
Speech. One documented rule: **2+ of 4 flagged → recommend follow-up**, no accuracy claim.
Must not fuse, weight or average; no diagnosis; no invented stage or threshold;
schema-validated output, violations rejected.

---

## Final flow

```text
Clock ──┐
Motor ──┤
Eye ────┼──→ Independent results (each with its own classes + CI)
Speech ─┘
              ↓
       LLM summary (rule: 2+ of 4 flagged → follow-up)
              ↓
       Clinician review
```
