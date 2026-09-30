# Project Outline — Alzheimer's Multimodal Screening

Four **independent** modality assessments. No shared label space, no numerical fusion,
no invented thresholds. Rule for each: **Paper → Dataset → Published protocol →
Published methodology/model → Our implementation.**

All model training runs on **Google Colab or Kaggle GPU. Nothing trains on the local
CPU.** The local machine does data prep, inspection, and inference only.

---

## 1. Clock Drawing

| | |
|---|---|
| **Dataset** | NHATS clock drawings — **54,027 `.tif` images, count verified exactly** across 11 round zips. Files are named by SPID only. |
| **Paper** | Bone et al. (2025), *A vision transformer approach for fully automated and scalable dementia screening using clock drawing test images.* |
| **Protocol in the paper** | Command clock drawing from the NHATS survey; scanned drawing is the sole model input — no hand-engineered features, no clinician scoring in the loop. |
| **How we reproduce it** | Extract all 11 zips → detect/crop the clock → deskew → binarize → resize to 224×224 → fine-tune ViT-L/16 → export ONNX. Splits are **by SPID**, since the same person recurs across rounds. |
| **Model/method** | **ViT-L/16**, ImageNet-pretrained (`torchvision.models.vit_l_16`), fine-tuned on Colab/Kaggle GPU. |
| **Components** | `clock/extract.py` (`extract_all_zips`, `build_manifest`) · `clock/preprocess.py` (`detect_clock_bbox`, `deskew`, `binarize`, `to_vit_input`) · `clock/annotate/` (local labeling UI) · `clock/train_vit.py` · `clock/export_onnx.py` (`export`, `verify_parity`) |
| **Expected output** | The paper's target is a **dementia screening classification from the clock image** — binary, not a 3-way MCI/AD/normal split. |
| **Patient test** | Draw a clock reading **11:10**; capture runs the identical preprocessing path. |

> ⚠️ **FLAGGED DEVIATION — label source.** The NHATS zips contain **images only**. The
> paper's labels come from the restricted NHATS survey files, which we do not have.
> Per your decision we substitute a different label source: annotate a stratified
> subsample (~1,500–2,000 images, ~15% dual-rated for Cohen's κ) using the published
> **Shulman 6-point** clock rubric. The dataset, paper, model and protocol are
> unchanged — **only the training target is substituted**, so our numbers are not
> comparable to Bone et al.'s reported performance. The paper's exact label definition
> must still be read and confirmed before training. This annotation is the project's
> critical path.

---

## 2. Eye Movement

| | |
|---|---|
| **Dataset** | ADEM_TEST (Zenodo 10.5281/zenodo.18796183) — **20 participants: 10 AD, 10 NC**; 12 binocular tasks each (4 saccade, 5 visual attention, 3 visual search); 24 gaze heatmaps per subject; `MMSE_MOCA.xlsx` with mmse/moca/age/edu/sex/disease. |
| **Paper** | *Deep Learning-Driven 3D Eye Tracking as a Biomarker for Alzheimer's Disease Detection and Assessment.* |
| **Protocol in the paper** | Dedicated binocular eye-tracking hardware; participants complete the 12-task battery; left/right gaze sequences and heatmaps are recorded per task. |
| **How we reproduce it** | Load the binocular sequences → extract eye-movement features → **task-contribution ablation to find which of the 12 tasks actually carry signal** (not assuming all 12 are needed) → classify. Selection runs **inside the training fold only** (nested LOSO) — with n=20, selecting on the held-out subject would manufacture the result. |
| **Model/method** | Feature extraction + classifier per the paper's methodology, evaluated by **leave-one-subject-out CV**. Trained on Colab/Kaggle. |
| **Components** | `eye/load.py` (`load_task`, `load_subject`, `quality_report`) · `eye/features.py` (`detect_fixations`, `saccade_metrics`, `disconjugacy`, `heatmap_features`) · `eye/ablation.py` (`rank_tasks`, `nested_loso`) · `eye/train.py` |
| **Expected output** | **AD vs Normal Control** — binary. Nothing else is supported by this dataset. |
| **Deployment** | Webcam + **MediaPipe Face Mesh** + **WebGazer.js**, treated as a separate target domain requiring its own validation. |

**Verified data limits, designed around rather than hidden:** each task spans only
157–434 ms (median 297 ms) — **≈4 s of gaze per subject across all 12 tasks**. Loaders
must also handle: `gazeheat/` naming `visual_search1..3` as `sensitivity1..3`;
`calil.pkl`/`calir.pkl` failing to unpickle (`invalid load key '\x00'`); and
`NC/220808_zjr_m` having non-monotonic timestamps in all 4 saccade files. `MMSE_MOCA`
groups overlap (AD 5–27, NC 26–30), so MMSE is **not** usable as a proxy label. At
n=20 this is a **pilot with wide confidence intervals**, and the report says so.

---

## 3. Motor / Finger Tapping

| | |
|---|---|
| **Dataset** | None public and labeled — **we collect our own target-domain data.** |
| **Paper** | Koppelmans et al. (2023), *Delayed and More Variable Unimanual and Bimanual Finger Tapping in Alzheimer's Disease: Associations with Biomarkers and Applications for Classification.* |
| **Protocol in the paper** | vnckppl **FingerTappingTask** (PsychoPy): 4 conditions — left index, right index, simultaneous bimanual, alternating bimanual — **3 trials × 10 s each**, with exact tap timing recorded. |
| **How we reproduce it** | Reimplement the same task in **React**, preserving the original condition sequence and trial timing; log every tap at millisecond resolution. |
| **Model/method** | **Random Forest / Balanced Random Forest**, per the paper. Trained on Colab/Kaggle. |
| **Components** | `app/motor/FingerTappingTask.tsx` (`useTapRecorder`, `TrialRunner`, `ConditionSequencer`) · `motor/parse_logs.py` · `motor/features.py` (`tap_rate`, `iti_stats`, `iti_variability`, `pause_gaps`, `bimanual_sync`, `alternation_accuracy`) · `motor/train_brf.py` |
| **Expected output** | Koppelmans' own target and **their stated limitations preserved** — the study spanned cognitively normal, MCI and AD, but per-group classification performance is respected as reported, not upgraded into a clean 3-way classifier. |

**Timing honesty:** browser `event.timeStamp` is not automatically millisecond-true.
A calibration run measures actual jitter and logs device/browser per session, so the
"millisecond-level" claim is evidenced.

**Dependency:** with no labeled motor data yet, the classifier cannot be trained. The
task, timing harness and feature extraction are buildable now; until labels exist this
module reports **features only**, not a classification.

---

## 4. Speech

| | |
|---|---|
| **Dataset** | ADReSS-IS2020. Full corpus 78 AD / 78 control; **on disk we have 108 labeled training subjects (54 cc + 54 cd)** with full-wave audio, VAD chunks and CHAT `.cha` transcripts. |
| **Paper** | Sarawgi-based approach, as previously selected. |
| **Protocol in the paper** | **Cookie Theft** picture description; participant describes the scene, speech is recorded and transcribed. |
| **How we reproduce it** | Load audio + `.cha` transcripts → build the Sarawgi feature branches → ensemble per the paper's combination rule → subject-level stratified CV. **The exact branch definitions get read out of the paper before implementation, not written from memory.** |
| **Model/method** | Sarawgi multi-branch ensemble (turn/intervention structure, disfluency + pause features, acoustic embeddings), trained on Colab/Kaggle. |
| **Components** | `speech/load.py` (`load_wav`, `parse_cha`, `extract_turns`) · `speech/features/` (`intervention.py`, `disfluency.py`, `acoustic.py`) · `speech/ensemble.py` · `speech/cv.py` |
| **Expected output** | **AD vs control/non-AD** — binary, exactly as ADReSS supports. Not converted into MCI classification. |

> ⚠️ **FLAGGED LIMITATION — evaluation.** ADReSS test labels are redacted on disk
> (`@ID` → `_X_`/`_Y_`; `test_results.txt` is an empty submission template). Per your
> decision we evaluate by **subject-level CV on the 108 labeled training subjects**.
> These results are **not comparable to published ADReSS leaderboard numbers** and must
> never be reported as if they were. The harness is built so a labels file can be
> dropped in later and a true held-out evaluation runs unchanged.

---

## Domain Shift — explicit, per modality

| Modality | Research setup | Our deployment | Consequence |
|---|---|---|---|
| **Clock** | Scanned NHATS paper forms, uniform scanner | Phone/webcam photo — lighting, perspective, shadow, paper texture | Largest visual gap. Needs perspective correction + capture-condition augmentation during training. |
| **Eye** | Dedicated binocular eye tracker, ~300 Hz, calibrated | Webcam + MediaPipe Face Mesh + WebGazer.js, ~30 Hz, uncalibrated | **Different sensor, ~10× lower sampling.** Fine-grained saccade metrics may not survive. Webcam path validated separately — never assumed to transfer. |
| **Motor** | PsychoPy on fixed hardware, dedicated input | React in browser, arbitrary keyboard/touch, variable latency | Timing jitter must be measured, not assumed. Device logged per session. |
| **Speech** | Clean, noise-reduced Pitt corpus recordings | Patient microphone, room noise, varied device | Apply the ADReSS enhancement/VAD chain to our audio so train and inference paths match. |

**Rule:** no modality makes a patient-facing claim until its own domain-shift
validation has run. Domain shift is reported per modality, never averaged away.

---

## Training, Checkpoints, and How the App Uses Them

**Where training happens:** Colab or Kaggle GPU notebooks in `notebooks/`, one per
modality. **No training on the local CPU** — local is for preprocessing, inspection
and inference only.

**Data to the GPU:** preprocessed, cached archives — not raw files. Clock ships a
preprocessed 224×224 image archive (not 54k raw TIFFs); eye/motor/speech ship small
feature tables. Re-uploading raw data every session is avoided by caching to Drive.

**Checkpoint flow:**

```
Colab/Kaggle GPU  →  Google Drive / Kaggle dataset  →  D:\Alzheimers\models\
```

| Modality | Artifact | Format |
|---|---|---|
| Clock | `models/clock/vit_l16_clock.onnx` | ONNX (torch→ONNX parity verified on a fixed batch) |
| Eye | `models/eye/eye_clf.joblib` + selected-task list | joblib |
| Motor | `models/motor/brf_motor.joblib` | joblib |
| Speech | `models/speech/sarawgi_ensemble.joblib` | joblib |

Every artifact ships a **model card** beside it: target definition, training
population, CV protocol, metrics with CIs, and its domain-shift statement.

**How the app consumes them:** a local FastAPI service loads the ONNX model via
`onnxruntime` (CPU) and the joblib models via scikit-learn, exposing one endpoint per
modality. Each returns **its own independent result** with its own label space and
uncertainty. **The service never computes a combined score.**

---

## LLM — Options Compared

Scope first: the LLM's only job is to turn four already-computed structured results
into readable patient-level prose. That is a **narrow, constrained generation task**,
which makes the sizing decision easy.

| Option | Effort | Quality for this task | Learning value |
|---|---|---|---|
| **A. Hosted API** (e.g. Claude Haiku — cheap, fast, strong instruction-following) | Lowest | Best | Low — teaches prompting/API, not internals |
| **B. Local open-source 7–8B** (Llama / Qwen / Mistral via Ollama or llama.cpp) | Low–medium | Good; runs on your CPU, keeps patient data local | Medium — teaches serving, quantization, prompting |
| **C. Small Transformer from scratch** (nanoGPT-style, ~10–30M params, on Colab) | High | Not usable in the product | **Highest** — teaches attention, tokenization, training loops end to end |

**Recommendation — do both B and C, in different places:**

- **In the product: Option B**, a local open-source 7–8B model via Ollama. It is more
  than capable of templated summarization, and it **keeps patient data on your
  machine**, which matters here. Option A is a drop-in fallback if quality falls short.
- **For learning: Option C, as a separate exercise outside the product path.** Build a
  small GPT from scratch on Colab to learn the fundamentals. Do **not** attempt to
  train a large LLM from scratch, and do not put the from-scratch model anywhere near
  patient output.

**Hard constraints on the LLM, enforced in code:**
1. Runs **only after** all four modality assessments are complete.
2. Receives the four results as **structured input** and summarizes them as evidence.
3. **Must not** weight, average, fuse, or otherwise combine modalities mathematically.
4. **Must not** produce a diagnosis or introduce a threshold.
5. Output is schema-validated; any fused score or diagnostic claim is rejected, not
   post-edited.

Load the **`claude-api` skill** before writing this component if the hosted API path
is used, so model IDs and usage are current rather than remembered.

---

## Corrections to Previous Plan

1. **Training location — corrected.** The previous plan kept speech and eye training
   local ("CPU-trainable", "small, local, fast") and sequenced speech first *because*
   it ran on CPU. That conflicts with requirement 5. **All four modalities now train on
   Colab/Kaggle GPU, including the scikit-learn models.** The local CPU does
   preprocessing and inference only, and the sequencing rationale is void.
2. **Sequencing — revised accordingly.** New order: scaffold → **start NHATS extraction
   and annotation immediately** (longest lead time) → speech → eye → motor harness →
   clock ViT training → app/service/LLM. Speech still goes first, but because its data
   is complete and labeled, not because of CPU.
3. **LLM — newly specified.** The previous plan named an LLM summary step without
   choosing an approach. Now resolved: local open-source 7–8B in the product, small
   from-scratch Transformer as a separate learning exercise.
4. **Label substitution — restated as a flagged deviation.** The previous plan
   described the Shulman-rubric substitution but did not mark it as a formal deviation.
   It is now explicitly flagged above, per requirement 2. No dataset, paper, model or
   protocol has been replaced anywhere in this project — **only the clock training
   target**, and only because the paper's labels are not obtainable.
5. **No conflict** on: independent label spaces per modality, no numerical fusion,
   no invented thresholds, or the domain-shift treatment — the previous plan already
   met those and they are carried forward unchanged.

---

## Final End-to-End Flow

```text
Clock ──┐
Eye ────┤
Motor ──┼──→ Independent patient-level results
Speech ─┘
              ↓
             LLM
              ↓
     Patient-level evidence summary
              ↓
       Clinician review
```
