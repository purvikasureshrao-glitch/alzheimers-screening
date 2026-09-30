# Implementation Plan (evidence-based)

Every design decision below cites the paper or measurement it came from.
Status: **plan only — no code written yet.**
Date: 2026-09-30

---

## A. Corrections to PROJECT_OUTLINE.md

Three of these come from reading Bone et al. properly. They matter.

### A1. The clock labels are PUBLIC. The 40-hour annotation may be unnecessary.

PROJECT_OUTLINE.md says the labels "come from the restricted NHATS survey files,
which we do not have," and proposes hand-annotating ~2,000 images with the Shulman
rubric.

**The paper says otherwise.** Section 2.1.2:

> "The **anonymized publicly available** NHATS database comprises annual interviews
> and cognitive assessments... Data were obtained from rounds 1 to 11 (collected
> 2011-2021) **from the NHATS website** in February 2023, totaling 54,027 samples."

And section 2.4.1 names the exact variables used as training targets:

| Variable | Meaning | Range |
|---|---|---|
| `cg<round>dclkdraw` | Clock score, assigned by the NHATS interviewer | 0-5 |
| `hc<round>disescn9` | Self-reported dementia/AD diagnosis (yes and previously-reported = 1, else 0) | 0/1 |

The NHATS 0-5 rubric is also printed in the paper (section 2.2.2): 0 = not recognizable
as a clock, 1 = severely distorted, 2 = moderately distorted, 3 = mildly distorted,
4 = reasonably accurate, 5 = accurate.

**Consequence if the survey files download successfully:**
- No hand annotation. ~40 hours saved.
- Labels for all 54,027 images, not a 2,000-image subsample.
- The single-rater / no-kappa problem disappears.
- The "flagged deviation" is withdrawn — we reproduce the paper's actual target.

**This must be verified before it is relied on.** Action: register at nhats.org, download
the Round 1-11 interview/sensitive-demographic files, and confirm both variables are
present and joinable to the image filenames by SPID. Until that check passes, the
annotation fallback stays on the table.

### A2. Input resolution is 512x512, not 224x224.

PROJECT_OUTLINE.md specifies "resize to 224x224". Section 2.3 of the paper:

> "Preprocessing was essential to standardize these images for input into the
> vit_l_16 dementia classification neural network, **requiring a 512 x 512
> resolution**."

224 is the torchvision default; the authors deliberately used 512 to preserve clock
detail. Using 224 would not be a reproduction. **Note this raises GPU cost** — a 512px
ViT-L/16 processes ~5x the tokens of a 224px one.

### A3. The paper's architecture is three stages, and we cannot build stage 3.

| Stage | What it does | Trained on | Can we? |
|---|---|---|---|
| 1. **Crop-Net** | Custom CNN producing a heatmap that locates the clock in the image; applied iteratively to refine the crop. MSE loss, Adam. | NHATS | Yes |
| 2. **ViT-L/16** | Fine-tuned to predict `[clock_score, dementia_flag]`. Head then discarded; the **1024-dim body output** is the feature vector. | NHATS | Yes |
| 3. **Linear model** | PCA + Lasso linear model on the 1024-d ViT features; this is what actually outputs dementia. | **TDRA (n=862)** | **No — TDRA is not public** |

The headline 76.5% balanced accuracy is stage 3 evaluated on TDRA. We have no clinical
cohort, so **we cannot reproduce that number and must not claim it.**

What we *can* do: build stages 1-2 exactly, and report the ViT's own NHATS validation
performance (clock-score MSE and dementia-flag accuracy). That is an honest,
reproducible result — just a different one from the paper's headline.

---

## B. Confirmed specifications, read from source

### B1. Clock — Bone et al. 2025

- Pretrained weights: `ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1` (torchvision)
- Head replaced with: 3 x FC(1024 to 1024) + LeakyReLU(0.2), then FC(1024 to 2) + sigmoid
- Clock output multiplied by 5 to span 0-5; dementia output left unmodified
- Loss: MSE. Optimizer: Adam, lr = 0.00002, betas = (0.5, 0.999)
- LR divided by 5 on validation plateau; early stopping on plateau
- **Two-phase training:** body frozen, head only, batch size 50, then unfreeze body,
  batch size 4 with gradients accumulated over 10 iterations (their VRAM limit)
- Split: 47,485 train / 1,000 validation
- Class balance: underrepresented clock scores replicated so scores 0-5 are equal
- Augmentation: scale 0.7-1.0, translate 0-0.5 of each dimension, rotate -180 to 180
  degrees, shear -10 to 10 degrees, all uniformly sampled
- NHATS images are 2560x3312 single-channel and pre-filtered; they must be **inverted**,
  and black rectangles censoring participant names removed by a custom filter
- Participant instruction: draw an analog clock with the hands showing 11:10

### B2. Speech — Sarawgi et al. (MIT), ADReSS

Three branches, exactly as published:

| Branch | Input | Model |
|---|---|---|
| **Disfluency** | 11 features from transcripts (word rate, intervention rate, various pause rates), normalized by audio length then scaled | MLP projecting to higher dimension |
| **Acoustic** | ComParE 2013 via openSMILE v2.1 giving 6,373 features, z-score normalized, then **PCA to 21 components** | MLP with one hidden layer |
| **Interventions** | Speaker sequence (subject vs interviewer) from the transcript, **padded/truncated to 32 steps** | Recurrent network |

- Ensemble: hard (majority vote), soft (equal-weight probability sum), or **learnt**
  (logistic regression over the three models' class probabilities)
- Training: batch size 8, Adam, lr 0.01, categorical cross-entropy
- Evaluation: 5-fold CV and LOOCV (equals LOSO here, one sample per subject)
- Uses **full audio samples**, not the provided normalized segments
- Reported: 83.3% accuracy/precision/recall/F1 on the ADReSS held-out test set
- Also performs MMSE regression by transfer learning (freeze layers, swap output head,
  lr 0.001, MSE loss). **Out of scope for us** — classification only.

Our constraint is unchanged: ADReSS test labels are redacted on our disk, so we
evaluate by CV on the 108 labeled training subjects and never compare to 83.3%.

### B3. Motor — Koppelmans et al. 2023

Confirmed in FINDINGS.md. Key points for implementation: 4 conditions x 3 trials x 10 s,
**keyboard shift keys**, "Get Ready" for 3 s then "Start Pressing", 20 continuous
features plus tapping-gap (pause of 1 s or more) measures, Balanced Random Forest,
70% accuracy / 0.76 AUC / **45% precision on AD**.

### B4. Eye — ADEM_TEST

Confirmed by direct measurement in FINDINGS.md: ~333 Hz, ~297 ms per task, ~3.6 s per
subject. Saccade features do not survive a 30 Hz webcam. The webcam path is
coarse-features-only and validated separately.

---

## C. Build order

Rationale: verify the cheap blocking question first, then build what has complete
labeled data, then what needs new data collection.

| # | Step | Why here | Blocked by |
|---|---|---|---|
| 0 | **Verify NHATS survey files download** | Decides whether 40 h of annotation is needed. Cheapest possible check, largest consequence. | User registering at nhats.org |
| 1 | Repo scaffold, 3.11 venv, pinned dependencies | Everything else imports it | - |
| 2 | **Speech** end to end | Only modality with complete labeled data on disk and a fully specified method | - |
| 3 | **Eye** loaders, features, nested LOSO | Data verified, small, n=20 pilot | - |
| 4 | **Clock** Crop-Net, then ViT stage 2 | Longest GPU time; needs step 0 resolved | Step 0 |
| 5 | **Motor** React task and timing calibration | Buildable now, but reports features only until labels exist | Labels (none) |
| 6 | FastAPI service, one endpoint per modality, no fused score | Needs models to serve | 2-5 |
| 7 | LLM summary layer (local 7-8B via Ollama), schema-validated | Last; consumes the four results | 6 |

---

## D. Decisions needed before step 1

1. **Will you register at nhats.org** to test the label download? Free and public.
   If yes, step 0 runs first and may remove the annotation work entirely.
2. **ViT-B/16 at 512px as the working baseline**, ViT-L/16 only if B/16 justifies it.
   Agreed earlier. This is a deviation from the paper and is recorded as one.
3. **Colab or Kaggle** — either is fine; pick one so checkpoint plumbing is written once.
4. **Speech MMSE regression is out of scope.** Confirm.
