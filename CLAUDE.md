# CLAUDE.md — Alzheimer's Multimodal Screening

Read this first every session. The `project/` folder explains how the app works.

## What this project is

Four **independent** screening modalities: Clock drawing, Eye movement, Motor (finger
tapping), Speech. Each has its own paper, dataset, label space and metrics. An LLM
summarises the four results as prose for clinician review at the end.

## Where things live

| File | Purpose |
|---|---|
| `project/` | Plain-language guides: front end, back end, database, each test, summary, sharing |
| `Paper/` | Source papers (Clock_drawing.pdf, Speech.pdf, DAD2-17-e70171.pdf) |
| `datasets/` | ADEM_TEST, ADReSS-IS2020-train/test, NHATS |
| `scripts/` | Local helper scripts (e.g. `check_nhats_labels.py`) |

## Hard rules (do not break)

1. **Paper → Dataset → Published protocol → Published method → Our implementation.**
   Any deviation is flagged explicitly in the docs. Read specs from the paper; never
   write them from memory.
2. **No fusion.** Never combine modalities numerically. No overall accuracy, no fused
   score. The only combination is the documented rule "2+ of 4 flagged → recommend
   follow-up", with no accuracy claim.
3. **No invented thresholds, no diagnosis, no severity/staging we construct.** Each
   modality outputs the classes its own dataset provides (speech/eye/clock: 2;
   motor: CN/aMCI/AD if data is obtained).
4. **All preprocessing and training run on Kaggle (primary) or Colab (fallback).**
   Nothing ML on the local PC. The finished test runs on a laptop via ONNX models in one
   local app.
5. **NHATS data must never be uploaded to Claude or any AI platform, nor to Kaggle**
   (NHATS Conditions of Use). Write scripts the user runs (locally or in
   their own Colab notebook, files uploaded per session); only aggregate summaries come
   back.
6. Report every metric with its 95% CI, n and CV protocol. Balanced accuracy is the
   headline. Never compare our numbers to papers' leaderboard/headline figures when
   the evaluation differs.
7. Domain shift is validated per modality before any patient-facing claim.

## Environment

- Windows 11, local Python **3.13** (3.11 is blocked by Smart App Control for new
  native wheels).
- GPU: Kaggle (~30 free GPU hours/week), Colab fallback. **NHATS (clock): Colab only, uploaded per session
  (no Drive mount; Drive is full) — never Kaggle, never Colab's Gemini/AI assistant**.
- Display order everywhere: Clock → Motor → Eye → Speech.
- Workflow: plan on Opus, switch to Sonnet (`/model sonnet` — forward slash) for coding.
