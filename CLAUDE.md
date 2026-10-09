# CLAUDE.md — Alzheimer's Multimodal Screening

Read this first every session, then read the latest entries in `SESSION_LOG.md`.

## What this project is

Four **independent** screening modalities: Clock drawing, Eye movement, Motor (finger
tapping), Speech. Each has its own paper, dataset, label space and metrics. An LLM
summarises the four results as prose for clinician review at the end.

## Where things live

| File | Purpose |
|---|---|
| `PROJECT_OUTLINE.md` | Per-modality design: paper → dataset → protocol → model → our implementation |
| `PLAN.md` | Evidence-based implementation plan, build order (section C), settled decisions (section D) |
| `FINDINGS.md` | Facts measured from data or read from papers. Nothing assumed |
| `SESSION_LOG.md` | Dated log of every session: decisions, changes, next steps |
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
   (NHATS Conditions of Use, FINDINGS 3c). Write scripts the user runs (locally or in
   their own Colab notebook with data on Google Drive); only aggregate summaries come
   back.
6. Report every metric with its 95% CI, n and CV protocol. Balanced accuracy is the
   headline. Never compare our numbers to papers' leaderboard/headline figures when
   the evaluation differs.
7. Domain shift is validated per modality before any patient-facing claim.

## Environment

- Windows 11, local Python **3.13** (3.11 is blocked by Smart App Control for new
  native wheels).
- GPU: Kaggle (~30 free GPU hours/week), Colab fallback. **NHATS (clock): Colab + Google
  Drive only — never Kaggle, never Colab's Gemini/AI assistant** (FINDINGS 3c).
- Display order everywhere: Clock → Motor → Eye → Speech.
- Workflow: plan on Opus, switch to Sonnet (`/model sonnet` — forward slash) for coding.

## Session log workflow

- When the user says **"update the log"** (or similar), append a new dated entry to
  `SESSION_LOG.md` using the template at the top of that file. Newest entry at the
  bottom. Do not rewrite older entries.
- At the start of a session, read the last 1–2 log entries to know where we left off.
- If a decision in the log changes a doc (`PLAN.md`, `FINDINGS.md`, etc.), update that
  doc too, and note it in the log entry.
