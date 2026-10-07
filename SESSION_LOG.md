# Session Log

One entry per session, newest at the bottom. Say **"update the log"** to add one.

Template:

```
## YYYY-MM-DD — <short title>
**Discussed:** ...
**Decided:** ...
**Changed:** files created/edited, commits
**Open / next:** ...
```

---

## 2026-09-30 — Project setup, findings and plan
*(Reconstructed on 2026-10-01 from git history and docs; the original conversation
was not saved word-for-word.)*

**Discussed:**
- Four-modality design (Clock, Eye, Motor, Speech), each independent, with an LLM
  summary layer at the end.
- Measured ADEM_TEST eye data: ~333 Hz, ~297 ms per task, ~3.6 s gaze per subject.
  Saccade features will not survive a ~30 Hz webcam.
- Read Koppelmans et al. 2023 (motor): 4 conditions × 3 trials × 10 s, shift keys,
  Balanced Random Forest, 70% acc / 0.76 AUC / 45% AD precision.
- Read Bone et al. 2025 (clock): labels are in the **public** NHATS survey files;
  input is 512×512; 3-stage pipeline, and stage 3 (TDRA cohort) can't be reproduced.
- Read Sarawgi et al. (speech): 3 branches (disfluency, acoustic ComParE→PCA 21,
  interventions RNN) + ensemble.

**Decided:**
- Clock backbone: ViT-B/16 at 512px as baseline; ViT-L/16 only if justified (flagged deviation).
- GPU platform: Kaggle. Local Python 3.13.
- Binary labels only — severity/staging and MMSE regression excluded.
- Eye: ablation over all 12 tasks, nested LOSO. Motor: keep all 4 conditions.
- No combined/overall accuracy; four independent result rows.
- LLM: local 7–8B via Ollama; rule "2+ of 4 flagged → follow-up"; schema-validated.
- NHATS data never uploaded to Claude; local scripts only.
- Plan on Opus, implement on Sonnet.

**Changed:** commits a2131d5, e3861b0, f9be3f7, c0fee55, 19d24fc —
`PROJECT_OUTLINE.md`, `FINDINGS.md`, `PLAN.md`, `scripts/check_nhats_labels.py`.

**Open / next:** verify NHATS survey labels (build-order step 0).

---

## 2026-10-01 — NHATS labels verified; memory/log setup

**Discussed:**
- Ran `scripts/check_nhats_labels.py`: 54,027 images (exact match to paper);
  `cg<r>dclkdraw` and `hc<r>disescn9` present for rounds 1–10; SPID join 99.8–100%.
- How to keep context between sessions (`claude --resume`, CLAUDE.md, this log).

**Decided:**
- **Hand annotation of clock images is not needed** — public NHATS labels are used.
- Dementia target = 1 for codes 1 and 7, else 0; negative codes excluded (confirm
  against NHATS codebook).
- Train/validation split by SPID.

**Changed:** `FINDINGS.md` section 3, `PROJECT_OUTLINE.md`,
`scripts/check_nhats_labels.py` (uncommitted); created `CLAUDE.md` and `SESSION_LOG.md`.

**Open / next:**
- Download NHATS Round 11 survey file (3,091 R11 images currently unlabelled).
- Confirm negative value codes in the NHATS codebook.
- Update `PROJECT_OUTLINE.md` clock section (still says 224×224 and hand annotation).
- Build-order step 1: repo scaffold, venv, pinned dependencies → then Speech.

---

## 2026-10-01 (evening) — Supplement read, crop QC decided, Sonnet handoff

**Discussed:**
- SAS vs Stata: same data, two formats. **Stata** used (smaller, pandas reads it).
  SAS folder unused. Nothing needs unzipping — scripts read from the zips.
- Round 11 survey: first attempt left unfinished `.crdownload` files; re-downloaded,
  now `survey/STATA/NHATS_R11_Final_Release_STATA_V3.zip`.
- Read `Paper/dad270171-sup-0001-suppmat.docx` (Bone et al. Supplementary Methods).
  Crop-Net targets are generated automatically; full cropping and training specs found.
  Gaps: name-box filter (code unpublished), handwriting source unnamed, author's manual
  review of all 54,027 crops. Details in `FINDINGS.md` 3a.
- `Paper/DAD2-17-e70171.pdf` is a byte-identical duplicate of `Clock_drawing.pdf`.
- Zenodo note in the clock paper refers to the **authors' clock code** ("Code will be
  available on Zenodo upon publication") — not found; separate from the ADEM eye data.

**Decided:**
- **Crop quality check = Option A:** automatic rules, validated against ~300 crops the
  user grades by eye. Flagged deviation.
- Clock split: SPID-grouped train / validation / test (test added beyond the paper).
- Colab is the fallback if NHATS terms disallow Kaggle; only 512px crops + labels upload.

**Changed:** `scripts/check_nhats_labels.py` (reads SP `.dta` from inside Stata zips,
Python 3.13); `FINDINGS.md` (section 3, 3a, open items); `PLAN.md` (status, step 0 done,
section D rows, **new section H: clock build spec**); `PROJECT_OUTLINE.md` (clock section:
labels resolved, 512px, deviations; stray `c` on line 1 removed). Not committed.

**Open / next (for Sonnet — follow `PLAN.md` section H, in order):**
1. ✅ Round 11 confirmed: both columns found, 3,091/3,091 images joined.
1b. Plan review fixes (same evening): dementia negative codes -> 0 per paper (was:
    excluded); negative clock codes excluded [OURS]; threshold picked on validation, not
    test; Crop-Net canvas-fit interpretation; file map added; torch-on-3.13 check.
1c. Do plan step 1 (scaffold) **before** H1.
2. H1 label table → H2 SPID split → H3 preprocess (all local, read from zips).
3. Before H4: read Crop-Net layers from Figure 1 of the paper (do not guess).
4. Confirm NHATS negative codes in the codebook.
5. Then plan step 2 (speech) per `PLAN.md` section C.


**Session end (2026-10-01):** plan reviewed and fixed; user walked through every clock
step, file and function (`PLAN.md` H, file map). Resume tomorrow: user switches to
`/model sonnet`, then "read the session log and do plan step 1 (setup), then H1".
Doc changes from this session are **not committed** yet.

---

## 2026-10-07 / 08 — Plan finalised: all four modalities, ONNX, cloud-only ML

**Discussed:**
- Where we left off: scaffold not started; last session's doc changes uncommitted.
- User's working model: all ML (preprocessing + training) on Kaggle/Colab, nothing on the
  local PC; Claude writes code from the papers; the user runs NHATS code; build each
  modality separately, integrate afterwards; the finished test runs **on a laptop only**.
- Teacher-facing questions: "did you set thresholds yourself?" and "can I see what it
  captured?" — answered in the plan (eye threshold table; "see what was captured" views).
- Why live speech needs Whisper: two of the three networks read a human-typed transcript,
  and the interventions network needs an interviewer, which a self-test lacks.
- Why motor needs a classifier: tapping features are only numbers until a model learns
  from labelled people which patterns go with which group.

**Verified / read this session:**
- **NHATS negative codes** from the Stata value labels (metadata only), R1/2/5/8/11:
  clock -1 inapplicable, -2/-3 proxy, -4 not attempted, -7 refused, -8 DK, -9 missing;
  dementia 1 yes, 2 no, 7 previously reported, -1/-7/-8/-9. FINDINGS 3b.
- **Crop-Net Figure 1** read by extracting the embedded image from the PDF: 7 conv-down
  blocks (+2 downsampled-input channels each), 2 conv, 4 conv-up, final conv → 1x64x64 →
  sigmoid. **Discrepancy:** supplement says 2x512x512 target. PLAN H4, FINDINGS 3a.
- **Speech:** official code read — exact 11 disfluency features, layer sizes, ComParE
  config, per-fold scaler/PCA; code hparams differ from the paper. Paper checkpoints on
  the reported fold (optimistic). FINDINGS 5.
- **Eye:** the "paper" title is the Zenodo dataset record; no published paper found;
  related Tianjin papers need stimulus images we don't have. FINDINGS 6.
- **Motor:** paper is 3-class (CN/aMCI/AD), BRF 50 trees depth 5; data "on request".
  No public labelled tapping dataset exists (Kutz 2022 on request; Lima 2024 healthy
  only). No validated threshold exists. FINDINGS 2.
- NHATS Conditions of Use: not found online.

**Decided:**
- All preprocessing + training on Kaggle (primary) / Colab (fallback); laptop-only
  deployment; every model exported to **ONNX** with a parity check.
- **Speech:** paper text hparams (batch 8, lr 0.01) + layer sizes from code; inner
  validation split for checkpointing [OURS]; live = Whisper + Disfluency + Acoustic,
  Interventions on research data only [OURS].
- **Eye:** own method [OURS] — adaptive velocity threshold, classic features, logistic
  regression (0.5 cut-off), LOSO; **3 tasks fixed in advance**: antisaccade1,
  visual_attention1, visual_search1.
- **Motor:** 3 of 4 conditions (right, simultaneous, alternating) [OURS]; user emails
  Koppelmans et al. for data; classifier outputs the dataset's groups (CN/aMCI/AD) if
  data arrives; features only until then.
- **Output classes rule:** each modality outputs its dataset's own classes; no severity
  or staging we construct (CLAUDE.md rule 3 updated).
- **NHATS upload:** on hold until the user pastes the Conditions of Use.
- Every test gets a "see what was captured" view in the app.
- Display order: Clock → Motor → Eye → Speech. Training order: Speech → Eye → Clock →
  Motor → ONNX → app → LLM.

**Changed:** `PLAN.md` (rewritten: method cards B1-B4, build order, decisions, H4
Crop-Net layers, new sections I and J), `FINDINGS.md` (motor search + BRF, 3a Figure 1 +
discrepancy, 3b negative codes, open items, new sections 5 speech code and 6 eye paper
search), `PROJECT_OUTLINE.md` (rewritten, display order, ONNX, laptop, deviations),
`CLAUDE.md` (rules 3, 4, 5; environment). Not committed.

**Open / next:**
1. User pastes the NHATS Conditions of Use → decide upload scope (blocks clock).
2. Claude drafts the email to Koppelmans et al.; user sends it.
3. Resolve Crop-Net output shape (1x64x64 vs 2x512x512) before coding H4.
4. Commit the docs when the user says so.
5. Then plan step 1 (scaffold), then Speech.
6. Optional: ADReSS test labels from DementiaBank.

**Session end (2026-10-08):** plan approved; all docs updated to match it (`PLAN.md`,
`FINDINGS.md`, `PROJECT_OUTLINE.md`, `CLAUDE.md`, this log); consistency scan clean.
Still **not committed**. Resume with: user pastes the NHATS Conditions of Use → Claude
decides upload scope; Claude drafts the Koppelmans data-request email; resolve the
Crop-Net output-shape discrepancy; commit on the user's word; then scaffold → Speech.
