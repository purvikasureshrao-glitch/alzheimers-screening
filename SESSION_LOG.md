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

---

## 2026-10-09 — Crop-Net output and NHATS upload scope settled

**Discussed:**
- Where we left off: docs from 2026-10-08 were already committed (ccb78b6).
- Crop-Net discrepancy re-read in the supplement: it is self-consistent (2x512x512
  target; "the 512x512 heatmap was reshaped to the original image size"); only Figure 1
  (1x64x64) differs. Authors' code unpublished, so no source settles it.
- Plain-words explanation of Crop-Net, the two maps, and what "the supplement" is.
  User asked why map 2 is needed: it is only a training target ("clockness", score/5),
  never used for cropping; kept to follow the paper (rule 1), no evidence it helps.
- NHATS Conditions of Use found at nhats.org/researcher/data-access (verbatim in
  FINDINGS 3c): no upload to LLM/AI platforms, no transfer, aggregate reporting only.

**Decided:**
- **Crop-Net output [OURS]:** Figure 1 layers, final conv 2 channels → 2x64x64;
  supplement's 2x512x512 target area-averaged to 64x64; channel 0 (clock mask) used for
  cropping. Map 2 kept to follow the paper.
- **NHATS:** Google Drive + Colab allowed; **Kaggle (even private) not used**; never use
  Colab's Gemini/AI assistant on a notebook with NHATS data. Speech/Eye may still use Kaggle.
- Koppelmans data-request email: **sent by the user**; awaiting reply.

**Changed:** `PLAN.md` (H4 output resolved, H5, D rows, C step 4, H intro/H1, J),
`FINDINGS.md` (3a resolved, new 3c Conditions of Use, open items), `PROJECT_OUTLINE.md`
(clock: where it runs, Crop-Net output), `CLAUDE.md` (rule 5, environment). Commit 090c410.

**Open / next:**
1. User switches to `/model sonnet` → plan step 1 (scaffold), then Speech (PLAN B1, C).
2. Await Koppelmans reply (motor data).
3. Optional: confirm Drive + Colab with NHATSdata@Westat.com; ADReSS test labels.

---

## 2026-10-09 (later) — Sonnet: scaffold (plan step 1) and Speech code

**Discussed:**
- User switched to Sonnet 5.5 and asked for plan step 1, then Speech.
- Default `python` here is 3.14.5, not 3.13; the 3.13 launcher entry exists, so the venv
  uses `py -3.13`.
- Re-read the official Sarawgi code (dataset.py, utils.py, config.py, models.py) rather
  than rely on the FINDINGS paraphrase.

**Decided / found:**
- Official feature code quirks reproduced as published: regexes run over the whole `.cha`
  (headers, `%mor`/`%gra` too); a line counts as INV/PAR if it merely contains the string.
  Sequence length 32 (config.py; dataset.py default is 40).
- **Corrected regularizers** (earlier notes were wrong): disfluency = only the softmax layer
  (kernel l2 0.01; activity l1 is a no-op on softmax); acoustic = hidden Dense 24 (kernel l2 +
  activity l1) + softmax (kernel l2); interventions = none. `models.py` also holds an unused
  Conv1D "silences" model, not part of the paper's three.
- Flagged [OURS]: PyTorch port of the Keras models (for ONNX); `opensmile-python`
  `ComParE_2016` instead of `IS13_ComParE.conf` (both 6,373 features; equivalence unverified);
  inner-split checkpointing with patience 200.

**Changed (all UNCOMMITTED):** new `common/metrics.py`, `speech/{features,models,cv}.py`,
`tests/`, `notebooks/` (5 stubs; `speech.ipynb` has real cells), `requirements-local.txt` +
`.lock.txt`, `pytest.ini`, empty `clock/ eye/ motor/` packages, `.venv` (git-ignored).
Docs: `FINDINGS.md` section 5 (corrections + quirks), `PLAN.md` B1 rows + status line.
Tests: 6 metrics + 4 speech-feature tests pass locally (first run slow, ~2 min import).
Models/CV are **untested** — no torch locally by rule 4; the notebook starts with a shape check.

**Open / next (resume tomorrow):**
1. User says "commit" (code + docs are uncommitted).
2. Check the ADReSS/DementiaBank terms before uploading to Kaggle as a private dataset.
3. Run `notebooks/speech.ipynb` on Kaggle: shape-check cell first, then CV; send back errors
   and the metrics (95% CI, n, protocol).
4. Then plan step 3 (Eye), per PLAN B2.
5. Awaiting Koppelmans reply (motor data).

---

## 2026-10-10 — Commit, TalkBank terms, Eye code

**Discussed:**
- User asked to check the ADReSS/DementiaBank terms; no reply yet from Koppelmans;
  approved the plain Eye plan; asked to commit.
- Smart App Control also blocks sklearn's `liblinear` DLL locally, so the LOSO test is
  skipped locally (runs on Kaggle).

**Found:** TalkBank Ground Rules (FINDINGS 7): password-protected data may not be posted
on other servers or uploaded to web systems without a no-retention assurance; ADReSS has no
separate terms. A Kaggle private dataset of ADReSS is therefore likely not allowed.

**Decided:** Eye per PLAN B2. [OURS] deviations: threshold iteration starts from max
velocity; antisaccade-error and heatmap features dropped; tracker dropout not a feature.

**Changed:** commit d1f782b (scaffold + Speech). New `eye/features.py`, `eye/cv.py`,
`tests/test_eye_{features,cv}.py`, real `notebooks/eye.ipynb`; `PLAN.md` B2 features + step
3; `FINDINGS.md` 7 and 8. Eye code UNCOMMITTED. 4 eye tests pass locally, 1 skipped.

**Open / next:**
1. **User decides how Speech obtains ADReSS compute** (see message): Kaggle is likely off.
2. Run `notebooks/eye.ipynb` on Kaggle (ADEM_TEST is CC-BY-4.0 on Zenodo); send results.
3. Awaiting Koppelmans reply (consider a polite follow-up after ~2 weeks).
4. Then Clock (Colab + Drive, NHATS).

**Session end (2026-10-10):** Eye code committed (e6d427c); nothing uncommitted except this
entry. Resume with: (1) user decides the ADReSS compute route — written TalkBank query via
faculty advisor / local / Colab+Drive; Speech stays blocked until then. (2) User runs
`notebooks/eye.ipynb` on Kaggle and sends back the metrics. (3) Koppelmans: still no reply.
(4) Then Clock step 4 (Colab + Drive only), starting with a plain-words plan for approval.

---

## 2026-10-12 — Clock H1-H3, Motor code, platform decisions

**Discussed:**
- Resumed from the 2026-10-10 entry. User asked for H1/H2 in plain words, how epochs and
  training work for Clock, and a full done / to-do status.

**Decided:**
- **Speech and Clock run on Colab + Drive** (ADReSS and NHATS can't go to Kaggle).
  **Eye and Motor run on Kaggle** (ADEM_TEST is CC-BY; no motor data).
- **Koppelmans: no reply, stop waiting.** Motor is built from the paper without data.
- Clock split is numpy-stratified (sklearn's scipy DLL is blocked locally by Smart App
  Control).

**Found:**
- Motor data search repeated: no public labelled AD/MCI tapping set (FINDINGS 2).
- Reference task logs key-down only; paper gives no metric formulas (FINDINGS 2).

**Changed:** commit 51ebe54 = `clock/{labels,split,preprocess}.py`,
`tests/test_clock_prep.py`, `notebooks/clock_prep.ipynb`, PLAN file map. Then (uncommitted at
time of writing) `motor/{features,loader,cv}.py`, `tests/test_motor_*.py`,
`notebooks/motor.ipynb`, `common/metrics.py` (+`auc_ovr_ci`, bootstrap skips resamples that
raise), PLAN B4 + step 2/5, FINDINGS 2. Tests: 26 pass, 2 skipped (sklearn/imblearn CV).

**Open / next:**
1. User runs `notebooks/clock_prep.ipynb` on Colab (counts + name-box verdict).
2. User runs `notebooks/eye.ipynb` on Kaggle.
3. Adapt `notebooks/speech.ipynb` to Colab + Drive, then user runs it.
4. Clock H4 Crop-Net code (synthetic sample generator, model, training notebook).
5. Motor stays features-only until data exists.

---

## 2026-10-12 (later) — Notebooks merged to four

**Decided:** one notebook per modality only: `clock.ipynb` (part A prep ready; B Crop-Net and
C ViT are placeholders), `speech.ipynb`, `eye.ipynb`, `motor.ipynb`. Each has a
"SEND BACK TO CLAUDE" cell after every step saying exactly which output to paste.
**Changed:** deleted `clock_prep.ipynb`, `clock_cropnet_train.ipynb`, `clock_vit_train.ipynb`;
new `clock.ipynb`; PLAN file map updated. (Entry above mentions `clock_prep.ipynb`; it was
renamed to `clock.ipynb` part A.)
**Open:** user hit a Colab Drive-mount / path problem on clock cell 1 (not a code bug);
awaiting the `os.listdir` / `find` output to set `ROOT`.

---

## 2026-10-12 (later still) — Run logging for Colab

**Added:** `common/runlog.py` (`RunLog.log`, `RunLog.tee`) + `tests/test_runlog.py`. Speech and
Clock-A notebooks now wrap each step in `with log.tee(...)`; output is appended to
`Alzheimers/logs/speech.log` and `Alzheimers/clock_out/clock.log` on Drive. The image grid is
not logged. Checkpoint/resume for Crop-Net and ViT will be written with those notebooks.
**Note:** Clock part A is CPU work (GPU only for parts B and C); Speech's networks are tiny,
the slow part is CPU ComParE extraction.

---

## 2026-10-12 (end) — Drive full: outputs off Drive

**Decided:** Drive is full, so **no results are written to Drive**. Notebooks print results
(user pastes them to Claude); logs/CSVs go to Colab's temporary disk `/content/out`, with an
optional `files.download` cell. Input data still read from Drive. `speech.ipynb`, `clock.ipynb`
changed. **Open:** Crop-Net / ViT checkpoints (~hundreds of MB each) need either Drive space
or accept losing progress on disconnect - decide when those notebooks are written.

---

## 2026-10-12 (night) — No Drive at all; upload per session

**Decided:** Colab does **not** mount Drive (storage). Each new Colab session the user uploads
`code.tar` + the data tar into the Files panel; notebooks unpack to `/content`. Results are
printed/pasted or downloaded to `D:\Alzheimers\results\`; checkpoints (Crop-Net/ViT, later) to
`D:\Alzheimers\models\` and re-uploaded to resume.
**Changed:** `speech.ipynb`, `clock.ipynb` setup cells (no `drive.mount`); `.gitignore`
(+`colab_upload/`, `.pytest_cache/`); deleted `requirements-local.lock.txt`, `.claude-dev-helper/`,
`__pycache__`/`.pytest_cache`. Bundles built locally in `colab_upload/` (gitignored).

---

## 2026-10-12 (close) — Where to resume

**Next, in order:**
1. **Eye on Kaggle** (start here): private datasets `adem-test` (ADEM_TEST folder with AD/ and NC/)
   and `alz-code` (colab_upload/code.tar or common/ + eye/). Import `notebooks/eye.ipynb`, fix
   `REPO` / `DATA` in cell 1 (Kaggle inputs are read-only under /kaggle/input; code must be
   unpacked to /kaggle/working/Alzheimers). Paste each SEND BACK output to Claude.
   Option offered: rewrite eye cell 1 to unpack code.tar itself, like the Colab notebooks.
2. **Speech on Colab:** upload code.tar + adress_train.tar to /content, run, paste results.
3. **Clock part A on Colab:** upload code.tar + NHATS.tar (2 GB). Option: small bundle with only
   survey files + zip listings for step 1.
4. Motor: no data; only the synthetic sanity cell.

**Per Colab session:** re-upload the tar files (disconnect wipes /content); download logs to
D:\Alzheimers\results\ and later checkpoints to D:\Alzheimers\models\.
**Rebuild code bundle after code changes:** `tar -cf colab_upload/code.tar common speech clock eye motor`
**Uncommitted:** all changes from this session (notebooks, docs, .gitignore, lock-file delete).

---

## 2026-10-12 (after close) — All four on Colab; per-modality upload files

**Decided:** ADEM_TEST would not upload to Kaggle (745 small files), so **Eye and Motor also run
on Colab**; all four notebooks now use the same pattern (upload tars to `/content`, unpack).
**Changed:** upload files in `colab_upload/` renamed per modality, each notebook needs only its own:
`clock_code.tar` + `clock_data.tar` (NHATS), `speech_code.tar` + `speech_data.tar` (ADReSS train),
`eye_code.tar` + `eye_data.tar` (ADEM_TEST, 35 MB), `motor_code.tar` (no data). Each code tar holds
`common/` + that modality only. Old `code.tar`, `adress_train.tar`, `NHATS.tar` removed.
`eye.ipynb`, `motor.ipynb` now have Colab setup cells; `speech`/`clock` file names updated; PLAN
steps 3 and 5 updated.
**Rebuild a code tar:** `tar -cf colab_upload/<m>_code.tar --exclude=__pycache__ common <m>`
**Not yet run** on Colab (tars verified by listing only).

---

## 2026-10-12 (evening) — Eye pilot run and detector check

**Ran (Colab):** Eye notebook on ADEM_TEST. Pilot LOSO n=20: balanced accuracy 0.300 (0.101-0.505),
AUC 0.240 (0.000-0.510); per task 0.30 / 0.45 / 0.40. All CIs include 0.5 → inconclusive.
**Detector check** (new last cell of `eye.ipynb`): threshold above max velocity in 28/60 recordings →
0 saccades by construction. Cause is the published mean+6SD rule on ~0.3 s recordings; my guess
that the start-at-max iteration was the cause was only partly right (the paper's start gives the
same first step).
**Decided:** no detector change, no tuning on the same 20 subjects; Eye stays an inconclusive pilot
with webcam transfer unvalidated. Output is flagged / not flagged only, no stage (rule 3).
**Changed:** `notebooks/eye.ipynb` (detector-check cells), `FINDINGS.md` (new 8b).
**Open:** (a) Eye notebook and Motor have no log / download cell yet. (b) How the live app stores
per-session results is undecided; proposal: local-only per-session JSON, no names, no uploads.
**Next:** Speech on Colab (`speech_code.tar` + `speech_data.tar`), then Clock part A, then Motor
synthetic check. Still uncommitted: all of today's changes.

---

## 2026-10-17 — Eye/Motor logging, bundles rebuilt, commit

**Changed:** `eye.ipynb` and `motor.ipynb` now create a `RunLog` (`/content/out/<m>.log`), wrap their
result cells in `log.tee(...)`, and download the log (eye: with the ONNX files). `common/runlog.py`
docstring says session disk, not Drive. All four `colab_upload/<m>_code.tar` rebuilt (runlog change).
**Checked (local, no ML):** all packages compile; `speech_data.tar` holds 54 cc + 54 cd transcripts and
wavs, so Speech is ready to run on Colab. Nothing new was run on Colab.
**Next:** Speech on Colab, then Clock part A, then Motor synthetic check.

---

## 2026-10-17 (later) — Speech ran on Colab

**Ran:** `speech.ipynb` end to end on Colab (uploads worked after disabling a browser privacy/tracking
shield that blocked them). Results in new FINDINGS section 9: ensemble balanced accuracy 0.759
(0.675-0.834), Disfluency 0.759, Interventions 0.704, Acoustic 0.583 (CI includes 0.5). n=108, 5-fold CV.
**Changed:** `FINDINGS.md` (+section 9), `PLAN.md` results table (Speech row filled).
**Note:** `speech_oof.csv` exists in Colab `/content/out` but the notebook only downloads the log; add it
to the download cell. Not run yet: domain-shift check for Speech.
**Next:** Clock part A on Colab, then Motor synthetic check; add oof CSV to speech download cell.

---

## 2026-10-17 (export) — Speech ONNX export written (not yet run)

**Added:** `speech/export.py` (final fit on all 108, ONNX for Disfluency, Acoustic prep = scaler+PCA, Acoustic net, parity checks, `speech_features.json`); `speech.ipynb` sections 6-7 (export + downloads). `speech_code.tar` rebuilt.
**Not exported:** Interventions (research-only). **Still open for Speech:** Whisper-transcript features validated against human transcripts; domain shift (rule 7).
**Eye:** export cells exist in `eye.ipynb`, but `models/` is empty, so no Eye ONNX has been saved locally yet.
