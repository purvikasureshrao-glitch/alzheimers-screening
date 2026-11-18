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

---

## 2026-10-17 (speech saved) — Speech ONNX export done

**Ran (Colab):** `speech.ipynb` section 6. Parity: scaler+PCA max |ONNX - sklearn| = 1.5e-05;
Disfluency and Acoustic ONNX flags equal PyTorch on 108/108 (max |dP| 1.8e-07 / 6e-08).
**Saved:** `models/speech_disfluency.onnx`, `speech_acoustic.onnx`, `speech_acoustic_prep.onnx`,
`speech_features.json`; `results/speech.log`, `speech_oof.csv` (from `output/speech_results.zip`).
Trained on all 108; accuracy statement is still only the 5-fold CV (FINDINGS 9).
**Open for Speech:** Whisper-transcript feature check; domain shift. **Next:** Eye export re-run on Colab, then Clock part A.

---

## 2026-10-17 (whisper) — Whisper transcript check written (not yet run)

**Added:** `speech/whisper_features.py` (5 features recoverable from Whisper words + timestamps: fillers, mhm/hm, repetition, pause >= 0.5 s [fixed in advance, not tuned], word rate; intervention rate, retracing, replacement, incomplete, prolongation, trailing-off dropped, not recoverable), `disfluency_whisper` network (5 inputs) in `models.py`, `cross_validate_single` in `cv.py`, `notebooks/speech_whisper.ipynb`. `speech_code.tar` rebuilt.
**Notebook does:** Whisper `small` (fixed) with word timestamps on all 108 recordings; human-vs-Whisper feature table (Spearman, means, zero fractions); 5-fold CV of the 5-feature network, same protocol as FINDINGS 9. Word transcripts are deleted before the download zip.
**Next:** user runs it on Colab, pastes the table and CV lines; record in FINDINGS 9b. Then app model for live Speech = whichever Disfluency variant matches the live input (the 11-feature one cannot be fed from Whisper).

---

## 2026-10-17 (whisper result)

**Ran (Colab):** `speech_whisper.ipynb`, Whisper `small`, 108 files, 464 s. Results in FINDINGS 9b:
5-feature network balanced accuracy 0.639 (0.546-0.731) vs 0.759 for human-transcript features; Whisper
drops most fillers/repeats; pause feature uncorrelated with human marks (rho 0.02).
**Saved:** `results/speech_whisper*.csv/.log`. **Open:** whether a larger Whisper model or a verbatim ASR
recovers more (would be a second, pre-stated model size, reported next to `small`, not picked on CV).
Not exported to ONNX: the 5-feature network is not worth shipping at this level.

---

## 2026-10-17 (whisper medium) — Speech closed as research-only for live use

**Ran (Colab):** Whisper `medium`. 5-feature network balanced accuracy 0.556 (0.460-0.650); CI includes 0.5.
Recorded in FINDINGS 9c; PLAN B1 live-use row marked superseded.
**Decision:** Speech is finished as a *research* modality: human-transcript CV result (FINDINGS 9) + saved ONNX
for Disfluency/Acoustic (feature input needs CHAT-style transcripts, so not usable from a mic). No
patient-facing Speech claim. In the app Speech shows "not validated / research only" (to confirm when the app is built).
**Next:** Eye export re-run on Colab, then Clock part A, then Motor.

---

## 2026-10-17 (eye saved) — Eye ONNX export done

**Ran (Colab):** `eye.ipynb` re-run; pilot numbers reproduced exactly (LOSO balanced accuracy 0.300, 0.101-0.505,
n=20; detector check 28/60 recordings with 0 saccades). Export parity: ONNX == sklearn flags on 20/20
(17 rows with NaN features, handled by the imputer in the ONNX).
**Saved:** `models/eye_flag.onnx`, `models/eye_features.json`, `results/eye.log`. Trained on all 20 subjects,
so its flags on them are not an accuracy. Eye stays an inconclusive pilot; webcam transfer unvalidated.
**Next:** Clock part A on Colab.

---

## 2026-10-17 (clock bundle) — lean NHATS bundle

**Changed:** `colab_upload/clock_data_lean.tar` (1.04 GB: 11 image zips + 11 Stata survey zips) replaces the 2.0 GB
`clock_data.tar` in `clock.ipynb`. `labels.py` reads only `*STATA*.zip`, so the SAS copies (~1 GB) were unused.
Not yet run on Colab (label counts should still be 54,027 images indexed).

---

## 2026-10-17 (motor reference) — Lima dataset does not match

**Checked:** Lima et al. 2024 is smartphone-touchscreen tapping, 30 s, not keyboard shift keys, 10 s, 4 conditions.
Withdrawn as a healthy reference for the Motor features (FINDINGS 2 note). Motor stays features-only until a
matching dataset exists.
**Decided with user:** four-modality combination = documented rule "2+ of 4 flagged", no accuracy claim.
Clock outputs the NHATS-provided classification/score (no invented stages). Eye/Speech flag only.

---

## 2026-10-17 (motor reference built)

**Added:** `motor/reference.py` (Lima healthy reference, 3 shared features, cut-offs 5th/95th percentile, flag = 2 of 3),
`models/motor_reference.json` (git-ignored), FINDINGS section 10. User approved using Lima with the domain shift stated.
**Measured:** 6% of the reference flagged, but 33% (5/15) of those aged 50+; age confound strong. The app must show
this as "outside healthy reference range (unvalidated, young reference)", never as an Alzheimer's result.
**Clock:** user's Colab `tar -x` of `clock_data_lean.tar` failed (exit 2) = truncated upload again; re-upload and check size 1,044,080,640.

---

## 2026-10-17 (clock part A done)

**Ran (Colab):** Clock Part A (H1-H2). Results in FINDINGS 3d. 54,027 indexed, 53,914 kept, 0 SPIDs across splits.
**Open:** step 3 cleaning grid (H3 name-box removal + rough crop): user reports the crop looks "somewhat cropped" -
details asked. **Next:** write Part B (Crop-Net) and Part C (ViT) as one run-all notebook with checkpoints.

---

## 2026-10-17 (clock run-all written, NOT run)

**Added (no local test possible: no numpy/torch locally):** `clock/cropnet/{model,synth,train}.py`, `clock/crop.py`,
`clock/vit/{model,train}.py`, `clock/pipeline.py`, `preprocess.rough_crop_512`, `notebooks/clock_all.ipynb` (labels > split >
smoke test > full run > two download zips). Deviations in FINDINGS 3e. `clock_code.tar` rebuilt.
**First thing on Colab:** the smoke test; any bug in the untested code surfaces there in about 10 min.

---

## 2026-10-18 — Motor on Lima protocol; early Crop-Net download

**Decided (user):** Motor uses the Lima protocol (30 s, one hand) with an age-adjusted healthy reference; no training.
**Changed:** `motor/reference.py` rewritten (age-adjusted, 30 s); `models/motor_reference.json` rebuilt; FINDINGS 10b; PLAN B4 note.
Flags in-sample: 2/176 overall, 0/15 aged 50+. Sensitivity unknown.
**Clock:** `clock_all.ipynb` now splits the full run into part 1 (prep, Crop-Net, crop) > Crop-Net checkpoint download > part 2 (ViT,
metrics, ONNX). `clock_code.tar` rebuilt. User starting the run; PC must not sleep (Colab disconnects otherwise).

---

## 2026-10-18 (app skeleton)

**Added:** `app/` (stdlib only, no installs): `schema.py` (Result, display order), `evidence.py` (each modality's measured
evidence + caveat, copied from FINDINGS), `combine.py` (rule: 2+ of 4 flagged -> follow-up, no score, no accuracy claim),
`motor.py` (taps + age -> flag via `motor/reference.py` and `models/motor_reference.json`), `summary.py` (Ollama prose summary,
JSON-validated: rejects diagnosis/stage/severity wording and numbers not in the input, falls back to a fixed template; never
called unless all four screens ran), `server.py` (local HTTP on 127.0.0.1:8765, no request log), `static/index.html`
(30 s tapping test, four cards in order Clock > Motor > Eye > Speech, summary button), `runners.py` (stubs for eye/speech/clock).
`tests/test_app.py`: 14 tests pass (`PYTHONPATH=. python tests/test_app.py`).
**Run:** `python -m app.server` from the repo root, open http://127.0.0.1:8765. Needs `models/motor_reference.json`
(`python motor/reference.py datasets/Lima.csv models/motor_reference.json`).
**Not built:** eye capture (MediaPipe/WebGazer) + ONNX call, clock photo cleaning + ONNX call (waits for Colab), live speech (research only),
ONNX runtime install, "see what was captured" views, Ollama model not installed/tested.

---

## 2026-10-18 (clock complete)

**Discussed:** user slept through the Colab run; the saved zips show it finished (part 1 and part 2). No rerun needed.
**Decided (user):** keep the Crop-Net as trained and flag it as a limitation (option a); no retrain.
**Result:** test n=1089 people: balanced accuracy 0.654 (0.605-0.707), AUC 0.707, F1 0.157; flags 32.8% vs 4.8% prevalence.
Crop-Net validation error never moved (0.16222) and 90% of crops touch the border. Recorded in FINDINGS 3f; PLAN results table filled.
**Changed:** extracted `models/clock_{vit,cropnet}.onnx` + `clock_features.json` and `results/clock*.csv/log`; `clock/crop.py` torch import made lazy;
new `app/clock.py` (photo -> cleaning -> 3-step ONNX crop -> ViT -> flag at 0.060), `/api/clock` route, upload control on the clock card,
clock evidence/caveat text in `app/evidence.py`, 4 new tests (18 pass with Python 3.13: `PYTHONPATH=. python tests/test_app.py`).
Local deps installed into Python 3.13: onnxruntime, numpy, scipy, pillow, opencv-python-headless (inference only).
**Open / next:** photo path unvalidated (no real photos; only a synthetic drawing was tested); heat-map upsampling differs from torch (not compared);
eye capture, live speech, Ollama still not built. Commit not made yet.

---

## 2026-10-18 (eye webcam path, experimental)

**Discussed:** the eye model needs a ~333 Hz research tracker; a webcam gives ~9 samples per 0.3 s task. User chose option 2 (build webcam capture anyway)
and asked for a document stating the limit and that the result is flagged as such. Web search for literature: FINDINGS 8c (summaries only, papers not opened).
**Decided:** eye result is shown as EXPERIMENTAL and **not counted** in the 2-of-4 rule (my recommendation, user said "go ahead with plan 2"); left/right-eye
features left missing, not faked; `not_run` if a task has fewer than 4 valid samples.
**Changed:** `app/eye.py` (gaze -> `eye.features.extract` -> `eye_flag.onnx`), `app/static/eye.js` (MediaPipe iris + 9-point calibration + 3 tasks, 300 ms each),
eye card in `index.html`, `/api/eye` and `/static/` file serving in `server.py` (traversal blocked), `Result.counts`, `combine` (excluded list + wording),
eye caveat in `evidence.py`, FINDINGS 8c, PLAN eye row, 6 new tests (24 pass, Python 3.13).
**Vendored (git-ignored, 13 MB):** `app/static/vendor/` = `@mediapipe/tasks-vision@0.10.14` (`vision_bundle.mjs`, `wasm/vision_wasm_internal.{js,wasm}` from
cdn.jsdelivr.net) + `face_landmarker.task` (storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/). Redownload if missing.
**Open:** `eye.js` never run on a real webcam (try it: start the server, click "Start the eye test"); calibration error is shown but not used as a gate;
stimuli are ours; eye accuracy remains inconclusive (CI 0.101-0.505) and webcam accuracy unknown.

---

## 2026-10-18 (speech live path, experimental)

**Decided (user):** Plan A: live speech as EXPERIMENTAL, not counted in the 2-of-4 rule. Task is a spoken kitchen-scene description (Cookie Theft picture has rights questions).
**Changed:** `speech/export.py` `export_whisper()`; `notebooks/speech_live_export.ipynb` (Colab, CPU, features + labels only, no audio); `colab_upload/speech_code.tar` rebuilt
and `colab_upload/speech_whisper_{features,oof}.csv` copied in; `app/speech.py` (faster-whisper small int8 -> 5 features -> `models/speech_whisper.onnx`);
`/api/speech`; 60 s recording UI on the speech card; speech evidence/caveat text; FINDINGS 9d; 4 new tests (28 run, 1 skipped until the ONNX exists).
Installed `faster-whisper` into Python 3.13; Whisper `small` (~460 MB) is now cached by the first test run.
**NEXT (user, Colab):** open `notebooks/speech_live_export.ipynb`, upload `speech_code.tar`, `speech_whisper_features.csv`, `speech_whisper_oof.csv` from `colab_upload/`,
run all, send back the 2 printed lines, save `speech_whisper.onnx` + `speech_whisper_features.json` into `D:\Alzheimers\models\`.
**Open:** browser recording and end-to-end Whisper path never run on a real microphone; speech remains unvalidated beyond the 108-person CV.

---

## 2026-10-18 (speech model exported)

**Ran (Colab):** `speech_live_export.ipynb`: features (108, 5), classes 54/54, all finite; ONNX flag == PyTorch flag on 108/108, max |dP| 1.19e-07.
**Saved:** `models/speech_whisper.onnx` + `speech_whisper_features.json` (from `output/speech_live_results.zip`). All 28 app tests pass (none skipped).
Trained on all 108 subjects, so its flags on them are not an accuracy; the only accuracy is the CV 0.639 (0.546-0.731).
**Next:** Ollama summary + end-to-end run of the app on a real webcam/microphone.

---

## 2026-10-18 (Ollama set up, summary tested)

**Installed:** Ollama 0.40.6 (official signed installer, `%LOCALAPPDATA%\Programs\Ollama`), model `llama3.1:8b` (4.9 GB). Machine: 15.6 GB RAM, no discrete GPU, so CPU only.
**Tested (synthetic results, no patient data):** first run took 84 s and the model returned nested objects without `overall`; the validator rejected it and the
fixed template was used (the safe fallback worked). **Fix:** `app/summary.py` now passes a JSON schema as Ollama `format` (shape enforced) and always sets
`overall` to the fixed rule text (it carries the "no accuracy claim" line). Re-run: accepted by the validator, 38 s. 29 tests pass.
**Note:** the model's paragraphs shorten the evidence (it dropped the CIs and n in its wording). The validator only checks banned wording and invented numbers,
not omissions; the card for each modality still shows its full evidence. Wall time 40-90 s on CPU, so the UI must say "Writing...".
**Next:** real webcam/microphone test; commit.

---

## 2026-10-18 (ONNX check + frontend finished)

**ONNX:** every model the app needs exists: `clock_vit`, `clock_cropnet`, `eye_flag`, `speech_whisper` (+ their json). Motor needs none (reference table).
`speech_disfluency/acoustic(_prep).onnx` stay unused (need CHAT transcripts). New `GET /api/status` reports missing model files, Ollama state and the vendored MediaPipe files.
**Frontend rewritten (`app/static/index.html`):** status banner, "n of 4 done" strip, numbered cards in order Clock > Motor > Eye > Speech, "Experimental - not counted" tags,
clock shows the cropped image the model saw, speech shows the transcript (page only, not stored), eye result shows calibration error, summary shows a running timer,
Copy / Print / Start over buttons, print stylesheet, dark mode, aria-live messages, permission-denied messages.
**Also:** `run_app.bat` (Python 3.13, opens the browser), `requirements-local.txt` updated, `clock.analyse`, `speech.analyse`, `eye.score(calibration_pct)`.
**Tested:** 31 tests pass; page and eye.js JavaScript syntax-checked with node. **Not tested:** the page in a real browser (Chrome extension not connected), webcam, microphone.
**Next:** device test together; commit.

---

## 2026-10-18 (end of session: committed, resume point)

**Committed:** `e7798ba` (clock, eye, speech paths; Ollama summary fix; frontend; docs). Not pushed. Left untracked on purpose: `output/*.zip` (backups; models are
already in `models/`) and the per-image NHATS files `results/split.csv`, `clock_test_preds.csv`, `crop_qc.csv` (participant-level; add to `.gitignore` before any push).
**State:** all four modalities have an app path. Counted in the 2-of-4 rule: Clock, Motor. Shown but not counted: Eye (webcam, experimental), Speech (Whisper, experimental).
Ollama 0.40.6 + `llama3.1:8b` installed; summary works (38-90 s on CPU) and falls back to a fixed template if the model is off or breaks a rule. 31 tests pass.
**Resume here (device test):** run `run_app.bat` (Python 3.13), check the banner is empty, then try: clock photo upload, 30 s tapping, eye test (webcam - never run on
real hardware), speech 60 s (microphone - never run; Whisper small is cached), then "Write the summary". Send back any error text (F12 > Console).
**Still unvalidated:** clock on photos (only a synthetic drawing was tried); eye and speech on live devices; all domain shift (rule 7). Crop-Net did not learn (FINDINGS 3f).

---

## 2026-10-18 (new patient/doctor app flow)

**Asked (user):** landing page with Patient / Doctor; patient: questionnaire (name, ethnicity, BP, diabetes, gender, risk factors) -> clock drawn with the mouse -> motor -> eye -> speech with the Cookie Theft picture -> LLM summary; everything stored in Supabase; doctor logs in, searches by name/ID, reviews. Smooth, aesthetic UI.
**Decided (user):** simple doctor login (not high security); clock drawn on a canvas, saved as JPEG and sent through the existing clock pipeline; Cookie Theft picture; Supabase setup steps wanted (no project yet).
**Changed:** `app/store.py` (Supabase REST via server-side service key in `.env`, else `data/local_store.json`; doctor login from `.env`), routes `/api/patient`, `/api/save`, `/api/doctor/{login,search,record}` in `server.py`, `index.html` rewritten (landing, stepper, questionnaire, canvas clock, doctor view), `supabase/schema.sql` + `SETUP.md` (RLS on, no public policies), `.gitignore` (.env, data/, NHATS per-image CSVs), `tests/test_store.py` (3), `test_app.py` updated. App tests pass (34).
**Rules kept:** questionnaire answers are stored and shown only; they never change a result (no fusion). Summary wording still "follow-up recommended", no diagnosis. Eye/speech still not counted.
**Open:** (1) mouse-drawn clock is a different input from the NHATS photos the model was trained on: unvalidated, domain shift (rule 7). (2) Cookie Theft image file not in repo: user must add `app/static/cookie_theft.jpg` and confirm they may display it. (3) Supabase project + `.env` not yet created. (4) Page never viewed in a browser by me (Chrome extension not connected); webcam/mic untested. (5) EVIDENCE/CAVEAT text for clock should mention mouse input.

---

## 2026-10-19 — Supabase live, full browser self-check

**Done:** Supabase connected (`.env`, user-made; I never read the key; `app/store_check.py` prints no secrets). Fixed: pasted URL with `/rest/v1/` suffix; network/Supabase errors now return a readable JSON error instead of dropping the connection ("Failed to fetch"); `confirm()`/`alert()` replaced by in-page messages (Brave can block popups); `run_app.bat` starts Ollama and waits before opening the browser; Cookie Theft picture (Heyselaar 2021, CC BY-NC, from OSF 459y2, redrawn version, NOT the ADReSS original) added at `app/static/cookie_theft.png` (git-ignored) with credit line; clock caveat now mentions mouse drawings.
**Self-test in Brave (extension):** landing, questionnaire, real mouse drawing on the clock canvas -> model, 30 s motor (simulated taps), speech picture, final summary + save, record read back from Supabase, doctor wrong-password rejected, brand-click leave prompt. No console errors. Summary uses the fixed template unless all four screens are done (by design).
**Not tested:** eye test on a webcam, speech on a microphone, doctor login with the real password (not read), anything on a second device.
**Test records left in Supabase (delete in Table Editor):** Storage Check, Net Test, Test Patient, Auto Test patients.
**Next (user, 2026-10-20):** run the full flow incl. webcam and microphone; regenerate the Supabase key afterwards.

---

## 2026-10-19 (end of session: resume point)

**State:** all four modalities + patient/doctor flow + Supabase working; committed through `7a47ba1` (not pushed). Server stopped.
**Resume here (2026-10-20, user runs the check):** `run_app.bat` (keep window open) -> full flow incl. webcam eye test and microphone speech test -> doctor login with own credentials -> send any error text (F12 > Console or the red message under a form).
**Then:** delete test patients in Supabase (Storage Check, Net Test, Test Patient, Auto Test); regenerate the Supabase key; fix whatever the device test finds.
**Still unvalidated (rule 7):** mouse-drawn clock, webcam eye, microphone speech with the redrawn Cookie Theft picture. Eye and speech are not counted in the 2-of-4 rule.

---

## 2026-10-20 (UI redesign + user-requested changes)

**Asked (user):** patient ID visible; all questionnaire answers required; remove "limits/evidence" and "experimental/not counted" from the pages; simple "X: flagged / not flagged" summary (no rule text); motor flag: slow = flagged, faster than healthy never flagged; eye stimuli 3 s; doctor can replay clock drawing, gaze path, audio, tap chart; forgot-password; nicer UI with dark/light toggle and animated background on sign-up/log-in only.
**Changed:** `app/combine.py` (text = per-modality flag + "N of M flagged"; `follow_up` kept only for box colour), `app/summary.py` (prompt/template without evidence/caveat), `app/motor.py` (**deviation from `motor/reference.flag`**: slow tps flagged alone; tps above the 95th-percentile never flagged; otherwise 2-of-3 as before; cut-offs unchanged), `app/eye.py` (stimulus recorded 3 s, **only first 300 ms scored** = training window), eye/speech no longer "experimental / not counted", `app/store.py` (required answers, partial-ID search, doctor full name, recovery code + reset, lock-out after 5 wrong tries), `app/static/{index.html,style.css,app.js}` (rewrite; old page in `backups/index_before_redesign.html`), `supabase/schema.sql` (3 new `doctors` columns: run the ALTER lines once). Results now carry an `artifact` (clock preview, tap times, gaze path, audio + transcript) saved in `screenings.results`; transcript is now stored (earlier: page only).
**Decisions to revisit:** eye and speech now count like clock/motor in the "N flagged" line (user wish); evidence/limits still shown to doctors under "About this measure". 40 tests pass. Page NOT viewed in a browser this session (extension not connected).

---

## 2026-10-20 (speech improvement notebook, not yet run)

**Asked (user):** raise clock and speech accuracy, no extra false alarms, no overfitting. Speech first.
**Built:** `speech/improve_features.py` (patient-only filtering from CHAT time marks, 11 extended word-based features, picture information units; 5 tests in `tests/test_speech_improve.py`), `scripts/build_speech_improve_nb.py` -> `notebooks/speech_improve.ipynb` (Colab, upload `colab_upload/speech_data.tar` only; code embedded).
**Pre-registered in the notebook:** 8 candidates (baseline A = today's app path), 5x repeated 5-fold CV, inner-CV regularisation, threshold 0.5, selection rule (specificity >= baseline, stable), paired-bootstrap CI vs baseline, shuffled-label check. CrisperWhisper is optional (may need an HF token; check its licence).
**Facts found:** all 1,492 *PAR utterances in the 108 train transcripts have time marks, so patient-only audio can be cut. ADReSS test labels are NOT in the folder (`test_results.txt` blank), so no held-out test yet.
**Not done:** notebook never run on real data (logic smoke-tested on random numbers only). Nothing in the app changed. Clock notebook is next.

---

## 2026-10-20 (clock improvement notebook, not yet run; resume tomorrow 2026-10-21)

**Built:** `clock/improve.py` (plain crop without Crop-Net, mouse-style training augmentation, an independent mouse-style probe, specificity-targeted threshold), `scripts/build_clock_improve_nb.py` -> `notebooks/clock_improve.ipynb` (Colab GPU, NHATS Colab only, no Gemini; code embedded). Uploads: `colab_upload/clock_code.tar`, `clock_data_lean.tar`, `output/clock_models.zip`; optional `my_drawings.zip` (JPEGs from the app).
**Pre-registered in the notebook:** variants V0 (current), V1 (plain crop + retrained ViT), V2 (V1 + mouse augmentation); same person-grouped split (seed 0); thresholds from validation only at two operating points (BA = today's, SPEC90 = >=90% healthy unflagged); eligibility = specificity >= V0, BA not worse than V0 by >0.02 (paired bootstrap), probed healthy-flag rate <= V0. Probe family differs from the training augmentation. Expect 4-6 h on a T4.
**Not done:** torch is not installed locally, so the new torch code was only syntax-checked (the notebook starts with a GPU smoke test that fails fast). Nothing in the app changed.

**Resume here (2026-10-21), in this order:**
1. Run `notebooks/speech_improve.ipynb` on Colab (upload `speech_data.tar`), send back the table + selection output (cells marked SEND BACK).
2. Run `notebooks/clock_improve.ipynb` on Colab, send back the table + selection output.
3. I switch the app to a winner ONLY if the rule says supported/eligible (speech: new ONNX + patient-only feature code in `app/speech.py`; clock: plain crop in `app/clock.py` + new threshold). If nothing wins, the app stays as is.
4. Still open from the UI session: view the new front end in a browser, run the Supabase ALTER lines, get a new recovery code for the existing doctor account, delete test patients in Supabase.

---

## 2026-10-20 (later) — Clock Colab fix, report chapters 3-5 drafted
**Discussed:** clock notebook upload kept failing (`BadZipFile` on `clock_models.zip`); what is left while the clock run goes; Supabase delete error; college report (Ch 3-5) and PPT.
**Decided (user):** wait for the clock run and the speech verdict before the full report and PPT. Report format: Word .docx, A4, Times New Roman 12, 1.5 spacing; diagrams drawn as images; Ch 4 results wait for final numbers; comparison with existing work is qualitative plus like-for-like only (no paper headline figures); screenshots as numbered placeholders.
**Changed:**
- Local `output/clock_models.zip` tested fine (5 members, no errors), so the Colab copy was corrupted. `colab_upload/clock_models_parts/` = `cropnet_best.pt` and `vit_full_best.pt` split into <=80 MB parts + `manifest.json` (SHA-256). `scripts/build_clock_improve_nb.py` first cell now rejoins and verifies the parts (no zip); `notebooks/clock_improve.ipynb` rebuilt. Not run in Colab by me.
- New `report/` (untracked): `diagrams.py`, `charts.py`, `eqs.py`, `docbuild.py`, `chapter3.py`, `chapter4.py`, `chapter5.py`, `build_report.py`, `topdf.py`; outputs `Chapter3.docx`, `Chapters_3_to_5.docx` (46 pages: Ch3 28, Ch4 14, Ch5 3), `fig/`. Needs a venv with python-docx, matplotlib, pywin32 (Word used for page counts); the one used was in the session scratchpad.
- Memory note `report-ppt-plan.md` added.
- No change to app code, PLAN, FINDINGS or tracked docs.
**Facts found:** `output/speech_improve_results.zip` table (n=108, 5x5 CV): A 0.611 (0.516-0.700), B 0.667, B+ 0.750 (0.667-0.827), E_B 0.713 (0.623-0.793). By the notebook's stability rule (BA spread <= 0.04) only E_B qualifies (B+ 0.046, B 0.056); the notebook's own paired-bootstrap/shuffled-label output was not seen, so no winner is chosen. `PROJECT_OUTLINE.md` still says WebGazer for eye; the code uses MediaPipe Face Landmarker + 9-point ridge calibration (report follows the code).
**Open / next (in order):**
1. User sends the clock Colab results (variant table + selection) and the speech SEND BACK output.
2. Apply the selection rule; change the app and Ch 3 only if a winner qualifies.
3. User: device test (webcam eye, microphone speech, doctor login with the real password); delete test patients in Supabase (delete `screenings` rows first, then `patients`, because of the foreign key); regenerate the `service_role` key and update `.env`; run the Supabase ALTER lines if not done; new doctor recovery code.
4. Fill the yellow placeholders in Ch 4/5 (clock V0/V1/V2, speech verdict, device test, 12 screenshots, one sentence in the Ch 5 conclusion).
5. User sends the previous document + report rules; then build the full report and the PPT. User also has other tasks to hand over.
6. Many files are still uncommitted (app, notebooks, `clock/improve.py`, `report/`); commit when asked.
