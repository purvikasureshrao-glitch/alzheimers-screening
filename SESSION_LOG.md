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
