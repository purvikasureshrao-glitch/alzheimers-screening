# 9. The whole app — how it all works together

## The idea in one paragraph

A person answers a questionnaire and does four short tasks (clock, finger tapping, eye movement, speech).
Each task is checked by its **own** method and gives its **own** result. A small local language model writes a
summary. A doctor can log in later and review everything. All the checking runs on the laptop.

## The picture

```
 BROWSER (front end)                     PYTHON SERVER (back end)                 STORAGE
 ───────────────────                     ────────────────────────                 ───────
 questionnaire ───── /api/patient ─────► store.py ───────────────────────────►  Supabase or
 clock drawing ───── /api/clock ───────► clock.py   (Crop-Net + ViT)             data/local_store.json
 finger taps ─────── /api/motor ───────► motor.py   (rules + reference)
 gaze points ─────── /api/eye ─────────► eye.py     (30 numbers + small model)
 voice recording ─── /api/speech ──────► speech.py  (Whisper + small model)
 all 4 results ───── /api/summary ─────► combine.py + summary.py (Ollama LLM)
 finished screening─ /api/save ────────► store.py
 doctor login/search /api/doctor/... ──► store.py
```

## One session, step by step

1. **Start.** Double-click `run_app.bat`. It starts Ollama (if installed), starts the server, and opens the browser at `http://127.0.0.1:8765`.
2. **Choose** "I'm a patient". Fill in the questionnaire. The server creates a patient record and a patient code.
3. **Clock.** Draw → image goes to the server → cleaned → cropped 3 times → ViT → flagged / not flagged. See [04_clock.md](04_clock.md).
4. **Finger tapping.** 30 seconds of taps → three numbers → compared with healthy people of the same age. See [05_motor_finger_tapping.md](05_motor_finger_tapping.md).
5. **Eye.** Webcam, 9-dot calibration, three tasks → 30 numbers → small model. See [06_eye_movement.md](06_eye_movement.md).
6. **Speech.** Describe the picture → Whisper → 5 numbers → small model. See [07_speech.md](07_speech.md).
7. **Summary.** The four results are counted ("n of 4 flagged") and the language model writes a short note, with safety checks. See [08_llm_summary.md](08_llm_summary.md).
8. **Save.** Results, summary and the clock drawing are stored (Supabase or the local file).
9. **Doctor later.** "I'm a doctor" → log in (or sign up the first time) → search by patient name or code → review all scores, the drawing, tap chart, gaze path and recording.

## The rules the project follows

- **The four tests are independent.** Each has its own paper, data and method. Nothing is combined into one score or one accuracy figure.
- **No diagnosis.** Results say whether something "resembles the patterns in the training data". Never a diagnosis, stage or severity.
- **Honest numbers.** Each result carries its own measured evidence and its limits (see the "Honest limits" part of each test file).
- **Privacy.** Models run on the laptop. Face video is never saved. The transcript is not stored. `.env` keys are never shared.
- **Training happened elsewhere.** Models were trained on Kaggle / Google Colab (free GPUs), exported to ONNX, and only the finished models are used here.

## Folder map

```
Alzheimers/
├─ app/            server (Python) + app/static (the web pages)
├─ clock/ eye/ motor/ speech/ common/   processing code shared with training
├─ models/         the trained ONNX models (too big for GitHub)
├─ supabase/       schema.sql and SETUP.md
├─ project/        these guides
├─ tests/          automated tests
├─ notebooks/ scripts/   Colab/Kaggle training notebooks and helpers
├─ datasets/ Paper/ College/   source data, papers and report files (not shared)
├─ requirements.txt  packages needed to run the app
├─ setup_new_pc.bat  one-time setup on a new PC
├─ run_app.bat       start the app
└─ .env.example      template for your own Supabase keys
```

## What's needed to run it

Windows 10/11 · about 10 GB free disk · internet for the first setup · webcam and microphone · Chrome or Edge.
Python 3.13 and the packages are installed by `setup_new_pc.bat`: `numpy`, `pillow`, `opencv-python-headless`,
`onnxruntime`, `faster-whisper`, `av`. Optional: Ollama + `llama3.1:8b` for the summary.
