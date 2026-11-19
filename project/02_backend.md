# 2. Back end — the Python server

The **back end** is a small Python program running on the same laptop. The browser sends it requests, it does the
heavy work (running the models), and sends back results.

Folder: `app/` · Start with `python -m app.server` (or double-click `run_app.bat`) · Address: `http://127.0.0.1:8765`

It uses Python's **built-in web server only** (no Flask or Django), so there is little to install.
`127.0.0.1` means "this computer only": nobody else on the network can reach the app.

## The files

| File | Job |
|---|---|
| `server.py` | Receives requests and sends each to the right module (the "router") |
| `clock.py` | Clock test: image → cleaning → crop → ViT model → result |
| `motor.py` | Finger tapping: tap times + age → result (no AI model) |
| `eye.py` | Eye test: gaze points → 30 numbers → small model → result |
| `speech.py` | Speech test: audio → Whisper → 5 numbers → small model → result |
| `combine.py` | Counts how many of the four tests were flagged (no averaging) |
| `summary.py` | Writes the plain-English summary using a local language model |
| `store.py` | Saves and reads patients, screenings and doctor accounts |
| `schema.py` | The shape of a result: modality, status, label, measured values, evidence, caveat |
| `evidence.py` | The "what this is based on" and "limits" text shown with each result |
| `runners.py`, `store_check.py` | Small helpers |

## The requests it understands (endpoints)

| Path | Does |
|---|---|
| `/api/patient` | Create a patient record, returns a patient code |
| `/api/clock`, `/api/motor`, `/api/eye`, `/api/speech` | Run one test and return its result |
| `/api/summary` | Count flags and write the summary |
| `/api/save` | Save the finished screening |
| `/api/doctor/login`, `/signup`, `/reset`, `/new_recovery` | Doctor accounts |
| `/api/doctor/search`, `/record` | Find a patient and load their record (login required) |
| `/api/status` | Which model files exist, whether Ollama is running, where records are stored (feeds the banner at the top of the page) |

## Design choices (and why)

- **Models load only when first used.** Starting the app is quick; the first clock or speech test takes a few seconds longer.
- **ONNX models.** Trained models are stored in the universal `.onnx` format and run with `onnxruntime`. No PyTorch and no graphics card needed on the laptop.
- **No request log.** Nothing about a person is written to a log file.
- **Friendly errors.** Bad input gives a clear message (400); not logged in (401); too many wrong passwords (429); storage problem (502). The server never just drops the connection.
- **Doctor sessions** are kept in memory for 8 hours; restarting the app logs doctors out. 5 wrong logins within 5 minutes temporarily blocks that username.
- **Same code as training.** `clock/`, `eye/`, `motor/`, `speech/` contain the processing code used when the models were trained. The app re-uses it, so live input is processed exactly like the training data.
