# 6. Eye movement test

**Task:** the person sits facing the laptop webcam and follows things on screen. How eyes jump and settle is
affected by brain health.

**Output:** *flagged* or *not flagged*. Code: `app/static/eye.js` (browser), `app/eye.py`, `eye/features.py` · Models: `models/eye_flag.onnx`, `models/eye_features.json`

## Method, step by step

1. **Track the eyes with the webcam.** MediaPipe Face Landmarker (Google's free library, stored in `app/static/vendor/`) finds the face and the iris. It runs inside the browser, offline. No video is uploaded or saved.
2. **Calibrate (9 dots).** The person looks at 9 red dots (a 3×3 grid). For each, the page learns how iris position maps to screen position. The average calibration error is shown as a percentage of screen width.
3. **Three short tasks**, each recorded for 3 seconds:
   - **Antisaccade:** look at a cross; when a dot appears on one side, look to the **opposite** side.
   - **Visual attention:** a picture with six coloured dots appears; look at it freely.
   - **Visual search:** find the one **T** among the **L**s.
4. **Turn gaze into numbers** (`eye/features.py`). For each task, 10 numbers: number of fixations, average fixation length, number of saccades (quick eye jumps), average jump size, peak jump speed, time to first jump, left-vs-right-eye disagreement (x, y), and spread of gaze (x, y). 3 tasks × 10 = **30 numbers**. Jumps and fixations are detected with an adaptive speed threshold (after Nyström & Holmqvist 2010).
5. **Model.** A small model (`eye_flag.onnx`) takes the 30 numbers and says flagged / not flagged.
   Only the **first 300 ms** after each task starts go to the model, because that is the window it was trained on.
   Numbers that can't be measured are left empty and filled in by the model's built-in imputer.

If too little gaze was captured (fewer than 4 samples in a task), the result is "not run".

## Where the training data came from

The ADEM eye-tracking dataset, recorded with a professional tracker at about 333 readings per second.

## Honest limits — please read

- A webcam gives about **30 readings per second**, so only about **9 points per task** reach the model. It cannot copy the fine jump measurements, and it has one combined gaze, not left and right eyes separately (those two numbers are always empty).
- Screen units in the training data are unpublished, so our pixel-based numbers don't match exactly.
- Pilot evidence: 20 people, leave-one-person-out, balanced accuracy **0.300 (95% CI 0.101–0.505)**. The range includes 0.5, which means **no detectable signal**. The eye test is the weakest part of the project and should be treated as experimental.
