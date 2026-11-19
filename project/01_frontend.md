# 1. Front end — what the person sees

The **front end** is the part that runs inside the web browser (Chrome or Edge). It is the screens, buttons,
drawing area and camera/microphone access. Nothing here needs installing: it is plain HTML, CSS and JavaScript,
with no framework and no build step.

Folder: `app/static/`

## The files

| File | What it does |
|---|---|
| `index.html` | The structure of every screen (see below) |
| `style.css` | Colours, layout, light/dark look |
| `app.js` | The main logic: moves between screens, the clock canvas, the tapping test, microphone recording, sending data to the server, the doctor screens |
| `eye.js` | The eye test: webcam, face/iris tracking, calibration, recording where the eyes look |
| `cookie_theft.png` | The picture the person describes in the speech test |
| `vendor/` | MediaPipe (Google's face-tracking library) and its model, stored locally so the eye test works **without internet** |

## The screens

1. **Landing** — two big buttons: "I'm a patient" and "I'm a doctor".
2. **Patient questionnaire** — name plus 15 questions (age, gender, education, blood pressure, diabetes, family history, smoking, head injury, hearing loss, mood, activity, etc.). Every question is required.
3. **Four tasks**, always in this order: **1 Clock → 2 Finger tapping → 3 Eye movement → 4 Speech.**
4. **Results** — each test shows its own result, then a written summary.
5. **Doctor area** — log in / sign up / reset password, then search a patient by name or code and review everything.

## How the page talks to the server

The page never calculates a result itself. It collects raw input (a drawing, tap times, gaze points, an audio clip),
sends it as JSON to the local server (`http://127.0.0.1:8765`), and shows what comes back.
It never sees the database key.

## Each task, from the browser's side

- **Clock:** the person draws with mouse, finger or pen on an HTML `<canvas>`, or uploads a photo. The image is sent as a JPEG.
- **Finger tapping:** one big button. The page records the time of every tap for 30 seconds and shows a countdown bar.
- **Eye:** the page asks for webcam permission, runs a 9-dot calibration, then shows three small tasks while recording gaze. Details: [06_eye_movement.md](06_eye_movement.md).
- **Speech:** asks for microphone permission, records with the browser's `MediaRecorder`, and sends the audio.

## Things worth knowing

- The browser asks permission for camera and microphone. The person must allow both.
- The pages and libraries are served by the local Python server, so everything works offline (except the optional Supabase storage and the first-time model downloads).
- The doctor view can replay the saved drawing, tap chart, gaze path and recording.
