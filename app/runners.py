"""ONNX runners for the modalities that need a model. Not wired into the server yet.

Status (2026-10-18; clock moved to app/clock.py):
  eye     models/eye_flag.onnx + eye_features.json exist. Missing: gaze capture (MediaPipe/WebGazer) and `eye.features` from live
          gaze; the pilot is inconclusive and webcam transfer is unvalidated.
  speech  DONE as EXPERIMENTAL: see app/speech.py (Whisper small + models/speech_whisper.onnx, FINDINGS 9d). The older
          speech_disfluency/acoustic ONNX files need CHAT transcripts and stay unused.
  clock   DONE: see app/clock.py (photo cleaning + 3-step Crop-Net ONNX crop + ViT ONNX), served at /api/clock.
"""


def run_eye(*args, **kwargs):
    raise NotImplementedError("eye capture is not built yet")


def run_speech(*args, **kwargs):
    raise NotImplementedError("live speech is research only")


def run_clock(*args, **kwargs):
    raise NotImplementedError("waiting for the Clock models from Colab")
