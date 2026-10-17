"""ONNX runners for the modalities that need a model. Not wired into the server yet.

Status (2026-10-18):
  eye     models/eye_flag.onnx + eye_features.json exist. Missing: gaze capture (MediaPipe/WebGazer) and `eye.features` from live
          gaze; the pilot is inconclusive and webcam transfer is unvalidated.
  speech  models/speech_*.onnx exist but need CHAT-style transcripts; a Whisper transcript cannot feed them (FINDINGS 9b-9c).
          Live speech stays 'not run (research only)'.
  clock   waits for the Colab run: clock_vit.onnx + clock_cropnet.onnx + clock_features.json (threshold). Missing: photo cleaning
          path, which must reproduce clock.preprocess at 1/4 scale and the 3-step crop with the Crop-Net ONNX.
"""


def run_eye(*args, **kwargs):
    raise NotImplementedError("eye capture is not built yet")


def run_speech(*args, **kwargs):
    raise NotImplementedError("live speech is research only")


def run_clock(*args, **kwargs):
    raise NotImplementedError("waiting for the Clock models from Colab")
