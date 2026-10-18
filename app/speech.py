"""Speech (EXPERIMENTAL): microphone audio -> local Whisper `small` word timestamps -> 5 recoverable features -> speech_whisper.onnx.

Trained on ADReSS Cookie Theft recordings (human-transcript-free features, FINDINGS 9b); the 5-fold CV for this exact input is 0.639 (0.546-0.731),
n=108. Differences from that CV, all unvalidated [OURS]: faster-whisper (CTranslate2, int8) instead of openai-whisper; a laptop microphone;
the redrawn "Updated Cookie Theft" picture (Heyselaar 2021) shown on a laptop screen, not the original Boston Exam drawing the ADReSS speakers saw (not the original clinical setting); a fixed-length recording. The result is never counted in the
2-of-4 rule. `MIN_WORDS` is a data-quality gate, not a clinical threshold.
"""
import io
import json
from pathlib import Path

import numpy as np

from speech.whisper_features import WHISPER_FEATURE_NAMES, features_from_words

from .evidence import CAVEAT, EVIDENCE
from .schema import Result

MODELS = Path(__file__).resolve().parent.parent / "models"
PROMPT = "Um, uh, so, er, I think, I think the, the mm-hmm."      # same prompt as the validation run (notebooks/speech_whisper.ipynb)
MIN_WORDS = 20
_S = {}


def _onnx():
    if "ort" not in _S:
        import onnxruntime as ort
        _S["ort"] = ort.InferenceSession(str(MODELS / "speech_whisper.onnx"))
    return _S["ort"]


def _whisper():
    if "w" not in _S:
        from faster_whisper import WhisperModel
        _S["w"] = WhisperModel("small", device="cpu", compute_type="int8")
    return _S["w"]


def transcribe(audio_bytes):
    """Returns (words [(text, start_s, end_s)], audio_seconds)."""
    segs, info = _whisper().transcribe(io.BytesIO(audio_bytes), language="en", word_timestamps=True, initial_prompt=PROMPT,
                                       condition_on_previous_text=False)
    return [(w.word, float(w.start), float(w.end)) for s in segs for w in (s.words or [])], float(info.duration)


def score_words(words, seconds):
    """Pure part (testable without audio): words + duration -> Result."""
    content = [w for w in words if w[0].strip()]
    if len(content) < MIN_WORDS or seconds <= 0:
        return Result("speech", "not_run", "Speech test: too few words were captured",
                      measured={"words heard": len(content)}, evidence=EVIDENCE["speech"], caveat=CAVEAT["speech"], counts=False)
    feats = features_from_words(words, seconds)
    logits = _onnx().run(None, {"x": np.asarray([feats], np.float32)})[0][0]
    p = float(np.exp(logits[1]) / np.exp(logits).sum())
    flagged = p >= 0.5
    return Result(
        "speech", "flagged" if flagged else "not_flagged",
        "Experimental: speech pattern resembles the Alzheimer's group in the training data" if flagged
        else "Experimental: speech pattern does not resemble the Alzheimer's group in the training data",
        measured={"words heard": len(content), "seconds": round(seconds), **{k: round(v, 2) for k, v in zip(WHISPER_FEATURE_NAMES, feats)}},
        evidence=EVIDENCE["speech"], caveat=CAVEAT["speech"], counts=False)


def score(audio_bytes):
    return analyse(audio_bytes)[0]


def analyse(audio_bytes):
    """(Result, transcript text). The transcript is only shown back to the user in the page; it is never stored."""
    try:
        words, seconds = transcribe(audio_bytes)
    except Exception:
        return Result("speech", "not_run", "Recording could not be read", evidence=EVIDENCE["speech"], caveat=CAVEAT["speech"], counts=False), ""
    return score_words(words, seconds), "".join(w[0] for w in words).strip()
