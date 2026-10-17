"""Disfluency features from a Whisper transcript with word timestamps (live-app input) [OURS].

The published 11 features (features.py) are counted from human CHAT transcripts. A Whisper
transcript has no CHAT markers and no speaker labels, so only the features that can be
recovered from words + timestamps are kept. Everything here is [OURS], not the paper's method.

  fillers      count of filler words (um, uh, er, ...)            <- CHAT '&uh'-type fillers
  mhm_hm       count of mm-hmm / hmm tokens                        <- CHAT 'mhm/hm'
  repetition   immediately repeated word ("the the")               <- CHAT '[/]' repetition (approx.)
  pause        gap between consecutive words >= PAUSE_GAP_S        <- CHAT '(.)' pause (approx.)
  word_rate    words / (50 * audio_seconds/10), as in features.py

NOT recoverable (dropped, not set to 0 in the model): intervention rate (no diarisation), retracing,
replacement, incomplete words, prolongation, trailing off.
PAUSE_GAP_S is fixed here BEFORE any result was seen and is not tuned.
"""
import re

PAUSE_GAP_S = 0.5
WHISPER_FEATURE_NAMES = ["fillers", "mhm_hm", "repetition", "pause", "word_rate"]
HUMAN_COLUMN = {"fillers": 1, "mhm_hm": 6, "repetition": 3, "pause": 4, "word_rate": 10}  # index in the 11 human features

FILLERS = {"um", "uh", "er", "erm", "ah", "uhm", "umm", "eh", "uhh"}
MHM = {"mhm", "mm-hmm", "mmhmm", "hmm", "hm", "mm", "mmm", "uh-huh"}


def _norm(w):
    return re.sub(r"[^a-z\-]", "", w.lower())


def features_from_words(words, audio_seconds, audio_length_normalization=10):
    """words: list of (text, start_s, end_s). Returns [fillers, mhm_hm, repetition, pause, word_rate]."""
    scale = audio_seconds / audio_length_normalization
    toks = [(_norm(t), s, e) for t, s, e in words]
    toks = [x for x in toks if x[0]]
    fillers = sum(t in FILLERS for t, _, _ in toks)
    mhm = sum(t in MHM for t, _, _ in toks)
    content = [x for x in toks if x[0] not in FILLERS and x[0] not in MHM]
    rep = sum(a[0] == b[0] for a, b in zip(content, content[1:]))
    pauses = sum((b[1] - a[2]) >= PAUSE_GAP_S for a, b in zip(toks, toks[1:]))
    word_rate = len(content) / (50 * scale)
    return [fillers / scale, mhm / scale, rep / scale, pauses / scale, word_rate]
