"""Extended speech features from ASR words + timing, and patient-only filtering [OURS, 2026-10-20].

Everything here can be computed live from the patient's own words, so training and the app use the same code.
Constants are fixed BEFORE any result was seen and are not tuned (see notebooks/speech_improve.ipynb).

  F5   the five features of whisper_features (fillers, mhm_hm, repetition, pause, word_rate)
  FEXT F5 + long pauses (gap >= 1.0 s), mean gap, root type-token ratio, mean word length, distinct picture-content
       words mentioned (info_units), content-word count
"""
import math
import re

from .whisper_features import FILLERS, MHM, WHISPER_FEATURE_NAMES, _norm, features_from_words

LONG_PAUSE_S = 1.0
# Cookie Theft picture content (classic "information units"); one point per distinct unit mentioned. Fixed in advance, from the picture, not from data.
INFO_UNITS = {
    "mother": {"mother", "woman", "lady", "mom", "mum", "wife"}, "boy": {"boy", "son", "brother"}, "girl": {"girl", "daughter", "sister"},
    "sink": {"sink", "basin"}, "water": {"water"}, "overflow": {"overflow", "overflowing", "overflows", "overflowed", "spilling", "running", "flooding", "flood"},
    "floor": {"floor"}, "cookie": {"cookie", "cookies", "biscuit", "biscuits"}, "jar": {"jar", "container", "cookiejar"},
    "stool": {"stool", "chair", "ladder"}, "fall": {"fall", "falling", "fell", "tipping", "tip", "tipped"}, "window": {"window", "windows"},
    "curtain": {"curtain", "curtains", "drapes"}, "dishes": {"dish", "dishes", "plate", "plates", "cup", "cups", "dishcloth"}, "cupboard": {"cupboard", "cupboards", "cabinet", "cabinets"},
    "kitchen": {"kitchen"}, "tap": {"tap", "taps", "faucet", "faucets"}, "outside": {"garden", "yard", "outside", "tree", "trees", "grass", "lawn", "bushes", "walk"},
}
FEXT_NAMES = WHISPER_FEATURE_NAMES + ["long_pause", "mean_gap", "root_ttr", "mean_word_len", "info_units", "content_words"]


def features_ext(words, seconds):
    """words: [(text, start_s, end_s)]; seconds: speaking/recording duration. Returns the FEXT vector (11 floats)."""
    base = features_from_words(words, seconds)
    scale = seconds / 10
    toks = [(_norm(t), s, e) for t, s, e in words]
    toks = [x for x in toks if x[0]]
    gaps = [max(0.0, b[1] - a[2]) for a, b in zip(toks, toks[1:])]
    content = [t for t, _, _ in toks if t not in FILLERS and t not in MHM]
    n = len(content)
    long_pause = sum(g >= LONG_PAUSE_S for g in gaps) / scale
    mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
    rttr = len(set(content)) / math.sqrt(n) if n else 0.0
    wlen = sum(map(len, content)) / n if n else 0.0
    said = set(content)
    info = sum(bool(said & syn) for syn in INFO_UNITS.values())
    return base + [long_pause, mean_gap, rttr, wlen, float(info), float(n)]


def transcript_text(words):
    return " ".join(t.strip() for t, _, _ in words if t.strip())


# ---- patient-only filtering (training only: uses the human CHAT time marks to find the patient's turns) ----
_BULLET = re.compile(r"\x15?(\d+)_(\d+)\x15?")


def par_spans(cha_text):
    """[(start_s, end_s)] of the participant's (*PAR) utterances, from the CHAT time marks 'start_end' in ms."""
    spans, cur = [], None
    for line in cha_text.splitlines():
        if line.startswith("*"):
            cur = line.startswith("*PAR:")
        elif line.startswith("%") or line.startswith("@"):
            cur = None
        if cur is None or not cur:
            continue
        m = _BULLET.findall(line)
        if m and (line.startswith("*PAR:") or line.startswith(("\t", " "))):
            a, b = m[-1]
            if int(b) > int(a):
                spans.append((int(a) / 1000, int(b) / 1000))
                cur = None   # one time mark ends the utterance
    return spans


def keep_in_spans(words, spans, tol=0.15):
    """Words whose midpoint lies inside any span (± tol seconds)."""
    return [w for w in words if any(a - tol <= (w[1] + w[2]) / 2 <= b + tol for a, b in spans)]


def spans_seconds(spans):
    return sum(b - a for a, b in spans)
