import numpy as np
import pytest
from speech.features import (LONGEST_SPEAKER_LENGTH, get_audio_length, get_intervention_features,
                             get_pause_features, get_pauses_cnt, words_count)

TRAIN = "datasets/ADReSS-IS2020-train/ADReSS-IS2020-data/train"


def test_pause_counts_on_known_string():
    # one filler, one trailing-off, one (.) pause
    c = "*PAR:\twell &uh the (.) dog +... ran . 100_200"
    n = get_pauses_cnt(c)
    assert n[0] == 1 and n[3] == 1 and n[4] == 1 and len(n) == 9


def test_words_count_ignores_markers():
    assert words_count("*PAR:\tthe &uh dog ran . 0_10") == 3


@pytest.mark.parametrize("sid,grp", [("S001", "cc"), ("S079", "cd")])
def test_real_files(sid, grp):
    cha = f"{TRAIN}/transcription/{grp}/{sid}.cha"
    wav = f"{TRAIN}/Full_wave_enhanced_audio/{grp}/{sid}.wav"
    f = get_pause_features(cha, wav)
    assert f.shape == (11,) and np.all(f >= 0) and get_audio_length(wav) > 10
    s = get_intervention_features(cha)
    assert s.shape == (LONGEST_SPEAKER_LENGTH, 3) and np.all(s.sum(axis=1) == 1)
