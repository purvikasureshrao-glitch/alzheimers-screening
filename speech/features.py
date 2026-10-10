"""Disfluency + intervention features, ported from the official Sarawgi et al. code
(github.com/wazeerzulfikar/alzheimers-dementia, dataset.py / utils.py, MIT licence;
read 2026-10-09). Regexes and cleaning are copied as published, NOT improved.

Known quirks of the original, reproduced on purpose (FINDINGS section 5):
  * the regexes run over the whole .cha text, including @header and %mor/%gra lines;
  * a line counts as INV/PAR speaker if it merely *contains* "INV"/"PAR", so header
    lines such as "@Participants: PAR ..., INV ..." are counted too.
"""
import contextlib
import re
import wave

import numpy as np

LONGEST_SPEAKER_LENGTH = 32  # config.py 'longest_speaker_length' (dataset.py default is 40)
AUDIO_LENGTH_NORMALIZATION = 10

PAUSE_FEATURE_NAMES = [
    "intervention_rate", "fillers_&", "retracing_<>", "repetition_[/+]", "pause_(.)",
    "trailing_+...", "mhm_hm", "replacement_[:word]", "incomplete_wor(d)",
    "prolongation_w:ord", "word_rate",
]


def clean_file(lines):
    return re.sub(r'[0-9]+[_][0-9]+', '', lines.replace("*INV:", "").replace("*PAR:", "")).strip().replace("\x15", "").replace("\n", "").replace("\t", " ").replace("[+ ", "[+").replace("[* ", "[*").replace("[: ", "[:").replace(" .", "").replace("'s", "").replace(" ?", "").replace(" !", "").replace(" ]", "]").lower()


def extra_clean(lines):
    lines = clean_file(lines)
    lines = lines.replace("[+exc]", "")
    lines = lines.replace("[+gram]", "")
    lines = lines.replace("[+es]", "")
    lines = re.sub(r'[&][=]*[a-z]+', '', lines)  # remove all &=text
    lines = re.sub(r'\[[*][a-z]:[a-z][-|a-z]*\]', '', lines)  # remove all [*char:char(s)]
    lines = re.sub(r'[^A-Za-z0-9\s_]+', '', lines)  # remove all remaining symbols except underscore
    lines = re.sub(r'[_]', ' ', lines)  # replace underscore with space
    return lines


def words_count(content):
    extra_cleaned = extra_clean(content).split(" ")
    return len(extra_cleaned) - extra_cleaned.count('')


def get_pauses_cnt(content):
    content = clean_file(content)
    pauses_list = []

    pauses_list.append(len(re.findall(r'&[a-z]+', content)))
    pauses_list.append(len(re.findall(r'<[a-z_\s]+>', content)))
    pauses_list.append(len(re.findall(r'\[/+\]', content)))
    pauses_list.append(len(re.findall(r'\([\.]+\)', content)))
    pauses_list.append(len(re.findall(r'\+[\.]+', content)))
    pauses_list.append(len(re.findall(r'[m]*hm', content)))
    pauses_list.append(len(re.findall(r'\[:[a-z_\s]+\]', content)))
    pauses_list.append(len(re.findall(r'[a-z]*\([a-z]+\)[a-z]*', content)))

    temp = re.sub(r'\[[*][a-z]:[a-z][-|a-z]*\]', '', content)
    pauses_list.append(len(re.findall(r'[a-z]+:[a-z]+', temp)))

    return np.array(pauses_list)


def _speakers(content):
    speakers = []
    for c in content.split('\n'):
        if 'INV' in c:
            speakers.append('INV')
        if 'PAR' in c:
            speakers.append('PAR')
    return speakers


def _par_span(speakers):
    first = speakers.index('PAR')
    last = len(speakers) - speakers[::-1].index('PAR') - 1
    return speakers[first:last]


def get_n_interventions(content):
    return _par_span(_speakers(content)).count('INV')


def get_audio_length(filename):
    with contextlib.closing(wave.open(str(filename), 'r')) as f:
        return f.getnframes() / float(f.getframerate())


def get_pause_features(cha_path, wav_path, audio_length_normalization=AUDIO_LENGTH_NORMALIZATION):
    """11 disfluency features: [intervention rate, 9 pause-type rates, word rate]."""
    audio_len = get_audio_length(wav_path) / audio_length_normalization
    with open(cha_path, 'r', encoding='utf-8') as f:
        content = f.read()
    word_rate = words_count(content) / (50 * audio_len)
    pause_rates = get_pauses_cnt(content) / audio_len
    inv_rate = get_n_interventions(content) / audio_len
    return np.concatenate(([inv_rate], pause_rates, [word_rate]), axis=-1)


def get_intervention_features(cha_path, max_length=LONGEST_SPEAKER_LENGTH):
    """Speaker sequence (first -> last PAR line) one-hot as (max_length, 3), padded/truncated."""
    speaker_dict = {'INV': [0, 0, 1], 'PAR': [0, 1, 0], 'padding': [1, 0, 0]}
    with open(cha_path, 'r', encoding='utf-8') as f:
        seq = [speaker_dict[s] for s in _par_span(_speakers(f.read()))]
    if len(seq) > max_length:
        seq = seq[:max_length]
    else:
        seq = seq + [speaker_dict['padding']] * (max_length - len(seq))
    return np.array(seq, dtype=np.float32)
