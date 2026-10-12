import numpy as np
import pandas as pd

from clock.labels import make_table, round_of, spid_of
from clock.preprocess import crop_to_mass, invert, pad_square, remove_name_boxes
from clock.split import split_by_spid


def test_stems_and_rounds():
    assert spid_of("10000123") == 10000123
    assert spid_of("10000123_V2") == 10000123
    assert round_of("NHATS_R7_clock.zip") == 7
    assert round_of("Round_2_Clock.zip") == 2


def _images(spids, rnd=1):
    return pd.DataFrame({"spid": spids, "round": rnd, "zip_name": "z", "member_path": "m"})


def test_label_rules():
    sp = pd.DataFrame({"spid": [1, 2, 3, 4], "round": 1,
                       "clock_raw": [5, 0, -4, 3], "dementia_raw": [2, 1, 7, -8]})
    t = make_table(_images([1, 2, 3, 4, 5]), sp).set_index("spid")
    assert t.loc[1, "clock_score"] == 5 and t.loc[1, "dementia"] == 0
    assert t.loc[2, "dementia"] == 1 and t.loc[2, "clock_score"] == 0
    assert t.loc[3, "exclude_reason"] == "clock_code_not_a_score"
    assert t.loc[3, "dementia"] == 1            # code 7 counts as dementia
    assert t.loc[4, "dementia"] == 0            # negative code -> 0 (paper)
    assert t.loc[5, "exclude_reason"] == "no_survey_match"


def test_split_is_grouped_and_deterministic():
    rng = np.random.default_rng(0)
    spids = np.repeat(np.arange(2000), rng.integers(1, 4, 2000))
    dem = (spids % 13 == 0).astype(int)
    df = pd.DataFrame({"spid": spids, "dementia": dem, "exclude_reason": ""})
    a = split_by_spid(df, val_images=500, seed=1)
    b = split_by_spid(df, val_images=500, seed=1)
    assert (a.split == b.split).all()
    assert (a.groupby("spid").split.nunique() == 1).all()
    assert set(a.split) == {"train", "val", "test"}
    assert 400 < (a.split == "val").sum() < 650


def test_name_box_removed_strokes_kept():
    img = np.zeros((300, 300), np.float32)
    img[10:60, 10:200] = 1.0                      # solid box
    img[150, 20:280] = 1.0                        # thin line
    img[100:250, 150] = 1.0                       # thin line
    out = remove_name_boxes(img)
    assert out[10:60, 10:200].sum() == 0
    assert out[150, 20:280].sum() > 0 and out[100:250, 150].sum() > 0


def test_crop_and_pad():
    img = np.zeros((400, 600), np.float32)
    img[100:140, 200:300] = 1.0
    c = crop_to_mass(img, mass=0.96, margin=10)
    assert c.shape[0] < 400 and c.shape[1] < 600
    assert c.sum() >= 0.96 * img.sum()
    assert pad_square(c, size=64).shape == (64, 64)
    assert invert(np.ones((2, 2), np.float32)).sum() == 0
