"""H2 SPID-grouped split [OURS]: a person is in exactly one of train / val / test."""
import numpy as np


def _stratified_take(per, frac, rng):
    """Pick `frac` of the rows of `per`, separately within each `ever` group."""
    picked = np.zeros(len(per), bool)
    for g in (0, 1):
        idx = np.flatnonzero(per.ever.to_numpy() == g)
        picked[rng.choice(idx, size=int(round(frac * len(idx))), replace=False)] = True
    return per[~picked], per[picked]


def split_by_spid(df, val_images=1000, test_frac=0.10, seed=0):
    """Add a `split` column: train / val / test for kept rows, 'excluded' otherwise.

    Test = `test_frac` of SPIDs; validation = enough of the remaining SPIDs to hold about
    `val_images` images. Both stratified on "ever dementia = 1" for that SPID.
    """
    df = df.copy()
    kept = df.exclude_reason == ""
    per = (df[kept].groupby("spid")
           .agg(n=("spid", "size"), ever=("dementia", "max")).reset_index())
    rng = np.random.default_rng(seed)
    rest, test = _stratified_take(per, test_frac, rng)
    val_frac = min(0.5, val_images / rest.n.sum())
    train, val = _stratified_take(rest, val_frac, rng)
    which = {s: "train" for s in train.spid}
    which.update({s: "val" for s in val.spid})
    which.update({s: "test" for s in test.spid})
    df["split"] = np.where(kept, df.spid.map(which), "excluded")
    return df


def report(df):
    """Aggregate counts per split and a check that no SPID sits in two splits."""
    kept = df[df.split != "excluded"]
    g = kept.groupby("split").agg(images=("spid", "size"), spids=("spid", "nunique"),
                                  dementia_images=("dementia", "sum"))
    g["dementia_pct"] = (100 * g.dementia_images / g.images).round(1)
    print(g.to_string())
    sets = [set(kept.spid[kept.split == s]) for s in ("train", "val", "test")]
    overlap = sum(len(sets[i] & sets[j]) for i in range(3) for j in range(i + 1, 3))
    print("SPIDs in more than one split:", overlap)
