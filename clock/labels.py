"""H1 label table: NHATS clock images joined to the Sample Person survey (PLAN H1).

Reads the clock zips and the SP `.dta` inside each Stata zip directly, no extraction.
PRIVACY: nothing here prints an SPID or a file name; `summarize` prints aggregate counts.
"""
import re
import zipfile
from pathlib import Path

import pandas as pd

CLOCK_COL = "cg{r}dclkdraw"      # interviewer clock score, 0-5
DEMENTIA_COL = "hc{r}disescn9"   # 1 yes, 2 no, 7 previously reported, negatives = special
DEMENTIA_YES = (1, 7)            # Bone et al. 2.4.1

COLUMNS = ["spid", "round", "zip_name", "member_path", "clock_score", "dementia",
           "dementia_raw", "exclude_reason"]


def round_of(name):
    """Round number from a name like NHATS_R7_... or Round_2_...; None if absent."""
    m = re.search(r"(?:_R|Round[_ ])(\d{1,2})", name, re.I)
    return int(m.group(1)) if m else None


def spid_of(stem):
    """SPID from an image stem: plain digits, or digits plus one `_V<n>` suffix."""
    m = re.fullmatch(r"(\d+)(?:_V\d+)?", stem, re.I)
    if m:
        return int(m.group(1))
    digits = re.findall(r"\d+", stem)
    return int(max(digits, key=len)) if digits else None


def index_images(image_dir):
    """One row per .tif inside the clock zips: spid, round, zip_name, member_path."""
    rows = []
    for zp in sorted(Path(image_dir).glob("*.zip")):
        rnd = round_of(zp.name)
        with zipfile.ZipFile(zp) as z:
            for member in z.namelist():
                if member.lower().endswith((".tif", ".tiff")):
                    rows.append((spid_of(Path(member).stem), rnd, zp.name, member))
    return pd.DataFrame(rows, columns=["spid", "round", "zip_name", "member_path"])


def read_sp_file(zip_path, rnd):
    """Sample Person file of one round -> DataFrame[spid, clock_raw, dementia_raw, round]."""
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if re.search(r"_SP_File.*\.dta$", n, re.I)]
        if not names:
            raise FileNotFoundError(f"no SP .dta in {Path(zip_path).name}")
        with z.open(names[0]) as fh:
            df = pd.read_stata(fh, convert_categoricals=False)
    cols = {c.lower(): c for c in df.columns}
    need = [CLOCK_COL.format(r=rnd), DEMENTIA_COL.format(r=rnd)]
    missing = [c for c in need if c not in cols]
    if missing:
        raise KeyError(f"round {rnd}: columns not found: {missing}")
    out = pd.DataFrame({
        "spid": df[cols["spid"]].astype("int64"),
        "clock_raw": df[cols[need[0]]],
        "dementia_raw": df[cols[need[1]]],
    })
    out["round"] = rnd
    return out.drop_duplicates(["spid", "round"])


def read_all_sp(survey_dir):
    parts = []
    for zp in sorted(Path(survey_dir).rglob("*STATA*.zip")):
        rnd = round_of(zp.name)
        if rnd is not None:
            parts.append(read_sp_file(zp, rnd))
    if not parts:
        raise FileNotFoundError(f"no Stata release zips under {survey_dir}")
    return pd.concat(parts, ignore_index=True)


def make_table(images, sp):
    """Join images to survey rows and apply the label rules (pure function, testable)."""
    df = images.merge(sp, on=["spid", "round"], how="left")
    matched = df["clock_raw"].notna() | df["dementia_raw"].notna()
    clock = pd.to_numeric(df["clock_raw"], errors="coerce")
    valid = clock.between(0, 5) & (clock == clock.round())
    df["clock_score"] = clock.where(valid)
    # paper's rule: 1 or 7 -> 1, everything else (negatives, missing) -> 0
    df["dementia"] = df["dementia_raw"].isin(DEMENTIA_YES).astype(int)
    reason = pd.Series("", index=df.index)
    reason[~valid] = "clock_code_not_a_score"      # negative special code [OURS]
    reason[~matched] = "no_survey_match"
    df["exclude_reason"] = reason
    return df[COLUMNS]


def build_label_table(image_dir, survey_dir):
    return make_table(index_images(image_dir), read_all_sp(survey_dir))


def summarize(df):
    """Aggregate counts only."""
    kept = df[df.exclude_reason == ""]
    print(f"images indexed: {len(df)}")
    for reason, n in df.exclude_reason.replace("", "KEPT").value_counts().items():
        print(f"  {reason:<26} {n}")
    print(f"kept images: {len(kept)}   unique SPIDs: {kept.spid.nunique()}")
    print("clock score counts (kept):",
          kept.clock_score.astype(int).value_counts().sort_index().to_dict())
    print(f"dementia = 1: {int(kept.dementia.sum())} of {len(kept)} "
          f"({100 * kept.dementia.mean():.1f}%)")
    print("by round (kept):")
    print(kept.groupby("round").agg(n=("spid", "size"), dem=("dementia", "sum")).to_string())
