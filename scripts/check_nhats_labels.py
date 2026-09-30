"""
Check that the NHATS Sample Person survey files contain the two label columns
used by Bone et al. (2025), and that they join to the clock images we have.

PRIVACY: this script runs entirely on your machine and prints ONLY aggregate
counts. No SPID, filename, or participant row is ever displayed. NHATS
Conditions of Use permit aggregate statistical summaries.

Run:  py -3.11 scripts\check_nhats_labels.py
"""

import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is missing. Run:  py -3.11 -m pip install pandas pyreadstat")

IMAGES_DIR = Path(r"D:\Alzheimers\datasets\NHATS")
SURVEY_DIR = Path(r"D:\Alzheimers\datasets\NHATS\survey")

CLOCK_COL = "cg{r}dclkdraw"      # interviewer clock score, 0-5
DEMENTIA_COL = "hc{r}disescn9"   # reported dementia/AD diagnosis


def round_of(name):
    """Pull the round number out of a file name like NHATS_R7_... or Round_2_..."""
    m = re.search(r"(?:_R|Round[_ ])(\d{1,2})", name, re.I)
    return int(m.group(1)) if m else None


def spids_from_zip(zip_path):
    """Return the set of SPIDs referenced by the .tif files inside one zip."""
    spids = set()
    shapes = Counter()
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            if not member.lower().endswith((".tif", ".tiff")):
                continue
            stem = Path(member).stem
            shapes[re.sub(r"\d", "N", stem)] += 1
            digits = re.findall(r"\d+", stem)
            if digits:
                spids.add(max(digits, key=len))   # longest digit run = SPID
    return spids, shapes


def main():
    print("=" * 68)
    print("NHATS LABEL CHECK")
    print("=" * 68)

    # ---------- 1. images ----------
    zips = sorted(IMAGES_DIR.glob("*.zip"))
    if not zips:
        sys.exit(f"No zips found in {IMAGES_DIR}")

    images_by_round = {}
    total_images = 0
    all_shapes = Counter()

    print("\n[1] CLOCK IMAGES")
    for zp in zips:
        rnd = round_of(zp.name)
        spids, shapes = spids_from_zip(zp)
        all_shapes.update(shapes)
        n = sum(shapes.values())
        total_images += n
        if rnd is not None:
            images_by_round[rnd] = spids
        print(f"    round {str(rnd):>2}  images {n:>6}  unique SPIDs {len(spids):>6}")

    print(f"\n    TOTAL IMAGES: {total_images}   (Bone et al. reports 54,027)")
    print(f"    filename shapes seen (digits masked as N):")
    for shape, cnt in all_shapes.most_common(5):
        print(f"        {shape}  x{cnt}")

    # ---------- 2. survey files ----------
    print("\n[2] SURVEY FILES")
    if not SURVEY_DIR.exists():
        sys.exit(f"\n    MISSING: {SURVEY_DIR}\n    Put the SP .dta files there and re-run.")

    survey = sorted(list(SURVEY_DIR.glob("*.dta")) + list(SURVEY_DIR.glob("*.sas7bdat")))
    if not survey:
        sys.exit(f"    No .dta or .sas7bdat files in {SURVEY_DIR}")

    found_any = False
    for sp in survey:
        rnd = round_of(sp.name)
        try:
            if sp.suffix == ".dta":
                df = pd.read_stata(sp, convert_categoricals=False)
            else:
                df = pd.read_sas(sp)
        except Exception as e:
            print(f"    round {str(rnd):>2}  COULD NOT READ: {type(e).__name__}")
            continue

        cols = {c.lower(): c for c in df.columns}
        clock = cols.get(CLOCK_COL.format(r=rnd))
        dem = cols.get(DEMENTIA_COL.format(r=rnd))

        print(f"\n    round {str(rnd):>2}  rows {len(df):>6}  columns {len(df.columns):>4}")
        print(f"        {CLOCK_COL.format(r=rnd):<18} {'FOUND' if clock else 'not found'}")
        print(f"        {DEMENTIA_COL.format(r=rnd):<18} {'FOUND' if dem else 'not found'}")

        if clock:
            found_any = True
            vc = df[clock].value_counts(dropna=False).sort_index()
            print(f"        clock score distribution:")
            for val, cnt in vc.items():
                print(f"            {str(val):>8} : {cnt}")

        if dem:
            vc = df[dem].value_counts(dropna=False).sort_index()
            print(f"        dementia flag distribution:")
            for val, cnt in vc.items():
                print(f"            {str(val):>8} : {cnt}")

        # ---------- 3. join ----------
        idcol = cols.get("spid")
        if idcol and rnd in images_by_round:
            survey_ids = set(df[idcol].dropna().astype("int64").astype(str))
            img_ids = images_by_round[rnd]
            matched = len(img_ids & survey_ids)
            pct = 100 * matched / len(img_ids) if img_ids else 0
            print(f"        JOIN: {matched} of {len(img_ids)} image SPIDs "
                  f"matched a survey row  ({pct:.1f}%)")
        elif not idcol:
            print(f"        JOIN: no 'spid' column; columns are named differently")

    print("\n" + "=" * 68)
    if found_any:
        print("RESULT: label columns present. Hand annotation is NOT needed.")
    else:
        print("RESULT: label columns NOT found in these files.")
        print("        Check you downloaded the SP (Sample Person) interview file,")
        print("        not the Tracker or Other Person file.")
    print("=" * 68)


if __name__ == "__main__":
    main()
