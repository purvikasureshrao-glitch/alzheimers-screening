"""Healthy reference range from Lima et al. 2024 (smartphone tapping) -> "outside reference" flag [OURS].

NOT the Koppelmans method and NOT validated. No classifier is trained and nothing here comes from an AD
dataset, so the flag does not say "Alzheimer's"; it says the single-hand tapping is outside a healthy
reference range. Known domain shift (FINDINGS 2, 2026-10-17): Lima is touchscreen, 30 s, 176 adults aged
18-74 (median 30, only 15 aged 50+); our task is keyboard shift keys, 10 s, target age ~75. Only features
computable in both are used. The cut-offs below are conventions fixed in advance, not tuned:
  tps      (taps/s over the first 10 s)           low  if < 5th percentile
  tps_var  (variance of ten 1-s counts)           high if > 95th percentile
  irt      (time to first tap, s)                 high if > 95th percentile
  flag     = at least 2 of the 3 outside their range
Dominant-hand trials only (the keyboard task's right index finger is assumed dominant).
"""
import csv
import json
from collections import defaultdict

DURATION, MIN_GAP = 10.0, 1.0
FEATURES = ("tps", "tps_var", "irt")


def trial_features(times_s):
    """times_s: tap times in seconds from the start of the trial."""
    t = sorted(x for x in times_s if 0 <= x <= DURATION)
    if not t:
        return None
    counts = [0] * 10
    for x in t:
        counts[min(int(x), 9)] += 1
    m = sum(counts) / 10
    edges = [0.0] + t + [DURATION]
    return {"tps": len(t) / DURATION, "tps_var": sum((c - m) ** 2 for c in counts) / 10, "irt": t[0],
            "gap": any(b - a >= MIN_GAP for a, b in zip(edges, edges[1:]))}


def percentile(xs, q):
    xs = sorted(xs); k = (len(xs) - 1) * q / 100; lo = int(k); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def lima_subjects(path):
    """Return {subject: (age, features)} from the dominant-hand trial, tap times converted ms -> s."""
    taps, age = defaultdict(list), {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Hand"] == "Dominant":
                taps[r["Code"]].append(float(r["time_of_tap"]) / 1000)
                age[r["Code"]] = int(r["Age during the test"])
    return {s: (age[s], trial_features(v)) for s, v in taps.items()}


def build_reference(path):
    subj = lima_subjects(path)
    feats = [f for _, f in subj.values() if f]
    ref = {"source": "Lima et al. 2024 (Sci Data), dominant-hand trials, first 10 s", "n_subjects": len(feats),
           "domain_shift": "touchscreen 30 s vs keyboard 10 s; ages 18-74 (median 30) vs target ~75",
           "tps_low": percentile([f["tps"] for f in feats], 5),
           "tps_var_high": percentile([f["tps_var"] for f in feats], 95),
           "irt_high": percentile([f["irt"] for f in feats], 95)}
    return ref, subj


def outside(f, ref):
    """Which of the 3 features are outside the healthy range."""
    return {"tps": f["tps"] < ref["tps_low"], "tps_var": f["tps_var"] > ref["tps_var_high"],
            "irt": f["irt"] > ref["irt_high"]}


def flag(f, ref):
    return sum(outside(f, ref).values()) >= 2


if __name__ == "__main__":
    import sys
    ref, subj = build_reference(sys.argv[1])
    json.dump(ref, open(sys.argv[2], "w"), indent=1)
    print(json.dumps(ref, indent=1))
    rows = [(a, f) for a, f in subj.values() if f]
    for name, sel in [("all", rows), ("age>=50", [r for r in rows if r[0] >= 50]), ("age>=60", [r for r in rows if r[0] >= 60])]:
        print(f"{name}: n={len(sel)} flagged={sum(flag(f, ref) for _, f in sel)} "
              f"median tps={sorted(f['tps'] for _, f in sel)[len(sel)//2]:.2f}")
