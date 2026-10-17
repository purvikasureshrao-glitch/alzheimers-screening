"""Motor: Lima et al. 2024 protocol + age-adjusted healthy reference -> "outside reference" flag [OURS].

Decision 2026-10-18 (user): the Koppelmans data and protocol are unobtainable, so the Motor test follows Lima et al.:
one hand, tap as fast as possible on one spot for 30 s, dominant hand (non-dominant optional). No classifier is trained and
nothing here comes from an AD dataset, so the flag does not say "Alzheimer's": it says the tapping is outside the range of 176
healthy adults aged 18-74 at the person's age. Not validated; no accuracy.

Features (taps in seconds from the start of the trial, first 30 s):
  tps      taps per second
  cv       coefficient of variation of inter-tap intervals (std / mean)
  fatigue  taps in the last 10 s / taps in the first 10 s
Age adjustment: each feature is regressed on age (ordinary least squares, Lima dominant-hand trials); the person's age is clamped to
18-74 (no extrapolation beyond the reference). Cut-offs (conventions fixed in advance, not tuned): residual below the 5th percentile
for tps and fatigue, above the 95th for cv. Flag = at least 2 of the 3 outside.
Known limits: touchscreen vs the app's input device, only 15 reference adults aged 50+, target group older than the reference.
"""
import csv
import json
from collections import defaultdict

DURATION = 30.0
AGE_MIN, AGE_MAX = 18, 74
FEATURES = ("tps", "cv", "fatigue")


def trial_features(times_s):
    t = sorted(x for x in times_s if 0 <= x <= DURATION)
    iv = [b - a for a, b in zip(t, t[1:])]
    first, last = sum(x <= 10 for x in t), sum(20 < x <= 30 for x in t)
    if len(t) < 10 or not first:
        return None
    m = sum(iv) / len(iv)
    sd = (sum((d - m) ** 2 for d in iv) / len(iv)) ** 0.5
    return {"tps": len(t) / DURATION, "cv": sd / m, "fatigue": last / first}


def percentile(xs, q):
    xs = sorted(xs); k = (len(xs) - 1) * q / 100; lo = int(k); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def ols(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sum((a - mx) ** 2 for a in x)
    return my - b * mx, b


def lima_subjects(path, hand="Dominant"):
    """{subject: (age, features)} from the chosen hand's trial; tap times ms -> s."""
    taps, age = defaultdict(list), {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Hand"] == hand:
                taps[r["Code"]].append(float(r["time_of_tap"]) / 1000)
                age[r["Code"]] = int(r["Age during the test"])
    return {s: (age[s], trial_features(v)) for s, v in taps.items()}


def build_reference(path, hand="Dominant"):
    subj = {s: v for s, v in lima_subjects(path, hand).items() if v[1]}
    ages = [a for a, _ in subj.values()]
    ref = {"source": f"Lima et al. 2024 (Sci Data), {hand} hand, 30 s smartphone tapping", "n_subjects": len(subj),
           "domain_shift": "touchscreen vs app input device; ages 18-74 (median 30), few over 50; target group older",
           "age_clamp": [AGE_MIN, AGE_MAX]}
    for name in FEATURES:
        y = [f[name] for _, f in subj.values()]
        a0, b = ols(ages, y)
        res = [v - (a0 + b * a) for v, a in zip(y, ages)]
        ref[name] = {"intercept": a0, "slope_per_year": b, "res_p5": percentile(res, 5), "res_p95": percentile(res, 95)}
    return ref, subj


def outside(f, age, ref):
    age = min(max(age, AGE_MIN), AGE_MAX)
    r = {n: f[n] - (ref[n]["intercept"] + ref[n]["slope_per_year"] * age) for n in FEATURES}
    return {"tps": r["tps"] < ref["tps"]["res_p5"], "cv": r["cv"] > ref["cv"]["res_p95"],
            "fatigue": r["fatigue"] < ref["fatigue"]["res_p5"]}


def flag(f, age, ref):
    return sum(outside(f, age, ref).values()) >= 2


if __name__ == "__main__":
    import sys
    ref, subj = build_reference(sys.argv[1])
    json.dump(ref, open(sys.argv[2], "w"), indent=1)
    print(json.dumps({k: v for k, v in ref.items()}, indent=1))
    rows = list(subj.values())
    for name, lo in [("all", 0), ("age>=40", 40), ("age>=50", 50), ("age>=60", 60)]:
        sel = [(a, f) for a, f in rows if a >= lo]
        print(f"{name}: n={len(sel)} flagged={sum(flag(f, a, ref) for a, f in sel)}")
