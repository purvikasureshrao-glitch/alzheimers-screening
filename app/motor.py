"""Motor test scoring: tap timestamps (s) + age -> result, against the Lima age-adjusted healthy reference.

Flag rule changed by the user on 2026-10-19 (deviation from motor/reference.flag, which needs 2 of 3 features outside):
  - tapping faster than the healthy range (taps/s residual above the reference 95th percentile) is never flagged;
  - slow tapping (taps/s residual below the reference 5th percentile) is flagged on its own;
  - otherwise flagged when at least 2 of the 3 features are outside, as before.
The cut-offs (5th / 95th percentile of the 176 healthy adults) are unchanged; only how they combine changed."""
import json
from pathlib import Path

from motor import reference

from .evidence import CAVEAT, EVIDENCE
from .schema import Result

REF_PATH = Path(__file__).resolve().parent.parent / "models" / "motor_reference.json"


def _tps_residual(f, age, ref):
    age = min(max(age, reference.AGE_MIN), reference.AGE_MAX)
    return f["tps"] - (ref["tps"]["intercept"] + ref["tps"]["slope_per_year"] * age)


def score(tap_times_s, age, ref=None):
    ref = ref or json.load(open(REF_PATH))
    f = reference.trial_features(tap_times_s)
    if f is None:
        return Result("motor", "not_run", "Test not completed (too few taps)", evidence=EVIDENCE["motor"], caveat=CAVEAT["motor"])
    out = reference.outside(f, age, ref)
    fast = _tps_residual(f, age, ref) > ref["tps"]["res_p95"]
    flagged = not fast and (out["tps"] or sum(out.values()) >= 2)
    label = ("Tapping faster than the healthy range for this age" if fast else
             "Tapping slower than the healthy range for this age" if flagged and out["tps"] else
             "Tapping pattern outside the healthy range for this age" if flagged else "Tapping within the healthy range for this age")
    return Result(
        "motor", "flagged" if flagged else "not_flagged", label,
        measured={"taps per second": round(f["tps"], 2), "interval variability (CV)": round(f["cv"], 3),
                  "last 10 s / first 10 s taps": round(f["fatigue"], 2),
                  "outside the range": [k for k, v in out.items() if v]},
        evidence=EVIDENCE["motor"], caveat=CAVEAT["motor"])
