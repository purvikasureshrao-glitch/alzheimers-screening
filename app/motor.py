"""Motor test scoring: tap timestamps (s) + age -> result, against the Lima age-adjusted healthy reference."""
import json
from pathlib import Path

from motor import reference

from .evidence import CAVEAT, EVIDENCE
from .schema import Result

REF_PATH = Path(__file__).resolve().parent.parent / "models" / "motor_reference.json"


def score(tap_times_s, age, ref=None):
    ref = ref or json.load(open(REF_PATH))
    f = reference.trial_features(tap_times_s)
    if f is None:
        return Result("motor", "not_run", "Test not completed (too few taps)", evidence=EVIDENCE["motor"], caveat=CAVEAT["motor"])
    out = reference.outside(f, age, ref)
    flagged = reference.flag(f, age, ref)
    return Result(
        "motor", "flagged" if flagged else "not_flagged",
        "Tapping outside the healthy range for this age" if flagged else "Tapping within the healthy range for this age",
        measured={"taps per second": round(f["tps"], 2), "interval variability (CV)": round(f["cv"], 3),
                  "last 10 s / first 10 s taps": round(f["fatigue"], 2),
                  "outside the range": [k for k, v in out.items() if v]},
        evidence=EVIDENCE["motor"], caveat=CAVEAT["motor"])
