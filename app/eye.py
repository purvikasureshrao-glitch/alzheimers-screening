"""Eye (EXPERIMENTAL): webcam gaze samples -> eye.features.extract -> eye_flag.onnx -> Result (never counted in the 2-of-4 rule).

The model was trained on ~333 Hz research-tracker recordings (FINDINGS 1, 8b). A webcam cannot supply what it expects:
 - left/right eye gaze separately -> disconj_x/y are left missing (NaN), not faked as 0;
 - saccade timing at 333 Hz -> ~9 samples per 0.3 s task, so saccade features are mostly NaN (the model's imputer fills them);
 - the training tracker's pixel units are unpublished, so amplitude / velocity / spread are in browser-screen pixels [OURS, unmatched].
The output is therefore labelled experimental and excluded from the rule. How many of the 30 features were measured is shown.
"""
import json
from pathlib import Path

import numpy as np

from eye import features

from .evidence import CAVEAT, EVIDENCE
from .schema import Result

MODELS = Path(__file__).resolve().parent.parent / "models"
_S = {}


def _session():
    if not _S:
        import onnxruntime as ort
        _S["ort"] = ort.InferenceSession(str(MODELS / "eye_flag.onnx"))
        _S["order"] = json.loads((MODELS / "eye_features.json").read_text(encoding="utf-8"))["feature_order"]
    return _S


def feature_vector(tasks):
    """tasks: {task: {"t": ms list, "x": px list, "y": px list}} -> (30 floats in model order, per-task samples kept)."""
    row, kept = {}, {}
    for task in features.TASKS:
        d = tasks.get(task) or {}
        t, x, y = (np.asarray(d.get(k, []), float) for k in ("t", "x", "y"))
        n = min(len(t), len(x), len(y))
        t, x, y = t[:n], x[:n], y[:n]
        keep = [0] if n else []
        for i in range(1, n):                       # same rule as load_gaze: timestamps must strictly increase
            if t[i] > t[keep[-1]]:
                keep.append(i)
        t, x, y = t[keep], x[keep], y[keep]
        ok = np.isfinite(x) & np.isfinite(y)
        kept[task] = int(ok.sum())
        xy = np.column_stack([x, y, np.full(len(t), np.nan), np.full(len(t), np.nan)])   # right eye absent
        f = features.extract(t - t[0] if len(t) else t, xy)
        row.update({f"{task}.{k}": f[k] for k in features.FEATURES})
    return [row[c] for c in _session()["order"]], kept


def score(tasks, calibration_pct=None):
    try:
        vec, kept = feature_vector(tasks)
    except Exception:
        return Result("eye", "not_run", "Eye test could not be read", evidence=EVIDENCE["eye"], caveat=CAVEAT["eye"], counts=False)
    measured_n = int(np.isfinite(vec).sum())
    if measured_n == 0 or min(kept.values()) < 4:
        return Result("eye", "not_run", "Eye test: too little gaze was captured", measured={"gaze samples per task": kept},
                      evidence=EVIDENCE["eye"], caveat=CAVEAT["eye"], counts=False)
    label = _session()["ort"].run(["label"], {"X": np.asarray([vec], np.float32)})[0][0]
    flagged = int(label) == 1
    return Result(
        "eye", "flagged" if flagged else "not_flagged",
        "Experimental: gaze pattern resembles the Alzheimer's group in the pilot data" if flagged
        else "Experimental: gaze pattern does not resemble the Alzheimer's group in the pilot data",
        measured={"gaze samples per task": kept, "features measured (of 30)": measured_n,
                  **({"calibration error (% of screen width)": round(calibration_pct, 1)} if calibration_pct is not None else {})},
        evidence=EVIDENCE["eye"], caveat=CAVEAT["eye"], counts=False)
