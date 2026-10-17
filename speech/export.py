"""Final fit on all 108 subjects and ONNX export for the app (PLAN B1, live use = Disfluency + Acoustic).

This is NOT an evaluation: the 108 subjects are the training data. The only accuracy statement is
the 5-fold CV in `cv.py`. Checkpointing uses the same inner 80/20 split as in CV [OURS].
Interventions is research-only (needs an interviewer) and is not exported.
"""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .cv import fit_one, predict_proba


def export(disfluency, acoustic_raw, y, out_dir, seed=0, device="cpu"):
    from skl2onnx import to_onnx
    import onnxruntime as ort

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y)
    itr, iva = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    disfluency = np.asarray(disfluency, np.float32); acoustic_raw = np.asarray(acoustic_raw, np.float32)

    prep = make_pipeline(StandardScaler(), PCA(21, random_state=seed)).fit(acoustic_raw)
    ac = prep.transform(acoustic_raw).astype(np.float32)
    nets = {"disfluency": (fit_one("disfluency", disfluency[itr], y[itr], disfluency[iva], y[iva], seed, device), disfluency),
            "acoustic": (fit_one("acoustic", ac[itr], y[itr], ac[iva], y[iva], seed, device), ac)}

    onnx_prep = to_onnx(prep, acoustic_raw[:1], target_opset=15)
    (out / "speech_acoustic_prep.onnx").write_bytes(onnx_prep.SerializeToString())
    p = ort.InferenceSession(str(out / "speech_acoustic_prep.onnx"))
    prep_diff = float(np.abs(p.run(None, {p.get_inputs()[0].name: acoustic_raw})[0] - ac).max())
    print(f"acoustic prep (scaler+PCA) max |ONNX - sklearn| = {prep_diff:.2e}")

    for name, (net, X) in nets.items():
        net = net.cpu().eval()
        path = out / f"speech_{name}.onnx"
        torch.onnx.export(net, torch.zeros(1, X.shape[1]), str(path), input_names=["x"], output_names=["logits"],
                          dynamic_axes={"x": {0: "n"}}, opset_version=15, dynamo=False)
        s = ort.InferenceSession(str(path))
        logits = s.run(None, {"x": X})[0]
        p_onnx = torch.softmax(torch.tensor(logits), -1)[:, 1].numpy()
        p_torch = predict_proba(net, X)
        same = int(((p_onnx >= .5) == (p_torch >= .5)).sum())
        print(f"{name}: ONNX flag == PyTorch flag on {same} of {len(X)} | max |dP| = {np.abs(p_onnx - p_torch).max():.2e}")

    json.dump({"networks": ["disfluency", "acoustic"], "disfluency_input": "11 features from speech.features.get_pause_features",
               "acoustic_input": "6373 ComParE_2016 functionals -> speech_acoustic_prep.onnx (21 PCA) -> speech_acoustic.onnx",
               "output": "logits [not AD-like, AD-like]; softmax, P(AD-like) >= 0.5 flags",
               "status": "trained on all 108 ADReSS train subjects; accuracy only from 5-fold CV (FINDINGS 9); acoustic CI includes 0.5; "
                         "Whisper-transcript features and domain shift unvalidated"},
              open(out / "speech_features.json", "w"), indent=1)
