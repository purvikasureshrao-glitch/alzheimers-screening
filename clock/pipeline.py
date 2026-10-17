"""Clock run-all pipeline (PLAN H3-H7): prep -> Crop-Net -> multi-step crop -> ViT (2 phases) -> test metrics -> ONNX.

Every stage is skipped if its `.done` marker exists in cfg['work'] (so a rerun in the same Colab session continues), and the
trainers resume from their checkpoints in cfg['out']. Aggregate numbers are printed; per-image predictions stay in cfg['out']
and are never read by Claude. Deviations from the paper are marked [OURS] in the modules and listed in FINDINGS 3e.
"""
import json
import os
import time
import zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from . import crop as cropmod
from . import preprocess
from .cropnet.train import train as train_cropnet
from .vit import train as vt
from .vit.model import ClockViT

SMOKE = dict(work="/content/work_smoke", out="/content/out_smoke", limit=dict(train=24, val=10, test=10), procs=2,
             bank=24, cropnet_iters=4, cropnet_eval=2, cropnet_patience=5, cropnet_budget=None,
             p1_epochs=1, p1_samples=64, p2_epochs=1, p2_samples=40, vit_patience=2, vit_budget=None)
FULL = dict(work="/content/work", out="/content/out", limit=None, procs=2, bank=2000,
            cropnet_iters=6000, cropnet_eval=100, cropnet_patience=15, cropnet_budget=3600 * 1.5,
            p1_epochs=10, p1_samples=12000, p2_epochs=12, p2_samples=12000, vit_patience=4, vit_budget=3600 * 2)

_NH, _Z = None, {}


def _init(nhats):
    global _NH
    _NH = nhats


def _work(task):
    zn, mp = task
    try:
        if zn not in _Z:
            _Z[zn] = zipfile.ZipFile(f"{_NH}/{zn}")
        return preprocess.rough_crop_512(_Z[zn].read(mp)), True
    except Exception:
        return np.zeros((512, 512), np.uint8), False


def _marker(cfg, name):
    return os.path.join(cfg["work"], name + ".done")


def stage_prep(cfg, tab, nhats, log):
    w = cfg["work"]; os.makedirs(w, exist_ok=True); os.makedirs(cfg["out"], exist_ok=True)
    if os.path.exists(_marker(cfg, "prep")):
        log("prep: already done"); return pd.read_csv(f"{w}/index.csv")
    kept = tab[tab.split.isin(["train", "val", "test"])]
    if cfg["limit"]:
        kept = pd.concat([kept[kept.split == s].sample(min((kept.split == s).sum(), n), random_state=0)
                          for s, n in cfg["limit"].items()])
    kept = kept.reset_index(drop=True)
    N = len(kept)
    mm = np.lib.format.open_memmap(f"{w}/rough512.npy", mode="w+", dtype=np.uint8, shape=(N, 512, 512))
    ok = np.zeros(N, bool); t0 = time.time()
    with Pool(cfg["procs"], initializer=_init, initargs=(nhats,)) as p:
        for i, (arr, good) in enumerate(p.imap(_work, zip(kept.zip_name, kept.member_path), chunksize=8)):
            mm[i] = arr; ok[i] = good
            if i % 2000 == 0:
                log(f"prep {i}/{N} [{time.time() - t0:.0f}s]")
    mm.flush(); del mm
    kept["ok"] = ok
    log(f"prep done: {N} images, {int((~ok).sum())} failed to read, {time.time() - t0:.0f}s")
    if (~ok).mean() > 0.01:
        raise RuntimeError("more than 1% of images failed in preprocessing")
    kept.to_csv(f"{w}/index.csv", index=False)
    open(_marker(cfg, "prep"), "w").close()
    return kept


def stage_cropnet(cfg, idx, device, log):
    from .cropnet.model import CropNet
    path = f"{cfg['out']}/cropnet_best.pt"
    if os.path.exists(_marker(cfg, "cropnet")):
        m = CropNet().to(device); m.load_state_dict(torch.load(path, map_location=device)); log("cropnet: already done")
        return m.eval()
    mm = np.load(f"{cfg['work']}/rough512.npy", mmap_mode="r")
    rng = np.random.default_rng(0)

    def bank(split, n):
        rows = np.flatnonzero((idx.split == split).to_numpy() & idx.ok.to_numpy())
        rows = np.sort(rng.choice(rows, size=min(n, len(rows)), replace=False))
        return (torch.from_numpy(np.asarray(mm[rows])).to(device),
                torch.tensor(idx.clock_score.to_numpy()[rows], dtype=torch.float32, device=device))
    btr, str_ = bank("train", cfg["bank"]); bva, sva = bank("val", 1000)
    model, best = train_cropnet(btr, str_, bva, sva, cfg["out"], cfg["cropnet_iters"], device,
                                eval_every=cfg["cropnet_eval"], patience=cfg["cropnet_patience"],
                                time_budget_s=cfg["cropnet_budget"], log=log)
    log(f"crop-net best synthetic val MSE {best:.5f}")
    open(_marker(cfg, "cropnet"), "w").close()
    return model


def stage_crop(cfg, idx, model, device, log):
    w = cfg["work"]; N = len(idx)
    if os.path.exists(_marker(cfg, "crop")):
        log("crop: already done"); return np.load(f"{w}/final224.npy", mmap_mode="r")
    mm = np.load(f"{w}/rough512.npy", mmap_mode="r")
    fin = np.lib.format.open_memmap(f"{w}/final224.npy", mode="w+", dtype=np.uint8, shape=(N, 224, 224))
    ink, border = np.zeros(N), np.zeros(N, bool); t0 = time.time()
    for s in range(0, N, 256):
        out = cropmod.multi_step_crop(np.asarray(mm[s:s + 256]), model, 3, device)
        fin[s:s + len(out)] = np.stack([cropmod.to224(o) for o in out])
        ink[s:s + len(out)], border[s:s + len(out)] = cropmod.qc_stats(out)
        if (s // 256) % 10 == 0:
            log(f"crop {s + len(out)}/{N} [{time.time() - t0:.0f}s]")
    fin.flush()
    log(f"crop QC (descriptive, no images dropped): ink touches the border in {border.mean() * 100:.1f}% of crops; "
        f"ink fraction < 0.5% in {(ink < 0.005).mean() * 100:.1f}%; median ink fraction {np.median(ink) * 100:.1f}%")
    pd.DataFrame(dict(ink=ink, border=border)).to_csv(f"{cfg['out']}/crop_qc.csv", index=False)
    open(_marker(cfg, "crop"), "w").close()
    return np.load(f"{w}/final224.npy", mmap_mode="r")


def _split(idx, fin, name, device):
    rows = np.flatnonzero((idx.split == name).to_numpy() & idx.ok.to_numpy())
    return dict(rows=rows, imgs=torch.from_numpy(np.asarray(fin[rows])),
                sc=torch.tensor(idx.clock_score.to_numpy()[rows].astype(int)),
                dem=torch.tensor(idx.dementia.to_numpy()[rows].astype(int)),
                spid=idx.spid.to_numpy()[rows])


def stage_vit(cfg, idx, fin, device, log):
    tr, va, te = (_split(idx, fin, s, device) for s in ("train", "val", "test"))
    gpu = lambda d: dict(imgs=d["imgs"].to(device), sc=d["sc"].to(device), dem=d["dem"].to(device))
    model = ClockViT().to(device)
    log(f"ViT-B/16 224px; train {len(tr['sc'])}, val {len(va['sc'])}, test {len(te['sc'])} images")
    gtr, gva = gpu(tr), gpu(va)
    model, b1 = vt.run_phase(model, "head", gtr, gva, cfg["out"], device, 50, 1, cfg["p1_epochs"], cfg["p1_samples"],
                             cfg["vit_patience"], 1, False, cfg["vit_budget"], log)
    model, b2 = vt.run_phase(model, "full", gtr, gva, cfg["out"], device, 4, 10, cfg["p2_epochs"], cfg["p2_samples"],
                             cfg["vit_patience"], 1, True, cfg["vit_budget"], log)
    log(f"ViT best balanced-val MSE: head {b1:.5f}, full {b2:.5f}")
    return model, va, te


def evaluate(cfg, model, va, te, device, log):
    from common.metrics import auc_ci, balanced_acc_ci, f1_ci
    _, pv = vt.predict(model, va["imgs"], device)
    cands = np.linspace(0.0, 1.0, 201)
    from sklearn.metrics import balanced_accuracy_score
    thr = float(max(cands, key=lambda t: balanced_accuracy_score(va["dem"].numpy(), (pv >= t).astype(int))))
    sc, pt = vt.predict(model, te["imgs"], device)
    y, g = te["dem"].numpy(), te["spid"]
    proto = f"NHATS test split, SPID-grouped, threshold {thr:.3f} chosen on validation (n_val={len(pv)})"
    log(f"threshold (max balanced accuracy on natural validation set): {thr:.3f}")
    for r in (balanced_acc_ci(y, (pt >= thr).astype(int), proto, g), f1_ci(y, (pt >= thr).astype(int), proto, g),
              auc_ci(y, pt, proto, g)):
        log(str(r))
    log(f"clock-score MSE on test (0-5 scale): {float(np.mean((sc - te['sc'].numpy()) ** 2)):.3f}, n={len(sc)} images, "
        f"{len(np.unique(g))} SPIDs; mean abs error {float(np.mean(np.abs(sc - te['sc'].numpy()))):.3f}")
    log(f"test dementia prevalence {y.mean() * 100:.1f}%; flagged {float((pt >= thr).mean()) * 100:.1f}%")
    pd.DataFrame(dict(spid=g, clock_score=te["sc"].numpy(), dementia=y, pred_score=sc, p_dementia=pt)).to_csv(
        f"{cfg['out']}/clock_test_preds.csv", index=False)
    return thr, te


def export(cfg, model, cropnet, thr, te, device, log):
    import onnxruntime as ort
    out = cfg["out"]
    m = model.cpu().eval(); c = cropnet.cpu().eval()
    x = te["imgs"][:16].float().div(255).unsqueeze(1)
    torch.onnx.export(m, x[:1], f"{out}/clock_vit.onnx", input_names=["x"], output_names=["y"],
                      dynamic_axes={"x": {0: "n"}, "y": {0: "n"}}, opset_version=17, dynamo=False)
    s = ort.InferenceSession(f"{out}/clock_vit.onnx")
    with torch.no_grad():
        ref = m(x).numpy()
    got = s.run(None, {"x": x.numpy()})[0]
    log(f"ViT ONNX vs PyTorch: max |diff| {np.abs(ref - got).max():.2e}; same dementia flag on "
        f"{int(((ref[:, 1] >= thr) == (got[:, 1] >= thr)).sum())} of {len(ref)}")
    xc = torch.zeros(1, 1, 512, 512)
    torch.onnx.export(c, xc, f"{out}/clock_cropnet.onnx", input_names=["x"], output_names=["heat"],
                      opset_version=17, dynamo=False)
    sc = ort.InferenceSession(f"{out}/clock_cropnet.onnx")
    xr = torch.rand(1, 1, 512, 512)
    with torch.no_grad():
        d = np.abs(c(xr).numpy() - sc.run(None, {"x": xr.numpy()})[0]).max()
    log(f"Crop-Net ONNX vs PyTorch: max |diff| {d:.2e}")
    json.dump({"vit_input": "(n,1,224,224) float32 in [0,1], ink bright on black, after cleaning + Crop-Net crop (3 steps)",
               "vit_output": "[clock score / 5, P(dementia)]; multiply output 0 by 5", "dementia_threshold": thr,
               "cropnet_input": "(1,1,512,512)", "cropnet_output": "(1,2,64,64), channel 0 = clock heat map",
               "status": "NHATS only (paper-like clean scans); photo/webcam domain shift unvalidated; deviations in FINDINGS 3e"},
              open(f"{out}/clock_features.json", "w"), indent=1)


def run(cfg, tab, nhats, log=print, stop_after=None):
    device = "cuda"
    assert torch.cuda.is_available(), "turn on a GPU: Runtime > Change runtime type"
    t0 = time.time()
    idx_all = stage_prep(cfg, tab, nhats, log)   # row order = memmap row order; `ok` marks readable images
    cropnet = stage_cropnet(cfg, idx_all, device, log)
    fin = stage_crop(cfg, idx_all, cropnet, device, log)
    if stop_after == "crop":
        log("stopped after the crop stage (part 1 done)"); return
    model, va, te = stage_vit(cfg, idx_all, fin, device, log)
    thr, te = evaluate(cfg, model, va, te, device, log)
    export(cfg, model, cropnet, thr, te, device, log)
    log(f"clock pipeline finished in {(time.time() - t0) / 60:.1f} min")
