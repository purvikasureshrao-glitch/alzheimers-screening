"""Builds notebooks/clock_improve.ipynb (run: python scripts/build_clock_improve_nb.py). Embeds clock/improve.py; uploads needed:
clock_code.tar, clock_data_lean.tar, clock_models_parts/* (the current models split into <=80 MB parts, for the baseline)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md("""
# Clock - fewer false alarms on mouse-drawn clocks, without overfitting [OURS]
**Runs on:** Colab **GPU** (T4), **no Drive**. NHATS must not go to Kaggle or any AI assistant: **do NOT use Colab's Gemini here**, do not paste or screenshot NHATS images.

Upload to the Files panel (`/content`) from `D:\\Alzheimers\\`: `colab_upload/clock_code.tar`, `colab_upload/clock_data_lean.tar` (1,044,080,640 bytes), and the 8 files in `colab_upload/clock_models_parts/` (`manifest.json` + 7 `.part` files, each <= 80 MB; the current models = baseline; select them all and drop them on the Files panel). Optional: `my_drawings.zip` (JPEGs your own app saved; see the last section). Then Run all. Expect about 4-6 hours (preprocess, then two ViT trainings); every stage resumes from its checkpoint if Colab disconnects.

**Problem (FINDINGS 3f).** On the NHATS test split the current model flags 32.8% of people while 4.8% have dementia, the Crop-Net did not learn, and a mouse drawing is a different kind of image from the pen scans it was trained on.

**Variants (fixed in advance):**
- `V0` current models (Crop-Net crop + ViT): the baseline, re-evaluated here.
- `V1` plain crop (no Crop-Net) + ViT retrained, original paper augmentation.
- `V2` = V1 + mouse-style augmentation (random line thickness, hand wobble, slight blur) during training.

**Same split, same protocol as before:** person-grouped split from `seed=0` (identical people in train / val / test), test = clean NHATS test images, no tuning on test.
**Two operating points, both chosen on VALIDATION only:** `BA` = threshold with max balanced accuracy (what the app uses today); `SPEC90` = smallest threshold that leaves >= 90% of healthy validation people unflagged (fewer false alarms, lower sensitivity; a design choice, shown openly).
**Mouse probe:** the clean test images are re-rendered with a *different* distortion family than V2 trained on (broken lines, thick lines, tilt). We report how many people WITHOUT dementia get flagged clean vs probed. A model that flags a healthy person's clock just because it looks mouse-drawn is not acceptable.
**Selection rule (at the BA operating point; SPEC90 is an optional trade-off you may choose afterwards):** a variant is eligible if its test specificity is >= V0's, its balanced accuracy is not lower than V0's by more than 0.02 (paired bootstrap over people, lower CI bound), and its probed false-flag rate is <= V0's. Recommended = eligible variant with the highest balanced accuracy; otherwise keep V0.
**Honest limits:** no mouse-drawn clocks with clinical labels exist, so the probe and the augmentation are proxies. Real validation needs labelled on-screen drawings.
""")
code("""
import os, sys, subprocess, json, zipfile
os.makedirs('/content/Alzheimers', exist_ok=True); os.makedirs('/content/data', exist_ok=True); os.makedirs('/content/out', exist_ok=True)
subprocess.run(['tar', '-xf', '/content/clock_code.tar', '-C', '/content/Alzheimers'], check=True)
subprocess.run(['tar', '-xf', '/content/clock_data_lean.tar', '-C', '/content/data'], check=True)
# baseline weights arrive as small parts (one big upload kept getting corrupted); rejoin and verify each file by sha256
import glob, hashlib
os.makedirs('/content/v0', exist_ok=True)
man = json.load(open(next(iter(glob.glob('/content/**/manifest.json', recursive=True)))))
for name, m in man.items():
    h = hashlib.sha256()
    with open(f'/content/v0/{name}', 'wb') as w:
        for part in m['parts']:
            f = glob.glob(f'/content/**/{part}', recursive=True)
            assert f, f'missing upload: {part}'
            b = open(f[0], 'rb').read(); h.update(b); w.write(b)
    assert h.hexdigest() == m['sha256'], f'{name}: parts corrupted, re-upload the part files'
    print(name, 'OK', m['size'])
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'onnx', 'onnxruntime'], check=True)
ROOT = '/content/Alzheimers'; NHATS = '/content/data/NHATS'; OUT = '/content/out'
sys.path.insert(0, ROOT)
""")
src = (ROOT / "clock" / "improve.py").read_text(encoding="utf-8")
code(f"%%writefile /content/Alzheimers/clock_improve.py\n{src}")
code("""
import numpy as np, pandas as pd, torch, time, json
from clock import labels, split, pipeline, crop as cropmod
from clock.vit import train as vt
from clock.vit.model import ClockViT
from common.runlog import RunLog
import clock_improve as ci
log = RunLog(OUT + '/clock_improve.log'); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE - set Runtime > GPU')
# fail-fast smoke test of the new torch code on random data
_g = torch.Generator(device='cuda'); _g.manual_seed(0); _x = (torch.rand(8, 1, 224, 224, device='cuda') > 0.97).float()
assert ci.mouse_aug(_x, _g).shape == _x.shape and ci.mouse_probe(_x).shape == _x.shape; print('augmentation smoke test OK')
""")
md("## 1. Labels and the same split as the baseline")
code("""
with log.tee('labels + split'):
    tab = labels.build_label_table(NHATS, NHATS + '/survey'); labels.summarize(tab)
    tab = split.split_by_spid(tab, val_images=1000, test_frac=0.10, seed=0); split.report(tab)
""")
md("## 2. Preprocess (same code as the baseline; about 30-60 min)")
code("""
CFG = dict(pipeline.FULL); CFG.update(work='/content/work', out=OUT)
with log.tee('prep'):
    idx = pipeline.stage_prep(CFG, tab, NHATS, log=print)
mm = np.load('/content/work/rough512.npy', mmap_mode='r'); N = len(idx)
""")
md("## 3. Inputs for each variant: baseline crop (Crop-Net) and plain crop")
code("""
os.makedirs('/content/work/fin', exist_ok=True)
fin_plain = np.lib.format.open_memmap('/content/work/fin/plain224.npy', mode='w+', dtype=np.uint8, shape=(N, 224, 224))
for s in range(0, N, 512): fin_plain[s:s + 512] = ci.bbox_crop224(np.asarray(mm[s:s + 512]))
fin_plain.flush()
from clock.cropnet.model import CropNet
cn = CropNet().cuda(); cn.load_state_dict(torch.load('/content/v0/cropnet_best.pt', map_location='cuda')); cn.eval()
fin_v0 = np.lib.format.open_memmap('/content/work/fin/v0_224.npy', mode='w+', dtype=np.uint8, shape=(N, 224, 224))
for s in range(0, N, 256):
    out = cropmod.multi_step_crop(np.asarray(mm[s:s + 256]), cn, 3, 'cuda'); fin_v0[s:s + len(out)] = np.stack([cropmod.to224(o) for o in out])
fin_v0.flush(); del cn
print('inputs ready', N)
""")
md("## 4. Train V1 and V2 (V0 is the existing model)")
code("""
dev = 'cuda'
def splits(fin):
    return [pipeline._split(idx, fin, s, dev) for s in ('train', 'val', 'test')]
def train_variant(name, fin, aug):
    d = f'{OUT}/{name}'; os.makedirs(d, exist_ok=True)
    orig = vt.augment
    vt.augment = (lambda x, g: ci.mouse_aug(orig(x, g), g)) if aug else orig
    try:
        tr, va, te = splits(fin); gpu = lambda s: dict(imgs=s['imgs'].to(dev), sc=s['sc'].to(dev), dem=s['dem'].to(dev))
        gtr, gva = gpu(tr), gpu(va); m = ClockViT().to(dev)
        m, b1 = vt.run_phase(m, 'head', gtr, gva, d, dev, 50, 1, CFG['p1_epochs'], CFG['p1_samples'], CFG['vit_patience'], 1, False, CFG['vit_budget'], print)
        m, b2 = vt.run_phase(m, 'full', gtr, gva, d, dev, 4, 10, CFG['p2_epochs'], CFG['p2_samples'], CFG['vit_patience'], 1, True, CFG['vit_budget'], print)
        print(name, 'best balanced-val MSE: head', round(b1, 5), 'full', round(b2, 5))
    finally:
        vt.augment = orig
    return m.eval()
fin_plain_arr = np.load('/content/work/fin/plain224.npy', mmap_mode='r')
with log.tee('train V1'): m1 = train_variant('V1', fin_plain_arr, aug=False)
torch.save(m1.state_dict(), f'{OUT}/V1_final.pt'); del m1; torch.cuda.empty_cache()
with log.tee('train V2'): m2 = train_variant('V2', fin_plain_arr, aug=True)
torch.save(m2.state_dict(), f'{OUT}/V2_final.pt'); del m2; torch.cuda.empty_cache()
""")
md("## 5. Predictions: clean and probed, for all three variants")
code("""
def load(path):
    m = ClockViT(pretrained=False).to(dev); m.load_state_dict(torch.load(path, map_location=dev)); return m.eval()
VAR = {'V0': ('/content/v0/vit_full_best.pt', '/content/work/fin/v0_224.npy'), 'V1': (f'{OUT}/V1_final.pt', '/content/work/fin/plain224.npy'),
       'V2': (f'{OUT}/V2_final.pt', '/content/work/fin/plain224.npy')}
P = {}
for name, (wp, fp) in VAR.items():
    va, te = [pipeline._split(idx, np.load(fp, mmap_mode='r'), s, dev) for s in ('val', 'test')]
    m = load(wp); _, pv = vt.predict(m, va['imgs'], dev); sc, pt = vt.predict(m, te['imgs'], dev)
    probed = torch.cat([ci.mouse_probe(te['imgs'][i:i + 256].to(dev).float().div(255).unsqueeze(1)).squeeze(1).mul(255).clamp(0, 255).byte().cpu() for i in range(0, len(te['imgs']), 256)])
    _, pp = vt.predict(m, probed, dev)
    P[name] = dict(pv=pv, yv=va['dem'].numpy(), pt=pt, yt=te['dem'].numpy(), g=te['spid'], pp=pp, sc=sc, true_sc=te['sc'].numpy()); del m; torch.cuda.empty_cache()
print({k: (len(v['pv']), len(v['pt'])) for k, v in P.items()})
""")
md("## 6. Metrics at both operating points (thresholds from VALIDATION only)")
code("""
from common.metrics import balanced_acc_ci, auc_ci
from sklearn.metrics import balanced_accuracy_score
rows, THR = [], {}
for name, d in P.items():
    cands = np.linspace(0, 1, 201)
    t_ba = float(max(cands, key=lambda t: balanced_accuracy_score(d['yv'], (d['pv'] >= t).astype(int)))); t_sp = ci.spec_threshold(d['yv'], d['pv'])
    THR[name] = dict(BA=t_ba, SPEC90=t_sp)
    for op, t in THR[name].items():
        pred = (d['pt'] >= t).astype(int) if op == 'BA' else (d['pt'] > t).astype(int); y = d['yt']
        pprobe = (d['pp'] >= t).astype(int) if op == 'BA' else (d['pp'] > t).astype(int)
        b = balanced_acc_ci(y, pred, f'NHATS test, SPID-grouped, thr {t:.3f} from validation', d['g'])
        rows.append(dict(variant=name, op=op, thr=round(t, 3), BA=round(b.value, 3), BA_lo=round(b.lo, 3), BA_hi=round(b.hi, 3), AUC=round(auc_ci(y, d['pt'], '', d['g']).value, 3),
                         sens=round(pred[y == 1].mean(), 3), spec=round(1 - pred[y == 0].mean(), 3), flagged_pct=round(100 * pred.mean(), 1),
                         healthy_flag_clean_pct=round(100 * pred[y == 0].mean(), 1), healthy_flag_probe_pct=round(100 * pprobe[y == 0].mean(), 1), n_images=len(y), n_people=b.n))
tab_res = pd.DataFrame(rows); print(tab_res.to_string(index=False)); print('\\nprevalence of dementia in test images: %.1f%%' % (100 * P['V0']['yt'].mean()))
""")
md("> **SEND BACK TO CLAUDE:** the whole table above (aggregate numbers only).")
md("## 7. Selection rule (fixed in advance)")
code("""
def paired_ba_diff(a, b, n_boot=2000, seed=0):
    ta, tb = THR[a]['BA'], THR[b]['BA']; pa = P[a]['pt'] >= ta; pb = P[b]['pt'] >= tb
    y, g = P[a]['yt'], P[a]['g']; ids = np.unique(g); rows = {i: np.flatnonzero(g == i) for i in ids}; rng = np.random.default_rng(seed); v = []
    for _ in range(n_boot):
        pick = np.concatenate([rows[i] for i in rng.choice(ids, size=len(ids))])
        if len(np.unique(y[pick])) < 2: continue
        v.append(balanced_accuracy_score(y[pick], pa[pick].astype(int)) - balanced_accuracy_score(y[pick], pb[pick].astype(int)))
    return np.percentile(v, [2.5, 97.5])
elig = []; base = tab_res[(tab_res.variant == 'V0') & (tab_res.op == 'BA')].iloc[0]
for v in ('V1', 'V2'):                                   # eligibility is judged at the BA operating point, like today's app
    r = tab_res[(tab_res.variant == v) & (tab_res.op == 'BA')].iloc[0]; lo, hi = paired_ba_diff(v, 'V0')
    ok = r.spec >= base.spec and lo > -0.02 and r.healthy_flag_probe_pct <= base.healthy_flag_probe_pct
    print(f'{v}: spec {r.spec} vs V0 {base.spec}; BA diff vs V0 95% CI ({lo:.3f}, {hi:.3f}); probed healthy flagged {r.healthy_flag_probe_pct}% vs {base.healthy_flag_probe_pct}% ->', 'ELIGIBLE' if ok else 'not eligible')
    if ok: elig.append((r.BA, v, 'BA'))
best = max(elig) if elig else None
print('\\nRECOMMENDED variant:', best[1] if best else 'keep V0 (no variant met the rule)')
print('SPEC90 rows in the table are an optional trade-off (fewer false alarms, lower sensitivity); choosing it is a decision for the user, not part of the rule.')
""")
md("> **SEND BACK TO CLAUDE:** everything this cell prints.")
md("""
## 8. Optional: your own drawings
Upload `my_drawings.zip` (JPEGs saved by your app, e.g. a clock you drew correctly) to see what each variant says about them. These are your own images, not NHATS, but still do not paste them anywhere.
""")
code("""
import glob, io, shutil
from PIL import Image, ImageOps
from clock import preprocess
def rough_512(b, LONG=828):
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(b))).convert('L'); k = LONG / max(im.size)
    im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.BOX)
    a = preprocess.remove_name_boxes(preprocess.invert(np.asarray(im, np.float32) / 255.0), min_side=10)
    a = preprocess.pad_square(preprocess.crop_to_mass(a, margin=12), 512); return (np.clip(a, 0, 1) * 255).astype(np.uint8)
if os.path.exists('/content/my_drawings.zip'):
    shutil.rmtree('/content/mine', ignore_errors=True); zipfile.ZipFile('/content/my_drawings.zip').extractall('/content/mine')
    files = sorted(glob.glob('/content/mine/**/*.jp*g', recursive=True)); R = np.stack([rough_512(open(f, 'rb').read()) for f in files])
    cn = CropNet().cuda(); cn.load_state_dict(torch.load('/content/v0/cropnet_best.pt', map_location='cuda')); cn.eval()
    inp = {'V0': np.stack([cropmod.to224(o) for o in cropmod.multi_step_crop(R, cn, 3, 'cuda')]), 'V1': ci.bbox_crop224(R), 'V2': ci.bbox_crop224(R)}
    for name, (wp, _) in VAR.items():
        m = load(wp); sc, p = vt.predict(m, torch.from_numpy(inp[name]), dev); t = THR[name]['BA']; t2 = THR[name]['SPEC90']
        print(name, [f'{os.path.basename(f)[:12]}: score {s:.1f}, P {q:.3f}, flag BA {int(q >= t)} / SPEC90 {int(q > t2)}' for f, s, q in zip(files, sc, p)])
else:
    print('no my_drawings.zip uploaded; skipped')
""")
md("## 9. Export the recommended ViT (ONNX) and download aggregate results")
code("""
import onnxruntime as ort
if best:
    _, v, op = best; m = load(VAR[v][0]).cpu().eval(); x = torch.zeros(1, 1, 224, 224)
    torch.onnx.export(m, x, f'{OUT}/clock_vit_{v}.onnx', input_names=['x'], output_names=['y'], dynamic_axes={'x': {0: 'n'}, 'y': {0: 'n'}}, opset_version=17, dynamo=False)
    xr = torch.rand(4, 1, 224, 224)
    with torch.no_grad(): d = np.abs(m(xr).numpy() - ort.InferenceSession(f'{OUT}/clock_vit_{v}.onnx').run(None, {'x': xr.numpy()})[0]).max()
    print('ONNX vs PyTorch max diff', d)
    json.dump(dict(variant=v, operating_point=op, dementia_threshold=THR[v][op], flag_rule=('p >= thr' if op == 'BA' else 'p > thr'),
                   crop=('Crop-Net x3' if v == 'V0' else 'plain 96%-mass crop, no Crop-Net'), table=tab_res.to_dict('records')), open(f'{OUT}/clock_improved.json', 'w'), indent=1)
tab_res.to_csv(f'{OUT}/clock_improve_table.csv', index=False)
with zipfile.ZipFile('/content/clock_improve_results.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for f in ('clock_improve.log', 'clock_improve_table.csv', 'clock_improved.json', 'clock_vit_V1.onnx', 'clock_vit_V2.onnx'):
        if os.path.exists(f'{OUT}/{f}'): z.write(f'{OUT}/{f}', f)
from google.colab import files
files.download('/content/clock_improve_results.zip')
""")
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks" / "clock_improve.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
