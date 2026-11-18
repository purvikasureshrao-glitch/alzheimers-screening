"""Builds notebooks/speech_improve.ipynb (run: python scripts/build_speech_improve_nb.py). The notebook embeds speech/whisper_features.py and
speech/improve_features.py so only speech_data.tar has to be uploaded."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md("""
# Speech - improving the live model without overfitting [OURS]
**Runs on:** Colab (GPU), **no Drive**. ADReSS must not go to Kaggle. Upload only `speech_data.tar` (687,421,440 bytes, from `D:\\Alzheimers\\colab_upload\\`). The code is embedded below.
Do NOT use Colab's Gemini assistant with the data loaded. Transcripts never leave Colab; only aggregate numbers and feature tables go back.

**Question.** Whisper hides most "um"s, repeats and pauses, so the live model sees weak signals (5-fold CV 0.639, 0.546-0.731, FINDINGS 9b). Can (a) patient-only words, (b) a transcriber that keeps fillers, (c) more robust features, (d) a text embedding raise balanced accuracy **without** flagging more healthy people?

**Fixed in advance (before any result):**
- Candidates (up to 8): `A` = Whisper small, whole file, 5 features (what the app does today = the baseline) | `B` Whisper small, patient words only, 5 features | `B+` same, extended features | `C`, `C+` same two with CrisperWhisper (if it loads) | `E_B`, `E_C` MiniLM transcript embedding on B / C words.
- Model: standardise + L2 logistic regression, strength picked by an INNER 5-fold CV on the training folds only. Balanced class weights, threshold 0.5.
- Evaluation: 5 x repeated stratified 5-fold CV on the 108 people (one recording each, so folds are subject-level). Per-person out-of-fold probability = mean over the 5 repeats. 95% CI = bootstrap over people.
- Selection rule: among candidates whose specificity (healthy people correctly NOT flagged) is at least the baseline's and whose BA spread across the 5 repeats is <= 0.04, take the highest balanced accuracy. "Supported" only if the paired bootstrap CI of (winner - baseline) is entirely above 0.
- Sanity: the winner is re-run with shuffled labels; balanced accuracy must be near 0.5 (otherwise something leaks).
- Honest limit: several candidates are compared on the same 108 people, so the winner's number is optimistic. A clean test needs another dataset, or the 48 ADReSS test recordings once their labels are released (not in the folder: `test_results.txt` is blank).
""")
code("""
!pip install -q openai-whisper scikit-learn transformers accelerate sentence-transformers skl2onnx onnx soundfile
import os, subprocess
os.makedirs('/content/data/adress_train', exist_ok=True); os.makedirs('/content/out', exist_ok=True); os.makedirs('/content/speechlib', exist_ok=True)
subprocess.run(['tar', '-xf', '/content/speech_data.tar', '-C', '/content/data/adress_train'], check=True)
open('/content/speechlib/__init__.py', 'w').write('')
""")
for name in ("whisper_features", "improve_features"):
    src = (ROOT / "speech" / f"{name}.py").read_text(encoding="utf-8")
    code(f"%%writefile /content/speechlib/{name}.py\n{src}")
md("## 1. Labels, patient time spans")
code("""
import sys, pathlib, json, time, re, numpy as np, pandas as pd, torch
sys.path.insert(0, '/content')
from speechlib import improve_features as imp
from speechlib.whisper_features import features_from_words, WHISPER_FEATURE_NAMES
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'; print(DEVICE)
root = pathlib.Path('/content/data/adress_train'); rows = []
for grp, label in [('cc', 0), ('cd', 1)]:
    for cha in sorted((root/'transcription'/grp).glob('*.cha')):
        text = cha.read_text(encoding='utf-8', errors='replace')
        rows.append(dict(id=cha.stem, label=label, wav=str(root/'Full_wave_enhanced_audio'/grp/f'{cha.stem}.wav'), spans=imp.par_spans(text)))
df = pd.DataFrame(rows); y = df.label.values
print(df.label.value_counts().to_dict(), 'n =', len(df), '| patient seconds (median):', int(np.median([imp.spans_seconds(s) for s in df.spans])))
import soundfile as sf
file_secs = np.array([sf.info(w).duration for w in df.wav])
""")
md("## 2. Whisper small on all 108 (same settings as FINDINGS 9b)")
code("""
import whisper
PROMPT = 'Um, uh, so, er, I think, I think the, the mm-hmm.'
wm = whisper.load_model('small', device=DEVICE); words_w = {}; t0 = time.time()
for i, r in enumerate(df.itertuples()):
    res = wm.transcribe(r.wav, language='en', word_timestamps=True, initial_prompt=PROMPT, condition_on_previous_text=False, fp16=(DEVICE == 'cuda'))
    words_w[r.id] = [(w['word'], float(w['start']), float(w['end'])) for s in res['segments'] for w in s['words']]
    if i % 20 == 0: print(i, len(words_w[r.id]), 'words', f'{time.time()-t0:.0f}s')
del wm
if DEVICE == 'cuda': torch.cuda.empty_cache()
""")
md("""
## 3. CrisperWhisper (keeps "um", "uh", repeats). Optional candidate
If the model needs a Hugging Face access token, add it in Colab: key icon > Secrets > `HF_TOKEN`, then re-run. Check the model page for its licence (the weights' licence is not the code's; research prototype only unless you check). If it fails to load, the notebook continues without candidates C / C+ / E_C.
""")
code("""
CRISPER_ID = 'nyrahealth/CrisperWhisper'
words_c = None
try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    try:
        from google.colab import userdata; tok = userdata.get('HF_TOKEN')
    except Exception:
        tok = None
    dt = torch.float16 if DEVICE == 'cuda' else torch.float32
    m = AutoModelForSpeechSeq2Seq.from_pretrained(CRISPER_ID, torch_dtype=dt, low_cpu_mem_usage=True, use_safetensors=True, token=tok).to(DEVICE)
    proc = AutoProcessor.from_pretrained(CRISPER_ID, token=tok)
    pipe = pipeline('automatic-speech-recognition', model=m, tokenizer=proc.tokenizer, feature_extractor=proc.feature_extractor, chunk_length_s=30, batch_size=8,
                    return_timestamps='word', torch_dtype=dt, device=DEVICE)
    words_c = {}; t0 = time.time()
    for i, r in enumerate(df.itertuples()):
        out = pipe(r.wav)
        ws = []
        for c in out['chunks']:
            a, b = c['timestamp']
            if a is None: continue
            ws.append((c['text'], float(a), float(b if b is not None else a)))
        words_c[r.id] = ws
        if i % 20 == 0: print(i, len(ws), 'words', f'{time.time()-t0:.0f}s')
    del m, pipe
    if DEVICE == 'cuda': torch.cuda.empty_cache()
except Exception as e:
    print('CrisperWhisper not used:', type(e).__name__, str(e)[:300])
""")
md("## 4. Features for every candidate")
code("""
def build(words, patient_only):
    F5, FX, TXT = [], [], []
    for i, r in enumerate(df.itertuples()):
        w = words[r.id]
        if patient_only:
            w = imp.keep_in_spans(w, r.spans); secs = imp.spans_seconds(r.spans)
        else:
            secs = float(file_secs[i])
        F5.append(features_from_words(w, secs)); FX.append(imp.features_ext(w, secs)); TXT.append(imp.transcript_text(w))
    return np.array(F5), np.array(FX), TXT
X, TXT = {}, {}
F5, FX, T = build(words_w, False); X['A'] = F5
F5, FX, T = build(words_w, True); X['B'] = F5; X['B+'] = FX; TXT['B'] = T
if words_c:
    F5, FX, T = build(words_c, True); X['C'] = F5; X['C+'] = FX; TXT['C'] = T
for name, ws in (('whisper', words_w), ('crisper', words_c)):
    if ws:
        share = np.mean([imp.features_ext(imp.keep_in_spans(ws[r.id], r.spans), 60)[0] > 0 for r in df.itertuples()])
        print(name, 'recordings with >=1 filler (patient-only):', round(100 * share), '%')
from sentence_transformers import SentenceTransformer
st = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=DEVICE)
for k in list(TXT):
    X['E_' + k] = st.encode(TXT[k], batch_size=16, show_progress_bar=False)
print({k: v.shape for k, v in X.items()}, '| all finite:', all(np.isfinite(v).all() for v in X.values()))
""")
md("## 5. Repeated cross-validation, same protocol for every candidate")
code("""
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import warnings; warnings.filterwarnings('ignore')
REPEATS = 5
def make_model(embed):
    cs = [0.0003, 0.001, 0.003, 0.01] if embed else [0.01, 0.03, 0.1, 0.3, 1.0]
    return make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=cs, cv=5, scoring='balanced_accuracy', class_weight='balanced', max_iter=2000))
def run_cv(Xc, yy, embed, seed0=0):
    oof = np.zeros((REPEATS, len(yy)))
    for rep in range(REPEATS):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed0 + rep).split(Xc, yy):
            m = make_model(embed).fit(Xc[tr], yy[tr]); oof[rep, te] = m.predict_proba(Xc[te])[:, 1]
    return oof
def sens_spec(yy, pred):
    return float(pred[yy == 1].mean()), float(1 - pred[yy == 0].mean())
rng = np.random.default_rng(0); BOOT = rng.integers(0, len(y), size=(2000, len(y)))
def ci(f):
    v = np.array([f(b) for b in BOOT if len(set(y[b])) == 2]); return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
res, oofs = {}, {}
for k, Xc in X.items():
    oof = run_cv(Xc, y, k.startswith('E_')); oofs[k] = oof; p = oof.mean(0); pred = (p >= 0.5).astype(int)
    rep_ba = [balanced_accuracy_score(y, (o >= 0.5).astype(int)) for o in oof]; se, sp = sens_spec(y, pred)
    lo, hi = ci(lambda b: balanced_accuracy_score(y[b], pred[b]))
    res[k] = dict(BA=balanced_accuracy_score(y, pred), BA_lo=lo, BA_hi=hi, AUC=roc_auc_score(y, p), sens=se, spec=sp, flagged_pct=100 * pred.mean(),
                  rep_spread=float(np.ptp(rep_ba)), n=len(y), protocol='5x repeated 5-fold CV, subject-level, thr 0.5')
tab = pd.DataFrame(res).T.round(3); print(tab.drop(columns=['protocol']).to_string()); print('\\nprevious app model (FINDINGS 9b, different CV): BA 0.639 (0.546-0.731)')
""")
md("> **SEND BACK TO CLAUDE:** the whole table above and the 'recordings with >=1 filler' lines.")
md("## 6. Selection rule (fixed in advance) and sanity checks")
code("""
base = res['A']; elig = [k for k, r in res.items() if k != 'A' and r['spec'] >= base['spec'] and r['rep_spread'] <= 0.04]
print('eligible (spec >= baseline spec and stable):', elig)
if elig:
    win = max(elig, key=lambda k: res[k]['BA']); pw, pb = oofs[win].mean(0), oofs['A'].mean(0)
    diff = lambda b: balanced_accuracy_score(y[b], (pw[b] >= .5).astype(int)) - balanced_accuracy_score(y[b], (pb[b] >= .5).astype(int))
    lo, hi = ci(diff)
    print(f"winner {win}: BA {res[win]['BA']:.3f} vs baseline {base['BA']:.3f}; paired difference 95% CI ({lo:.3f}, {hi:.3f}) ->", 'SUPPORTED' if lo > 0 else 'NOT supported (interval includes 0)')
    ysh = np.random.default_rng(1).permutation(y); sh = run_cv(X[win], ysh, win.startswith('E_')).mean(0)
    print('shuffled-label sanity BA (should be near 0.5):', round(balanced_accuracy_score(ysh, (sh >= .5).astype(int)), 3))
else:
    win = None; print('No candidate meets the rule: keep the current model.')
""")
md("> **SEND BACK TO CLAUDE:** everything this cell prints.")
md("## 7. Export the winner (only if it is a features model; use it in the app only if SUPPORTED)")
code("""
if win and not win.startswith('E_'):
    from skl2onnx import to_onnx
    final = make_model(False).fit(X[win], y); nfeat = X[win].shape[1]
    onx = to_onnx(final, X[win][:1].astype(np.float32), options={id(final.steps[-1][1]): {'zipmap': False}}, target_opset=15)
    open('/content/out/speech_improved.onnx', 'wb').write(onx.SerializeToString())
    json.dump(dict(candidate=win, n_features=nfeat, names=(imp.FEXT_NAMES if win.endswith('+') else WHISPER_FEATURE_NAMES), asr=('crisperwhisper' if win.startswith('C') else 'whisper-small'),
                   patient_only=win != 'A', cv=res[win]), open('/content/out/speech_improved.json', 'w'), indent=1)
    print('exported', win)
else:
    print('nothing exported')
""")
md("## 8. Download (features and numbers only; no transcripts)")
code("""
import shutil
tab.to_csv('/content/out/speech_improve_table.csv')
for k, v in X.items():
    if not k.startswith('E_'): pd.DataFrame(v).assign(id=df.id.values, label=y).to_csv(f'/content/out/features_{k}.csv', index=False)
pd.DataFrame({k: o.mean(0) for k, o in oofs.items()}).assign(id=df.id.values, label=y).to_csv('/content/out/oof.csv', index=False)
shutil.make_archive('/content/speech_improve_results', 'zip', '/content/out')
from google.colab import files
files.download('/content/speech_improve_results.zip')
""")
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks" / "speech_improve.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
