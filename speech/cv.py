"""5-fold CV on the 108 labelled ADReSS subjects (PLAN B1).

Paper text: batch 8, Adam lr 0.01, cross-entropy. [OURS] checkpointing: best validation
loss on an INNER split of each training fold, never on the fold being reported.
Acoustic scaler + PCA(21) are fitted on the training fold only (official code).
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from .models import MODELS

BATCH, LR, EPOCHS, PATIENCE = 8, 0.01, 2000, 200  # patience: [OURS] stops dead epochs early


def fit_one(name, Xtr, ytr, Xva, yva, seed, device="cpu"):
    torch.manual_seed(seed)
    model = MODELS[name]().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xtr, ytr = torch.tensor(Xtr, dtype=torch.float32, device=device), torch.tensor(ytr, device=device)
    Xva, yva = torch.tensor(Xva, dtype=torch.float32, device=device), torch.tensor(yva, device=device)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:  # BatchNorm needs >1 sample
                continue
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
            loss = loss + model.penalty()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = F.cross_entropy(model(Xva), yva).item()
        if v < best:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model.eval()


def predict_proba(model, X, device="cpu"):
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(X, dtype=torch.float32, device=device)), -1)[:, 1].cpu().numpy()


def cross_validate(feats, y, seed=0, n_folds=5, device="cpu"):
    """feats: {'disfluency': (N,11), 'acoustic_raw': (N,6373), 'interventions': (N,32,3)}.
    Returns out-of-fold P(AD) per network + hard-vote ensemble, and fold index per subject."""
    y = np.asarray(y)
    oof = {k: np.zeros(len(y)) for k in ["disfluency", "acoustic", "interventions"]}
    fold_of = np.zeros(len(y), int)
    for f, (tr, te) in enumerate(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(y, y)):
        fold_of[te] = f
        itr, iva = train_test_split(tr, test_size=0.2, stratify=y[tr], random_state=seed)  # inner split
        sc = StandardScaler().fit(feats["acoustic_raw"][tr])  # scaler + PCA on the training fold only
        pca = PCA(21, random_state=seed).fit(sc.transform(feats["acoustic_raw"][tr]))
        ac = pca.transform(sc.transform(feats["acoustic_raw"]))
        X = {"disfluency": feats["disfluency"], "acoustic": ac, "interventions": feats["interventions"]}
        for name in oof:
            m = fit_one(name, X[name][itr], y[itr], X[name][iva], y[iva], seed + f, device)
            oof[name][te] = predict_proba(m, X[name][te], device)
    votes = np.stack([oof[k] >= 0.5 for k in oof]).sum(0)
    oof["ensemble_hard_vote"] = (votes >= 2).astype(float)
    return oof, fold_of
