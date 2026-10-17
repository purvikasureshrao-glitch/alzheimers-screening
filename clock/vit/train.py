"""ViT training and evaluation (PLAN H7).

Paper: MSE, Adam lr 2e-5 betas (0.5, 0.999), LR /5 on plateau, early stopping; phase 1 head only (batch 50), phase 2 whole
network (batch 4, gradient accumulation 10); replicate under-represented clock scores so 0-5 are equal; affine augmentation
(scale 0.7-1, translate 0-0.5, rotate +-180, shear +-10) in train AND validation. Test: natural distribution, no augmentation.
[OURS] epoch = a fixed number of balanced draws; patience values; fixed seeded validation set; decision threshold chosen on
the natural validation set (max balanced accuracy) and applied unchanged to test.
Targets: output 0 = clock score / 5, output 1 = dementia (0/1). MSE on both.
"""
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F


def augment(x, g):
    """x: (B,1,H,W) float. Random affine on the GPU."""
    B, dev = x.shape[0], x.device
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(B, generator=g, device=dev)
    ang, sh, s = u(-math.pi, math.pi), u(-10, 10) * math.pi / 180, u(0.7, 1.0)
    t = torch.stack([u(0, 0.5) * (torch.randint(0, 2, (B,), generator=g, device=dev) * 2 - 1),
                     u(0, 0.5) * (torch.randint(0, 2, (B,), generator=g, device=dev) * 2 - 1)], 1) * 2   # fraction of the image -> [-1,1] units
    c, sn = ang.cos(), ang.sin()
    R = torch.stack([torch.stack([c, -sn], -1), torch.stack([sn, c], -1)], -2)
    one, zero = torch.ones_like(sh), torch.zeros_like(sh)
    Sh = torch.stack([torch.stack([one, sh.tan()], -1), torch.stack([zero, one], -1)], -2)
    L = (R @ Sh) * s[:, None, None]
    Li = torch.linalg.inv(L)
    theta = torch.cat([Li, -(Li @ t[:, :, None])], 2)
    grid = F.affine_grid(theta, tuple(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def balanced_idx(scores_int, n, g):
    w = 1.0 / torch.bincount(scores_int, minlength=6).float()[scores_int]
    return torch.multinomial(w, n, replacement=True, generator=g)


def _targets(sc, dem):
    return torch.stack([sc.float() / 5, dem.float()], 1)


@torch.no_grad()
def predict(model, imgs_u8, device, bs=128):
    model.eval(); outs = []
    for i in range(0, len(imgs_u8), bs):
        x = imgs_u8[i:i + bs].to(device).float().div(255).unsqueeze(1)
        with torch.autocast("cuda", dtype=torch.float16):
            outs.append(model(x).float().cpu())
    o = torch.cat(outs).numpy()
    return o[:, 0] * 5, o[:, 1]


def run_phase(model, tag, tr, va, ckpt_dir, device, batch, accum, epochs, samples, patience, lr_patience,
              body_trainable, time_budget_s=None, log=print):
    """tr / va: dict(imgs uint8 (N,224,224) on device, sc int tensor, dem tensor)."""
    for p in model.body.parameters():
        p.requires_grad = body_trainable
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=2e-5, betas=(0.5, 0.999))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.2, patience=lr_patience)
    scaler = torch.amp.GradScaler("cuda")
    gv = torch.Generator(device=device); gv.manual_seed(777)
    n_val = min(1500, 4 * len(va["sc"]))
    vi = balanced_idx(va["sc"], n_val, gv)
    vx = torch.cat([augment(va["imgs"][vi[i:i + 250]].float().div(255).unsqueeze(1), gv).half() for i in range(0, n_val, 250)])
    vy = _targets(va["sc"][vi], va["dem"][vi]).to(device)
    last, best_path = os.path.join(ckpt_dir, f"vit_{tag}_last.pt"), os.path.join(ckpt_dir, f"vit_{tag}_best.pt")
    ep0, best, bad = 0, float("inf"), 0
    g = torch.Generator(device=device); g.manual_seed(1)
    if os.path.exists(last):
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        ep0, best, bad = ck["epoch"], ck["best"], ck["bad"]
        g.manual_seed(ep0 + 1)
        log(f"resumed {tag} at epoch {ep0}, best val {best:.5f}")
    t0 = time.time()
    for ep in range(ep0, epochs):
        model.train()
        if not body_trainable:
            model.body.eval()
        idx = balanced_idx(tr["sc"], samples, g)
        tot, nb = 0.0, 0
        for k, i in enumerate(range(0, samples, batch)):
            j = idx[i:i + batch]
            x = augment(tr["imgs"][j].float().div(255).unsqueeze(1), g)
            y = _targets(tr["sc"][j], tr["dem"][j])
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.mse_loss(model(x).float(), y) / accum
            scaler.scale(loss).backward()
            if (k + 1) % accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            tot += loss.item() * accum; nb += 1
        model.eval(); vt = 0.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(0, n_val, 250):
                vt += F.mse_loss(model(vx[i:i + 250]).float(), vy[i:i + 250], reduction="sum").item()
        v = vt / (n_val * 2)
        sched.step(v)
        if v < best:
            best, bad = v, 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        epoch=ep + 1, best=best, bad=bad), last)
        log(f"{tag} epoch {ep + 1}/{epochs} train MSE {tot / nb:.5f} val MSE {v:.5f} best {best:.5f} "
            f"lr {opt.param_groups[0]['lr']:.1e} [{time.time() - t0:.0f}s]")
        if bad >= patience:
            log(f"{tag}: early stopping"); break
        if time_budget_s and time.time() - t0 > time_budget_s:
            log(f"{tag}: time budget reached"); break
    model.load_state_dict(torch.load(best_path, map_location=device))
    return model.eval(), best
