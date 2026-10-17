"""Crop-Net training (PLAN H4): MSE, Adam lr 2e-5 betas (0.5, 0.999), batch 50, LR /5 on plateau, early stopping.

Checkpoint (model + optimiser + counters) is written every evaluation; a rerun resumes from it. [OURS] iteration budget,
evaluation interval and patience (the paper gives only 'early stopping'); validation = fixed synthetic pages.
"""
import os
import time

import torch
import torch.nn.functional as F

from .model import CropNet
from .synth import make_batch


def train(bank_tr, sc_tr, bank_va, sc_va, out_dir, max_iters, device, eval_every=100, patience=15,
          batch=50, micro=25, time_budget_s=None, n_val=500, log=print):
    os.makedirs(out_dir, exist_ok=True)
    model = CropNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-5, betas=(0.5, 0.999))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.2, patience=5)
    scaler = torch.amp.GradScaler("cuda")
    gv = torch.Generator(device=device); gv.manual_seed(12345)
    val = [make_batch(bank_va, sc_va, min(50, n_val - i), gv) for i in range(0, n_val, 50)]
    val = [(x.half(), y) for x, y in val]
    g = torch.Generator(device=device); g.manual_seed(0)
    step, best, bad = 0, float("inf"), 0
    last = os.path.join(out_dir, "cropnet_last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step, best, bad = ck["step"], ck["best"], ck["bad"]
        g.manual_seed(step + 1)
        log(f"resumed Crop-Net at step {step}, best val {best:.5f}")
    t0 = time.time()
    while step < max_iters:
        model.train()
        for _ in range(batch // micro):
            x, y = make_batch(bank_tr, sc_tr, micro, g)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.mse_loss(model(x).float(), y) * micro / batch
            scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % eval_every == 0 or step == max_iters:
            model.eval(); tot = 0.0
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                for x, y in val:
                    tot += F.mse_loss(model(x).float(), y).item() * len(x)
            v = tot / n_val
            sched.step(v)
            if v < best:
                best, bad = v, 0
                torch.save(model.state_dict(), os.path.join(out_dir, "cropnet_best.pt"))
            else:
                bad += 1
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            step=step, best=best, bad=bad), last)
            log(f"crop-net step {step}/{max_iters} val MSE {v:.5f} best {best:.5f} lr {opt.param_groups[0]['lr']:.1e} "
                f"[{time.time() - t0:.0f}s]")
            if bad >= patience:
                log("early stopping"); break
            if time_budget_s and time.time() - t0 > time_budget_s:
                log("time budget reached, stopping Crop-Net"); break
    model.load_state_dict(torch.load(os.path.join(out_dir, "cropnet_best.pt"), map_location=device))
    return model.eval(), best
