"""Clock improvement experiments [OURS, 2026-10-20]: simple crop without Crop-Net, mouse-style training augmentation, an independent
mouse-style probe, and a specificity-targeted threshold. Used by notebooks/clock_improve.ipynb (Colab). Constants fixed before any result.

Why: the Crop-Net did not learn (FINDINGS 3f), and a drawing made with a mouse is a different kind of image from the NHATS pen scans the
ViT saw. The test is always the clean NHATS test split (person-grouped) plus the probe below; nothing is tuned on the test set.
"""
import numpy as np
import torch
import torch.nn.functional as F

from clock import crop as cropmod

SPEC_TARGET = 0.90   # fixed in advance: threshold with >= 90% of healthy validation people NOT flagged


def bbox_crop224(rough512_u8):
    """uint8 (N,512,512) rough crops (ink bright) -> uint8 (N,224,224): plain 96%-mass crop, no Crop-Net."""
    out = []
    for a in rough512_u8:
        out.append(cropmod.to224(cropmod._crop_one(a, np.zeros((512, 512), np.float32))))
    return np.stack(out)


def _u(B, lo, hi, g, dev):
    return lo + (hi - lo) * torch.rand(B, generator=g, device=dev)


def _warp(x, flow):
    B, _, H, W = x.shape
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H, device=x.device), torch.linspace(-1, 1, W, device=x.device), indexing="ij")
    base = torch.stack([xs, ys], -1)[None].expand(B, -1, -1, -1)
    return F.grid_sample(x, base + flow.permute(0, 2, 3, 1), mode="bilinear", padding_mode="zeros", align_corners=True)


def mouse_aug(x, g, p=0.7):
    """Training family: random line thickness (thin/normal/thick), smooth hand wobble, slight blur. x: (B,1,H,W) in [0,1], ink bright."""
    B, dev = x.shape[0], x.device
    k = torch.randint(0, 4, (B,), generator=g, device=dev)[:, None, None, None]
    variants = [x, F.max_pool2d(x, 3, 1, 1), F.max_pool2d(x, 5, 1, 2), -F.max_pool2d(-x, 3, 1, 1)]
    y = variants[0]
    for i in (1, 2, 3):
        y = torch.where(k == i, variants[i], y)
    amp = _u(B, 0.005, 0.03, g, dev)[:, None, None, None]
    coarse = torch.randn(B, 2, 5, 5, generator=g, device=dev) * amp
    y = _warp(y, F.interpolate(coarse, size=x.shape[-2:], mode="bicubic", align_corners=True))
    blur = torch.rand(B, generator=g, device=dev)[:, None, None, None] < 0.3
    y = torch.where(blur, F.avg_pool2d(y, 3, 1, 1), y)
    keep = torch.rand(B, generator=g, device=dev)[:, None, None, None] < p
    return torch.where(keep, y.clamp(0, 1), x)


def mouse_probe(x, seed=123):
    """EVALUATION family, deliberately different from mouse_aug: broken lines (erased patches), thick uniform lines, small rotation.
    Deterministic. Applied to the clean test images to ask 'does the model flag a healthy person's clock just because it looks mouse-drawn?'"""
    g = torch.Generator(device=x.device); g.manual_seed(seed)
    B, _, H, W = x.shape
    y = F.max_pool2d(x, 3, 1, 1)
    for _ in range(6):
        cy, cx = torch.randint(10, H - 10, (B,), generator=g, device=x.device), torch.randint(10, W - 10, (B,), generator=g, device=x.device)
        yy, xx = torch.arange(H, device=x.device)[None, :, None], torch.arange(W, device=x.device)[None, None, :]
        hole = ((yy - cy[:, None, None]).abs() < 5) & ((xx - cx[:, None, None]).abs() < 5)
        y = y * (~hole)[:, None].float()
    ang = _u(B, -4, 4, g, x.device) * np.pi / 180
    c, s = ang.cos(), ang.sin()
    theta = torch.stack([torch.stack([c, -s, torch.zeros_like(c)], -1), torch.stack([s, c, torch.zeros_like(c)], -1)], -2)
    return F.grid_sample(y, F.affine_grid(theta, tuple(y.shape), align_corners=False), mode="bilinear", padding_mode="zeros", align_corners=False)


def spec_threshold(y_val, p_val, target=SPEC_TARGET):
    """Smallest threshold t such that flagging p > t leaves >= target of the healthy validation people unflagged."""
    healthy = np.sort(p_val[np.asarray(y_val) == 0])
    return float(healthy[min(len(healthy) - 1, int(np.ceil(target * len(healthy))) - 1)])
