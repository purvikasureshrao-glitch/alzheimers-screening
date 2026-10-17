"""H5 multi-step crop (x3) at working scale [OURS]: heat map (channel 0, 64x64) -> bilinear to 512 -> window holding 84% of
heat mass (+20 px) -> window holding 96% of pixel mass (+8 px) -> pad square -> 512. The margins are the paper's 80 / 30 px
divided by 4 because images are area-averaged 4x on load (preprocess.PRE_REDUCE)."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import preprocess


def _heat(model, imgs, device, bs=32):
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(imgs), bs):
            x = torch.from_numpy(imgs[i:i + bs]).to(device).float().div(255).unsqueeze(1)
            h = model(x)[:, :1].float()
            out.append(F.interpolate(h, size=(512, 512), mode="bilinear", align_corners=False)[:, 0].cpu().numpy())
    return np.concatenate(out)


def _window(profile, mass, margin, n):
    s, e = preprocess._min_window(profile, mass)
    return max(0, s - margin), min(n, e + margin)


def _crop_one(img_u8, heat):
    img = img_u8.astype(np.float32) / 255.0
    if heat.sum() > 0:
        r0, r1 = _window(heat.sum(1), 0.84, 20, 512)
        c0, c1 = _window(heat.sum(0), 0.84, 20, 512)
        if r1 > r0 and c1 > c0:
            img = img[r0:r1, c0:c1]
    img = preprocess.crop_to_mass(img, 0.96, 8)
    return (np.clip(preprocess.pad_square(img, 512), 0, 1) * 255).astype(np.uint8)


def multi_step_crop(imgs, model, iterations=3, device="cuda"):
    """imgs: uint8 (N,512,512), ink bright. Returns uint8 (N,512,512)."""
    cur = np.asarray(imgs)
    for _ in range(iterations):
        heat = _heat(model, cur, device)
        cur = np.stack([_crop_one(a, h) for a, h in zip(cur, heat)])
    return cur


def qc_stats(crops):
    """Descriptive crop quality numbers (no rejection rule is fitted: H6 needs the user's grades)."""
    ink = crops > 51
    border = ink[:, :3].any((1, 2)) | ink[:, -3:].any((1, 2)) | ink[:, :, :3].any((1, 2)) | ink[:, :, -3:].any((1, 2))
    return ink.mean((1, 2)), border


def to224(crop512):
    return np.asarray(Image.fromarray(crop512).resize((224, 224), Image.BOX))
