"""Synthetic training pages for Crop-Net, built on the GPU (PLAN H4).

A real rough-cropped clock is rotated (<=180 deg), shrunk to 1/8..1 and placed on a black 512 canvas; noise, lines (single /
parallel / lined paper), writing and rectangles are added. Noise and lines may overlap the clock, writing and rectangles may not.
Target = (clock area, clock area x score/5), area-averaged to 64x64.
[OURS] "writing" = random scribbles (the paper's source of writing is not specified); all amplitudes/ranges are our choice.
"""
import math

import torch
import torch.nn.functional as F

SIZE = 512
_C = {}


def _u(g, lo, hi, *shape, device):
    return lo + (hi - lo) * torch.rand(*shape, generator=g, device=device)


def _coords(dev):
    key = str(dev)
    if key not in _C:
        ax = (torch.arange(SIZE, device=dev, dtype=torch.float32) + 0.5) / SIZE * 2 - 1
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
        _C[key] = (xx, yy)
    return _C[key]


def place(clocks, g):
    B, dev = clocks.shape[0], clocks.device
    ang = _u(g, -math.pi, math.pi, B, device=dev)
    c, s = ang.cos(), ang.sin()
    ext = c.abs() + s.abs()
    scale = torch.minimum(torch.exp(_u(g, math.log(1 / 8), 0.0, B, device=dev)), 1.0 / ext)
    maxt = (1 - scale * ext).clamp(min=0)
    t = torch.stack([_u(g, -1, 1, B, device=dev), _u(g, -1, 1, B, device=dev)], 1) * maxt[:, None]
    RT = torch.stack([torch.stack([c, s], -1), torch.stack([-s, c], -1)], -2)
    A = RT / scale[:, None, None]
    b = -(RT @ t[:, :, None]) / scale[:, None, None]
    grid = F.affine_grid(torch.cat([A, b], 2), (B, 1, SIZE, SIZE), align_corners=False)
    img = F.grid_sample(clocks, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    mask = (F.grid_sample(torch.ones_like(clocks), grid, mode="bilinear", padding_mode="zeros",
                          align_corners=False) > 0.5).float()
    return img, mask


def noise(B, g, dev):
    amp = _u(g, 0, 0.25, B, 1, 1, 1, device=dev)
    n = torch.randn(B, 1, SIZE, SIZE, generator=g, device=dev).abs() * amp
    sp = (torch.rand(B, 1, SIZE, SIZE, generator=g, device=dev) > 0.998).float() * _u(g, 0.3, 1, B, 1, 1, 1, device=dev)
    return n + sp


def lines(B, g, dev, lined):
    xx, yy = _coords(dev)
    phi = _u(g, 0, math.pi, B, device=dev)
    proj = xx[None] * phi.cos()[:, None, None] + yy[None] * phi.sin()[:, None, None]
    rho, w, d = (_u(g, a, b, B, 1, 1, device=dev) for a, b in [(-1, 1), (0.002, 0.006), (0.03, 0.2)])
    mode = torch.randint(0, 3, (B,), generator=g, device=dev)[:, None, None]
    single = (proj - rho).abs() < w
    par = (proj - rho - torch.round((proj - rho) / d) * d).abs() < w
    img = (((mode == 1) & single) | ((mode == 2) & par)).float() * _u(g, 0.3, 1, B, 1, 1, device=dev)
    k2 = torch.round(yy[None] / 0.045)
    paper = ((yy[None] - k2 * 0.045).abs() < 0.003).float().expand(B, -1, -1) * 0.4
    return torch.where(lined[:, None, None], paper, img)[:, None]


def writing(B, g, dev, strokes=6, pts=40):
    start = _u(g, -0.9, 0.9, B, strokes, 1, 2, device=dev)
    th = _u(g, -math.pi, math.pi, B, strokes, 1, device=dev) + (
        torch.randn(B, strokes, pts, generator=g, device=dev) * 0.5).cumsum(-1)
    step = _u(g, 0.01, 0.03, B, strokes, 1, device=dev)
    path = start + (torch.stack([th.cos(), th.sin()], -1) * step[..., None]).cumsum(2)
    active = (torch.rand(B, strokes, 1, generator=g, device=dev) < 0.5).float().expand(-1, -1, pts)
    val = active * (path.abs() < 1).all(-1).float()
    px = ((path + 1) / 2 * SIZE).long().clamp(0, SIZE - 1)
    idx = px[..., 1] * SIZE + px[..., 0]
    canvas = torch.zeros(B, SIZE * SIZE, device=dev)
    canvas.scatter_reduce_(1, idx.reshape(B, -1), val.reshape(B, -1), reduce="amax", include_self=True)
    return F.max_pool2d(canvas.view(B, 1, SIZE, SIZE), 5, 1, 2)


def rects(B, g, dev, n=3):
    xx, yy = _coords(dev)
    cx, cy = (_u(g, -0.9, 0.9, B, n, 1, 1, device=dev) for _ in range(2))
    hw, hh = (_u(g, 0.03, 0.4, B, n, 1, 1, device=dev) for _ in range(2))
    active = torch.rand(B, n, 1, 1, generator=g, device=dev) < 0.5
    filled = torch.rand(B, n, 1, 1, generator=g, device=dev) < 0.5
    dx, dy = (xx[None, None] - cx).abs(), (yy[None, None] - cy).abs()
    inside = (dx < hw) & (dy < hh)
    border = inside & ~((dx < hw - 0.01) & (dy < hh - 0.01))
    return (torch.where(filled, inside, border) & active).any(1, keepdim=True).float()


@torch.no_grad()
def make_batch(bank, scores, n, g):
    """bank: uint8 (M,512,512) on the device; scores: float (M,). Returns x (n,1,512,512), y (n,2,64,64)."""
    dev = bank.device
    idx = torch.randint(0, bank.shape[0], (n,), generator=g, device=dev)
    img, mask = place(bank[idx].float().div(255).unsqueeze(1), g)
    lined = torch.zeros(n, dtype=torch.bool, device=dev)
    lined[: max(1, n // 25)] = True                        # lined paper in 2 of 50
    canvas = torch.maximum(img + noise(n, g, dev), lines(n, g, dev, lined))
    free = 1 - F.max_pool2d(mask, 21, 1, 10)               # writing / rectangles stay off the clock
    canvas = torch.maximum(canvas, torch.maximum(writing(n, g, dev), rects(n, g, dev)) * free)
    area = F.avg_pool2d(mask, 8)
    y = torch.cat([area, area * (scores[idx] / 5)[:, None, None, None]], 1)
    return canvas.clamp(0, 1), y
