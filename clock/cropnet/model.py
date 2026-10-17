"""Crop-Net (PLAN H4, read from Bone et al. Figure 1). Input 1x512x512 -> 2x64x64 (channel 0 = clock heat map)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

# (in, mid, out) channels per "convolution downsample" block; +2 channels (max-pool + bilinear image) are concatenated after each
DOWN = [(1, 25, 48), (50, 50, 98), (100, 100, 198), (200, 200, 398), (400, 400, 423), (425, 425, 448), (450, 450, 498)]


class CropNet(nn.Module):
    def __init__(self):
        super().__init__()
        act = lambda: nn.LeakyReLU(0.2)
        self.down = nn.ModuleList(
            nn.Sequential(nn.Conv2d(i, m, 3, 1, 1), act(), nn.Conv2d(m, o, 4, 2, 1), act()) for i, m, o in DOWN)
        self.mid = nn.Sequential(nn.Conv2d(500, 500, 3, 1, 1), act(), nn.Conv2d(500, 500, 3, 1, 1), act())
        ups, ch = [], [500, 250, 150, 100, 50]
        for a, b in zip(ch, ch[1:]):
            ups += [nn.ConvTranspose2d(a, b, 4, 2, 1), act()]
        self.up = nn.Sequential(*ups)
        self.out = nn.Conv2d(50, 2, 3, 1, 1)   # [OURS] 2 channels (area, area x score/5), PLAN H4

    def forward(self, x):
        img, h = x, x
        for blk in self.down:
            h = blk(h)
            s = h.shape[-1]
            h = torch.cat([h, F.max_pool2d(img, img.shape[-1] // s),
                           F.interpolate(img, size=(s, s), mode="bilinear", align_corners=False)], 1)
        return torch.sigmoid(self.out(self.up(self.mid(h))))
