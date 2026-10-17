"""Clock ViT (PLAN B3 / H7). Input (B,1,224,224) in [0,1] -> sigmoid (clock score / 5, dementia probability).

Paper: ViT-L/16 SWAG at 512 px. [OURS] ViT-B/16 (torchvision ImageNet-1k weights) at 224 px so training fits free Colab GPU time
without Drive; position embeddings therefore stay at their native size. Head: 3 x FC(->1024) + LeakyReLU(0.2), FC(1024->2).
Grey->3 channels and ImageNet normalisation are inside the model, so the ONNX takes the raw single-channel image.
"""
import torch
import torch.nn as nn


class ClockViT(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import ViT_B_16_Weights, vit_b_16
        v = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        v.heads = nn.Identity()
        self.body = v
        act = lambda: nn.LeakyReLU(0.2)
        self.head = nn.Sequential(nn.Linear(768, 1024), act(), nn.Linear(1024, 1024), act(),
                                  nn.Linear(1024, 1024), act(), nn.Linear(1024, 2))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        return torch.sigmoid(self.head(self.body(x)))
