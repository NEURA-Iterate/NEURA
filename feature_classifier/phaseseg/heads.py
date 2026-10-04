from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int = 384, n_classes: int = 4) -> None:
        super().__init__()
        self.classifier = nn.Conv2d(in_dim, n_classes, kernel_size=1)

    def forward(self, features: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(features)
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)


class FusionHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 384,
        n_classes: int = 4,
        img_ch: int = 3,
        width: int = 32,
    ) -> None:
        super().__init__()
        self.feature_projection = nn.Conv2d(in_dim, 64, kernel_size=1)
        self.image_branch = nn.Sequential(
            nn.Conv2d(img_ch, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(64 + width, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, n_classes, kernel_size=1),
        )

    def forward(self, features: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        projected = self.feature_projection(features)
        projected = F.interpolate(
            projected,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        image_features = self.image_branch(image)
        return self.classifier(torch.cat((projected, image_features), dim=1))


def build_head(name: str, in_dim: int = 384, n_classes: int = 4) -> nn.Module:
    if name == "linear":
        return LinearProbe(in_dim=in_dim, n_classes=n_classes)
    if name == "fusion":
        return FusionHead(in_dim=in_dim, n_classes=n_classes)
    raise ValueError(f"Unknown segmentation head {name!r}; expected 'linear' or 'fusion'")
