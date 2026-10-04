from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _pad_to_patch(image: torch.Tensor, patch: int) -> torch.Tensor:
    height, width = image.shape[-2:]
    pad_height = (-height) % patch
    pad_width = (-width) % patch
    if pad_height or pad_width:
        image = F.pad(image, (0, pad_width, 0, pad_height), mode="replicate")
    return image


class Dinov2Backbone(nn.Module):
    """Frozen DINOv2 feature extractor returning patch tokens in BCHW order."""

    def __init__(
        self,
        model_id: str = "facebook/dinov2-small",
        upsample: int = 1,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        if upsample < 1:
            raise ValueError("upsample must be at least 1")
        from transformers import AutoModel

        self.model_id = model_id
        self.patch_size = 14
        self.upsample = upsample
        self.model = AutoModel.from_pretrained(model_id)
        self.dim = int(self.model.config.hidden_size)
        if self.dim != 384:
            raise ValueError(f"{model_id} has {self.dim} channels; DINOv2-small is expected to have 384")
        self.model.requires_grad_(False)
        self.model.eval()
        self.register_buffer(
            "mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        if device is not None:
            self.to(device)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected image shape (B,3,H,W), got {tuple(image.shape)}")
        image = image.to(device=self.mean.device, dtype=torch.float32)
        image = _pad_to_patch(image, self.patch_size)
        height, width = image.shape[-2:]
        image = (image - self.mean) / self.std
        with torch.no_grad():
            output = self.model(pixel_values=image, interpolate_pos_encoding=True)
        patch_count = (height // self.patch_size) * (width // self.patch_size)
        tokens = output.last_hidden_state[:, -patch_count:, :]
        features = tokens.transpose(1, 2).reshape(
            image.shape[0],
            self.dim,
            height // self.patch_size,
            width // self.patch_size,
        )
        if self.upsample != 1:
            features = F.interpolate(
                features,
                scale_factor=self.upsample,
                mode="bilinear",
                align_corners=False,
            )
        return features


class RandomBackbone(nn.Module):
    """Deterministic, frozen strided-convolution backbone for synthetic tests."""

    def __init__(self, dim: int = 384, patch: int = 14, upsample: int = 1) -> None:
        super().__init__()
        if dim < 1 or patch < 1 or upsample < 1:
            raise ValueError("dim, patch, and upsample must be positive")
        self.dim = dim
        self.patch_size = patch
        self.upsample = upsample
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.projection = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.requires_grad_(False)
        self.eval()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected image shape (B,3,H,W), got {tuple(image.shape)}")
        image = _pad_to_patch(image, self.patch_size)
        with torch.no_grad():
            features = self.projection(image)
        if self.upsample != 1:
            features = F.interpolate(
                features,
                scale_factor=self.upsample,
                mode="bilinear",
                align_corners=False,
            )
        return features
