import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


class Encoder(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()
        backbone = models.resnet50(weights=None) if backbone is None else backbone
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = 2048

    @classmethod
    def from_resnet(cls, backbone):
        return cls(backbone=backbone)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return torch.flatten(self.avgpool(x), 1)


class Classifier(nn.Module):
    def __init__(self, encoder: nn.Module, n_classes: int = 3):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(2048, n_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


class InputPrep(nn.Module):
    def __init__(self, mode: str = "none"):
        super().__init__()
        if mode not in {"none", "imagenet"}:
            raise ValueError(f"Unsupported input preprocessing mode: {mode}")
        self.mode = mode
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"InputPrep expects [N,H,W] or [N,1,H,W], got {tuple(x.shape)}")
        x = x.to(dtype=torch.float32)
        if not torch.isfinite(x).all():
            raise ValueError("InputPrep received non-finite pixels")
        if x.numel() and (x.min() < 0 or x.max() > 1):
            raise ValueError("InputPrep expects pixels in [0, 1]")
        x = x.expand(-1, 3, -1, -1)
        if self.mode == "imagenet":
            x = (x - self.mean) / self.std
        return x


class Projector(nn.Module):
    def __init__(self, dimension: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.BatchNorm1d(dimension),
            nn.ReLU(inplace=True),
            nn.Linear(dimension, dimension),
            nn.BatchNorm1d(dimension),
            nn.ReLU(inplace=True),
            nn.Linear(dimension, dimension),
            nn.BatchNorm1d(dimension, affine=False),
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, dimension: int = 2048, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dimension),
        )

    def forward(self, x):
        return self.net(x)


def simsiam_loss(p1, z1, p2, z2):
    def negative_cosine(p, z):
        return -(F.normalize(p, dim=1) * F.normalize(z.detach(), dim=1)).sum(dim=1).mean()

    return 0.5 * negative_cosine(p1, z2) + 0.5 * negative_cosine(p2, z1)


def set_trainable(encoder: nn.Module, parts: set[str] | frozenset[str]):
    parts = set(parts)
    unknown = parts - {"input_conv", "layer4", "all"}
    if unknown:
        raise ValueError(f"Unknown encoder parts: {sorted(unknown)}")
    if "all" in parts and len(parts) != 1:
        raise ValueError("'all' cannot be combined with other trainable parts")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    if "all" in parts:
        for parameter in encoder.parameters():
            parameter.requires_grad_(True)
    else:
        if "input_conv" in parts:
            for module in (encoder.conv1, encoder.bn1):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        if "layer4" in parts:
            for parameter in encoder.layer4.parameters():
                parameter.requires_grad_(True)
    return trainable_parameter_report(encoder)


def freeze_bn_stats(encoder: nn.Module):
    for module in encoder.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def trainable_parameter_report(encoder: nn.Module) -> dict:
    trainable = [name for name, parameter in encoder.named_parameters() if parameter.requires_grad]
    frozen = [name for name, parameter in encoder.named_parameters() if not parameter.requires_grad]
    return {
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad
        ),
        "frozen_parameter_count": sum(
            parameter.numel() for parameter in encoder.parameters() if not parameter.requires_grad
        ),
        "trainable_parameter_names": trainable,
        "frozen_parameter_names": frozen,
        "trainable_bn_affine": [
            name
            for name, parameter in encoder.named_parameters()
            if parameter.requires_grad and (name.endswith(".weight") or name.endswith(".bias"))
            and "bn" in name
        ],
    }
