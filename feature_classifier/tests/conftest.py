from __future__ import annotations

import numpy as np

from phaseseg import CLASSES
from phaseseg.train import ImageSample


def make_sem_image(seed: int, height: int = 112, width: int = 112) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.full((height, width), CLASSES.index("graphite"), dtype=np.uint8)
    yy, xx = np.ogrid[:height, :width]
    for cy, cx in ((42, 42), (70, 70)):
        labels[(yy - cy) ** 2 + (xx - cx) ** 2 <= 9**2] = CLASSES.index("si")
    for cy, cx in ((42, 70), (70, 42)):
        labels[(yy - cy) ** 2 + (xx - cx) ** 2 <= 9**2] = CLASSES.index("pore")
    labels[52:60, 30:38] = CLASSES.index("binder")
    phase_values = np.array(
        [
            [0.06, 0.08, 0.10],
            [0.52, 0.50, 0.49],
            [0.96, 0.83, 0.91],
            [0.30, 0.22, 0.18],
        ],
        dtype=np.float32,
    )
    image = phase_values[labels].transpose(2, 0, 1)
    image += rng.normal(0, 0.015, size=image.shape).astype(np.float32)
    return np.clip(image, 0, 1), labels


def make_image_sample(seed: int, image_id: str = "sample", batch: str = "Batch_1") -> ImageSample:
    import torch

    from phaseseg.backbone import RandomBackbone

    image, labels = make_sem_image(seed)
    backbone = RandomBackbone(dim=16, patch=14)
    with torch.inference_mode():
        features = backbone(torch.from_numpy(image).unsqueeze(0))[0].permute(1, 2, 0).numpy()
    return ImageSample(image_id, batch, image, features, labels)
