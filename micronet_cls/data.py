from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


def detector_indices(manifest: pd.DataFrame, detectors=None) -> np.ndarray:
    if detectors is None:
        return np.arange(len(manifest), dtype=np.int64)
    selected = tuple(dict.fromkeys(detectors))
    valid = {"BSE", "ETD", "Inlens"}
    unknown = set(selected) - valid
    if unknown:
        raise ValueError(f"Unknown detectors: {sorted(unknown)}")
    if not selected:
        raise ValueError("detectors must contain at least one detector")
    return np.flatnonzero(manifest["detector"].isin(selected).to_numpy())


class CropDataset(Dataset):
    def __init__(
        self,
        npy_path,
        manifest_csv,
        indices,
        transform=None,
        load_into_memory: bool = False,
    ):
        self.npy_path = Path(npy_path)
        self.manifest = (
            manifest_csv.copy()
            if isinstance(manifest_csv, pd.DataFrame)
            else pd.read_csv(manifest_csv)
        )
        mmap_mode = None if load_into_memory else "r"
        self.crops = np.load(self.npy_path, mmap_mode=mmap_mode)
        if len(self.crops) != len(self.manifest):
            raise ValueError(
                f"Crop cache has {len(self.crops)} rows but manifest has {len(self.manifest)}"
            )
        self.indices = np.asarray(indices, dtype=np.int64)
        if self.indices.size and (
            self.indices.min() < 0 or self.indices.max() >= len(self.manifest)
        ):
            raise IndexError("CropDataset indices are outside the manifest")
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        row_index = int(self.indices[item])
        crop = np.array(self.crops[row_index], dtype=np.float32, copy=True)
        tensor = torch.from_numpy(crop).unsqueeze(0)
        if self.transform is not None:
            tensor = self.transform(tensor)
        class_id = int(self.manifest.iloc[row_index]["class_id"])
        return tensor, class_id, row_index


class SpatialAug:
    def __init__(
        self,
        flips: bool = True,
        rot90: bool = False,
        rrc_scale: tuple[float, float] | None = None,
        brightness_contrast: tuple[float, float] | None = None,
    ):
        self.flips = flips
        self.rot90 = rot90
        self.rrc_scale = rrc_scale
        self.brightness_contrast = brightness_contrast

    @staticmethod
    def _rand(generator=None):
        return float(torch.rand((), generator=generator).item())

    def __call__(self, crop: torch.Tensor, generator=None):
        if crop.ndim == 2:
            crop = crop.unsqueeze(0)
        if crop.ndim != 3 or crop.shape[0] != 1:
            raise ValueError(f"SpatialAug expects [1,H,W], got {tuple(crop.shape)}")
        result = crop
        if self.flips:
            if self._rand(generator) < 0.5:
                result = torch.flip(result, dims=(-1,))
            if self._rand(generator) < 0.5:
                result = torch.flip(result, dims=(-2,))
        if self.rot90 and self._rand(generator) < 0.5:
            k = int(torch.randint(1, 4, (), generator=generator).item())
            result = torch.rot90(result, k=k, dims=(-2, -1))
        if self.rrc_scale is not None:
            low, high = self.rrc_scale
            if not 0 < low <= high <= 1:
                raise ValueError(f"Invalid rrc_scale: {self.rrc_scale}")
            scale = low + (high - low) * self._rand(generator)
            size = result.shape[-1]
            side = max(1, min(size, int(round(size * scale**0.5))))
            max_y = result.shape[-2] - side
            max_x = result.shape[-1] - side
            y = int(torch.randint(max_y + 1, (), generator=generator).item()) if max_y else 0
            x = int(torch.randint(max_x + 1, (), generator=generator).item()) if max_x else 0
            result = F.interpolate(
                result[:, y : y + side, x : x + side].unsqueeze(0),
                size=crop.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        if self.brightness_contrast is not None:
            brightness, contrast = self.brightness_contrast
            brightness_delta = (2 * self._rand(generator) - 1) * brightness
            contrast_factor = 1 + (2 * self._rand(generator) - 1) * contrast
            mean = result.mean(dim=(-2, -1), keepdim=True)
            result = (result - mean) * contrast_factor + mean + brightness_delta
        return result.clamp_(0, 1)


class TwoViewDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: SpatialAug | None = None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        crop, class_id, row_index = self.dataset[index]
        if self.transform is None:
            return crop, crop.clone(), class_id, row_index
        return (
            self.transform(crop.clone()),
            self.transform(crop.clone()),
            class_id,
            row_index,
        )


def balanced_sampler_weights(manifest: pd.DataFrame) -> torch.DoubleTensor:
    required = {"class_id", "detector", "source_image_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing sampler columns: {sorted(missing)}")
    if manifest.empty:
        return torch.empty(0, dtype=torch.double)
    class_count = manifest["class_id"].nunique()
    detector_counts = manifest.groupby("class_id")["detector"].nunique().to_dict()
    image_counts = (
        manifest.groupby(["class_id", "detector"])["source_image_id"].nunique().to_dict()
    )
    crop_counts = manifest.groupby(["class_id", "detector", "source_image_id"]).size().to_dict()
    weights = []
    for row in manifest.itertuples(index=False):
        key = (row.class_id, row.detector, row.source_image_id)
        weights.append(
            1.0
            / (
                class_count
                * detector_counts[row.class_id]
                * image_counts[(row.class_id, row.detector)]
                * crop_counts[key]
            )
        )
    return torch.as_tensor(weights, dtype=torch.double)
