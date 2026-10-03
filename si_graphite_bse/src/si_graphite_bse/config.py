"""Pipeline configuration (all tunable parameters in one place)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class PreprocessConfig:
    border_px: int = 2
    denoise_sigma: float = 1.5
    floor_percentile: float = 0.5
    hist_smooth_levels: float = 3.0


@dataclass
class SegmentConfig:
    hist_smooth_bins: float = 2.0
    si_k_min: float = 3.0
    si_k_low: float = 4.0
    si_k_high: float = 6.0
    si_peak_min_prominence: float = 0.3
    si_seed_fraction: float = 0.5
    pore_k: float = 4.0


@dataclass
class CleanupConfig:
    si_core_radius_px: float = 4.0
    si_min_seed_fraction: float = 0.3
    si_open_radius: int = 1
    si_min_area_px: int = 30
    si_max_hole_area_px: int = 200
    si_min_solidity: float = 0.7


@dataclass
class KPIConfig:
    pixel_size_um: float | None = None
    window_sizes: list[int] = field(default_factory=lambda: [256, 512, 1024])
    min_window_solid_fraction: float = 0.5
    cluster_gap_px: int = 5
    contact_ring_px: tuple[float, float] = (2.0, 5.0)
    profile_bands: int = 20
    graphite_min_area_px: int = 200
    density_si: float = 2.33
    density_graphite: float = 2.26


@dataclass
class Config:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    kpis: KPIConfig = field(default_factory=KPIConfig)

    @classmethod
    def from_yaml(cls, path: Path | None) -> Config:
        if path is None:
            return cls()
        raw = yaml.safe_load(Path(path).read_text()) or {}
        sections = {f.name: f.default_factory for f in fields(cls)}
        unknown = set(raw) - set(sections)
        if unknown:
            raise ValueError(f"Unknown config sections: {sorted(unknown)}")
        return cls(**{name: factory(**raw.get(name, {})) for name, factory in sections.items()})

    def to_dict(self) -> dict:
        return asdict(self)
