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
    si_low_offset: float = 0.0
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
class MultiModalConfig:
    """ETD / Inlens settings (used only when the sibling images exist)."""

    enabled: bool = True
    etd_pore_offset: float = 0.0
    etd_pore_fallback: float = 0.45
    gap_max_width_px: int = 4
    gap_min_depth: float = 0.35
    gap_sigma_px: float = 0.7
    gap_max_level: float = 0.6
    gap_min_area_px: int = 10
    si_max_internal_porosity: float = 0.15
    cbd_texture_window_px: int = 15
    cbd_texture_factor: float = 3.0
    cbd_bse_texture_factor: float = 3.0
    cbd_edge_band_px: float = 12.0
    cbd_big_pore_px: int = 500
    cbd_long_gap_px: int = 60
    cbd_open_radius: int = 4
    cbd_min_area_px: int = 300
    graphite_interior_px: float = 10.0
    crack_k: float = 6.0
    saturation_level: int = 250


@dataclass
class KPIConfig:
    pixel_size_um: float | None = None
    window_sizes: list[int] = field(default_factory=lambda: [256, 512, 1024])
    min_window_solid_fraction: float = 0.5
    cluster_gap_px: int = 5
    contact_ring_px: tuple[float, float] = (2.0, 5.0)
    profile_bands: int = 20
    graphite_min_area_px: int = 200
    alignment_sigma_px: float = 4.0
    n_shuffles: int = 100
    pore_min_area_px: int = 10
    density_si: float = 2.33
    density_graphite: float = 2.26


@dataclass
class UncertaintyConfig:
    pixel: bool = True
    p_confident: float = 0.95
    p_inclusive: float = 0.05
    edge_profile_px: int = 8
    mc_runs: int = 30
    mc_seed: int = 0


@dataclass
class Config:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    multimodal: MultiModalConfig = field(default_factory=MultiModalConfig)
    kpis: KPIConfig = field(default_factory=KPIConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)

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
