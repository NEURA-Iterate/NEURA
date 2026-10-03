"""Algorithm-choice uncertainty: rerun the pipeline with settings drawn within plausible ranges.

Root cause addressed: thresholds and cleanup rules with no single correct value. The same Latin-hypercube
draws are used for every image, so each run is one consistent alternative pipeline; batch comparisons are
repeated per run to test whether a conclusion survives every plausible setting.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import qmc

from .config import Config

# (section, field, low, high, is_int)
PARAM_RANGES: list[tuple[str, str, float, float, bool]] = [
    ("preprocess", "denoise_sigma", 1.0, 2.0, False),
    ("segment", "si_low_offset", -0.05, 0.05, False),
    ("segment", "si_seed_fraction", 0.3, 0.7, False),
    ("cleanup", "si_core_radius_px", 3.0, 5.0, False),
    ("cleanup", "si_min_seed_fraction", 0.2, 0.4, False),
    ("cleanup", "si_open_radius", 0, 2, True),
    ("cleanup", "si_min_area_px", 20, 50, True),
    ("cleanup", "si_min_solidity", 0.6, 0.8, False),
    ("multimodal", "si_max_internal_porosity", 0.10, 0.25, False),
    ("multimodal", "etd_pore_offset", -0.05, 0.05, False),
    ("multimodal", "gap_max_width_px", 2, 6, True),
    ("multimodal", "gap_min_depth", 0.25, 0.45, False),
    ("multimodal", "crack_k", 4.0, 8.0, False),
    ("multimodal", "cbd_texture_factor", 2.4, 3.6, False),
    ("multimodal", "cbd_bse_texture_factor", 2.4, 3.6, False),
    ("multimodal", "cbd_edge_band_px", 8.0, 16.0, False),
    ("kpis", "alignment_sigma_px", 2.0, 6.0, False),
    ("kpis", "contact_ring_inner", 1.0, 3.0, False),
    ("kpis", "contact_ring_outer", 4.0, 7.0, False),
]


def param_name(section: str, field: str) -> str:
    return f"{section}.{field}"


def draw_settings(n_runs: int, seed: int) -> list[dict[str, float]]:
    lhs = qmc.LatinHypercube(d=len(PARAM_RANGES), seed=seed).random(n_runs)
    out = []
    for row in lhs:
        s = {}
        for u, (sec, f, lo, hi, is_int) in zip(row, PARAM_RANGES, strict=True):
            v = lo + u * (hi - lo)
            s[param_name(sec, f)] = int(np.floor(lo + u * (hi - lo + 1))) if is_int else float(v)
        out.append(s)
    return out


def apply_settings(cfg: Config, setting: dict[str, float]) -> Config:
    c = copy.deepcopy(cfg)
    ring = list(c.kpis.contact_ring_px)
    for key, v in setting.items():
        sec, f = key.split(".")
        if f == "contact_ring_inner":
            ring[0] = v
        elif f == "contact_ring_outer":
            ring[1] = v
        else:
            setattr(getattr(c, sec), f, v)
    c.kpis.contact_ring_px = (ring[0], ring[1])
    return c


def summarise(runs: pd.DataFrame, kpis: list[str]) -> pd.DataFrame:
    """Per image: p5 / p95 of each KPI over runs and the setting with the largest |Spearman rho|."""
    params = [param_name(s, f) for s, f, *_ in PARAM_RANGES]
    rows = []
    for (batch, image_id), g in runs.groupby(["batch", "image_id"]):
        row = {"batch": batch, "image_id": image_id}
        for k in kpis:
            if k not in g:
                continue
            v = g[k].astype(float)
            row[f"{k}_alg_p05"], row[f"{k}_alg_p95"] = (
                np.nanpercentile(v, [5, 95]) if v.notna().any() else (np.nan, np.nan)
            )
            best, best_rho = "", 0.0
            for p in params:
                ok = v.notna() & g[p].notna()
                if ok.sum() < 5 or g[p][ok].nunique() < 2 or v[ok].nunique() < 2:
                    continue
                rho = abs(stats.spearmanr(g[p][ok], v[ok])[0])
                if np.isfinite(rho) and rho > best_rho:
                    best, best_rho = p, rho
            row[f"{k}_alg_driver"] = best
        rows.append(row)
    return pd.DataFrame(rows)


def batch_robustness(runs: pd.DataFrame, kpis: list[str], alpha: float = 0.05) -> pd.DataFrame:
    """Share of runs in which the Kruskal-Wallis batch difference is significant, per KPI."""
    rows = []
    for k in kpis:
        if k not in runs:
            continue
        sig = []
        for _, g in runs.groupby("run"):
            groups = [x[k].dropna().to_numpy(float) for _, x in g.groupby("batch")]
            groups = [x for x in groups if len(x) >= 2]
            if len(groups) < 2 or np.ptp(np.concatenate(groups)) == 0:
                continue
            sig.append(stats.kruskal(*groups)[1] < alpha)
        if sig:
            share = float(np.mean(sig))
            verdict = (
                "robust difference"
                if share >= 0.95
                else "robust no-difference"
                if share <= 0.05
                else "setting-dependent"
            )
            rows.append({"kpi": k, "runs": len(sig), "share_significant": share, "verdict": verdict})
    return pd.DataFrame(rows)
