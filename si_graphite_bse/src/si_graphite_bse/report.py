"""Per-batch aggregation, statistical comparison and Markdown report."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .uncertainty import KEY_KPIS

PRIMARY = {rank: cols[0] for rank, (_, cols) in KEY_KPIS.items()}
PRIORITY_KPIS = list(PRIMARY.values())
REPORT_KPIS = list(
    dict.fromkeys(
        [c for _, cols in KEY_KPIS.values() for c in cols]
        + [
            "frac_si",
            "frac_graphite",
            "frac_cbd",
            "si_ecd_d10",
            "si_count_per_mpx",
            "si_count_per_mm2",
            "si_dispersion_index_w256",
            "si_dispersion_index_w1024",
            "si_cv_w512",
            "graphite_alignment_coherence",
        ]
    )
)

# Trust flags: an image's KPIs are shown but excluded from batch decisions if any flag is raised.
TRUST_FLAGS = {
    "low_si_contrast": lambda d: (d["qc_threshold_method"] != "valley") | (d["qc_si_peak_n"] < 1.8),
    "polishing_streaks": lambda d: d.get("qc_streak_severity", pd.Series(0, index=d.index)) > 0.15,
    "charging": lambda d: d.get("qc_charging_index", pd.Series(0, index=d.index)) > 0.05,
    "detector_disagreement": lambda d: d.get("qc_modal_agreement", pd.Series(1, index=d.index)) < 0.95,
}


def trust_flags(per_image: pd.DataFrame) -> pd.Series:
    flags = pd.Series("", index=per_image.index)
    for name, rule in TRUST_FLAGS.items():
        hit = rule(per_image).fillna(False).astype(bool)
        flags[hit] = flags[hit] + name + " "
    return flags.str.strip()


def _fmt_interval(row: pd.Series, k: str) -> str:
    def f(x) -> str:
        return f"{x:.3g}" if pd.notna(x) else "–"

    s = f(row.get(k))
    if f"{k}_pix_lo" in row:
        s += f" [pix {f(row[f'{k}_pix_lo'])}–{f(row[f'{k}_pix_hi'])}]"
    if f"{k}_alg_p05" in row:
        s += f" [alg {f(row[f'{k}_alg_p05'])}–{f(row[f'{k}_alg_p95'])}]"
        if isinstance(row.get(f"{k}_alg_driver"), str) and row[f"{k}_alg_driver"]:
            s += f" ({row[f'{k}_alg_driver'].split('.')[-1]})"
    return s


def key_kpi_table(per_image: pd.DataFrame) -> pd.DataFrame:
    """One row per KPI: batch medians plus median pixel / algorithm interval widths and main driver."""
    rows = []
    for rank, (name, cols) in KEY_KPIS.items():
        for k in cols:
            if k not in per_image or per_image[k].isna().all():
                continue
            row = {"rank": rank, "kpi": name, "column": k}
            for b, g in per_image.groupby("batch"):
                row[b] = f"{g[k].median():.3g}"
            if f"{k}_pix_lo" in per_image:
                row["pixel width (median)"] = (
                    f"{(per_image[f'{k}_pix_hi'] - per_image[f'{k}_pix_lo']).median():.2g}"
                )
            if f"{k}_alg_p05" in per_image:
                row["algorithm width (median)"] = (
                    f"{(per_image[f'{k}_alg_p95'] - per_image[f'{k}_alg_p05']).median():.2g}"
                )
                drivers = per_image[f"{k}_alg_driver"].dropna()
                row["main driver"] = drivers.mode().iloc[0] if len(drivers) else ""
            rows.append(row)
    return pd.DataFrame(rows)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.sign(a[:, None] - b[None, :])
    return float(diff.mean())


def batch_summary(per_image: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in REPORT_KPIS if c in per_image]
    agg = per_image.groupby("batch")[cols].agg(["mean", "std", "count"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg


def batch_stats(per_image: pd.DataFrame) -> pd.DataFrame:
    rows = []
    batches = sorted(per_image["batch"].unique())
    for kpi in [c for c in REPORT_KPIS if c in per_image]:
        groups = {b: per_image.loc[per_image.batch == b, kpi].dropna().to_numpy() for b in batches}
        valid = [g for g in groups.values() if len(g) >= 2]
        if len(valid) >= 2:
            # scipy raises on all-identical values; identical groups mean no difference.
            h, p = stats.kruskal(*valid) if np.ptp(np.concatenate(valid)) > 0 else (0.0, 1.0)
            rows.append({"kpi": kpi, "comparison": "all (Kruskal-Wallis)", "statistic": h, "p_value": p})
        for a, b in combinations(batches, 2):
            ga, gb = groups[a], groups[b]
            if len(ga) >= 2 and len(gb) >= 2:
                u, p = stats.mannwhitneyu(ga, gb, alternative="two-sided")
                rows.append(
                    {
                        "kpi": kpi,
                        "comparison": f"{a} vs {b} (Mann-Whitney)",
                        "statistic": u,
                        "p_value": p,
                        "cliffs_delta": cliffs_delta(ga, gb),
                    }
                )
    return pd.DataFrame(rows)


def _boxplots(per_image: pd.DataFrame, kpis: list[str], path: Path) -> None:
    kpis = [k for k in kpis if k in per_image]
    batches = sorted(per_image["batch"].unique())
    fig, axes = plt.subplots(1, len(kpis), figsize=(4 * len(kpis), 3.8), squeeze=False)
    for ax, kpi in zip(axes[0], kpis, strict=False):
        data = [per_image.loc[per_image.batch == b, kpi].dropna() for b in batches]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(batches) + 1), batches)
        for i, d in enumerate(data, start=1):
            ax.scatter(
                np.full(len(d), i) + np.random.default_rng(0).uniform(-0.12, 0.12, len(d)), d, s=14, c="k"
            )
        ax.set_title(kpi, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _psd_plot(particles: pd.DataFrame, unit: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for batch, g in particles.groupby("batch"):
        ecd = np.sort(g["ecd"].to_numpy())
        ax.plot(ecd, np.linspace(0, 1, len(ecd)), label=f"{batch} (n={len(ecd)})")
    ax.set_xscale("log")
    ax.set_xlabel(f"Si equivalent circular diameter [{unit}]")
    ax.set_ylabel("cumulative number fraction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _profile_plot(profiles: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    band_cols = [c for c in profiles if c.startswith("band_")]
    x = np.linspace(0, 1, len(band_cols))
    for batch, g in profiles.groupby("batch"):
        m, s = g[band_cols].mean().to_numpy(), g[band_cols].std().to_numpy()
        ax.plot(x, m, label=batch)
        ax.fill_between(x, m - s, m + s, alpha=0.2)
    ax.set_xlabel("relative image row position (top = 0)")
    ax.set_ylabel("Si fraction of solids")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.4g}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(out_dir: Path) -> Path:
    kdir = out_dir / "kpis"
    per_image = pd.read_csv(kdir / "per_image.csv")
    particles = pd.read_csv(kdir / "si_particles.csv")
    profiles = pd.read_csv(kdir / "si_profiles.csv")
    unit = str(per_image["size_unit"].iloc[0])
    per_image["trust_flags"] = trust_flags(per_image)
    per_image.to_csv(kdir / "per_image.csv", index=False)
    trusted = per_image[per_image["trust_flags"] == ""]

    summary = batch_summary(trusted)
    summary.to_csv(kdir / "batch_summary.csv")
    st = batch_stats(trusted)
    st.to_csv(kdir / "batch_stats.csv", index=False)
    key_table = key_kpi_table(per_image)
    key_table.to_csv(kdir / "key_kpis.csv", index=False)

    rdir = out_dir / "report"
    rdir.mkdir(parents=True, exist_ok=True)
    _boxplots(trusted, PRIORITY_KPIS[:5], rdir / "priority_kpis.png")
    _boxplots(trusted, PRIORITY_KPIS[5:], rdir / "secondary_kpis.png")
    _psd_plot(particles[~particles["touches_border"]], unit, rdir / "si_psd.png")
    _profile_plot(profiles, rdir / "si_profile.png")

    if st.empty:
        st = pd.DataFrame(columns=["kpi", "comparison", "statistic", "p_value"])
    kw = st[st["comparison"].str.startswith("all")][["kpi", "statistic", "p_value"]]
    kw = kw[kw["kpi"].isin([c for _, cols in KEY_KPIS.values() for c in cols])]
    robust_path = kdir / "batch_robustness.csv"
    robust = pd.read_csv(robust_path) if robust_path.exists() else None
    flagged = per_image[per_image["trust_flags"] != ""][["batch", "image_id", "trust_flags"]]
    examples = per_image[["batch", "image_id"]].copy()
    for rank in (1, 2, 3):
        examples[PRIMARY[rank]] = [_fmt_interval(r, PRIMARY[rank]) for _, r in per_image.iterrows()]

    md = [
        "# Si / graphite anode QC report (BSE + ETD + Inlens)",
        "",
        f"Images: {len(per_image)} ({len(trusted)} without trust flags; flagged images are excluded from batch "
        f"statistics). Size unit: **{unit}**"
        + (" (pixel size unknown: sizes are in pixels)" if unit == "px" else "")
        + ".",
        "",
        "## Key QC KPIs (batch medians, all images)",
        "",
        "Ranked by importance for QC decisions. *Pixel width*: KPI range allowed by noise and edge blur. "
        "*Algorithm width*: 5th-95th percentile over reruns with plausible settings. *Main driver*: the setting "
        "that moves the KPI most. See `docs/KPI_SPEC.md` for definitions.",
        "",
        _md_table(key_table),
        "",
        "![priority KPIs](priority_kpis.png)",
        "",
        "![secondary KPIs](secondary_kpis.png)",
        "",
        "## Is a batch difference robust to algorithm choices?",
        "",
        "Share of algorithm runs in which the Kruskal-Wallis test across batches gives p < 0.05. Robust only if "
        ">= 95 % of runs agree.",
        "",
        _md_table(robust)
        if robust is not None and len(robust)
        else "Algorithm Monte Carlo not run (`--mc-runs`).",
        "",
        "## Batch differences with the default settings (Kruskal-Wallis, trusted images)",
        "",
        _md_table(kw),
        "",
        "Pairwise Mann-Whitney tests with Cliff's delta are in `kpis/batch_stats.csv`. "
        "With 7-17 images per batch, treat p-values as indicative only.",
        "",
        "## Per-image values with intervals (top 3 KPIs)",
        "",
        "`value [pix lo–hi] [alg p5–p95] (main driver)`",
        "",
        _md_table(examples),
        "",
        "## Si particle size distribution (particles touching the image border excluded)",
        "",
        "![Si PSD](si_psd.png)",
        "",
        "## Si fraction vs image row (through-thickness only if rows run through the electrode)",
        "",
        "![Si profile](si_profile.png)",
        "",
        "## Trust flags",
        "",
        _md_table(flagged) if len(flagged) else "None.",
        "",
        "## Caveats",
        "",
        "- `si_wt_pct_estimate` assumes pure Si (2.33 g/cm3) vs graphite + binder (2.26 g/cm3); invalid for SiOx/Si-C.",
        "- Sizes are 2D, number-weighted equivalent circular diameters; not comparable with laser-diffraction D50/D90.",
        "- Si grey level is a relative change flag, only comparable at identical microscope settings; it does not "
        "identify SiOx without calibration / EDS.",
        "- Binder/carbon-black is a texture-based class (experimental): rough graphite surfaces can still be "
        "counted and smooth binder missed.",
        "- Pores recessed below the surface but not black in ETD are undercounted.",
        "- Gradients assume image rows run through the electrode thickness, which is not confirmed.",
        "- Not quantified: field-of-view sampling, physical-model assumptions, bias against ground truth.",
        "",
    ]
    path = rdir / "report.md"
    path.write_text("\n".join(md))
    return path
