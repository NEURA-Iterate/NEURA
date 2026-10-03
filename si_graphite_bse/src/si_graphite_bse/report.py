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

PRIORITY_KPIS = ["si_fraction_of_solids", "si_ecd_d50", "si_cv_w512", "si_grey_median_n"]
REPORT_KPIS = PRIORITY_KPIS + [
    "frac_si",
    "frac_graphite",
    "frac_pore",
    "si_wt_pct_estimate",
    "si_ecd_d10",
    "si_ecd_d90",
    "si_count_per_mpx",
    "si_count_per_mm2",
    "si_nn_distance_median",
    "si_fraction_particles_clustered",
    "si_contact_graphite",
    "si_contact_pore",
    "si_cv_w256",
    "si_cv_w1024",
    "si_profile_rel_slope",
    "graphite_ecd_d50",
    "graphite_aspect_ratio_median",
    "graphite_orientation_order",
]


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

    summary = batch_summary(per_image)
    summary.to_csv(kdir / "batch_summary.csv")
    st = batch_stats(per_image)
    st.to_csv(kdir / "batch_stats.csv", index=False)

    rdir = out_dir / "report"
    rdir.mkdir(parents=True, exist_ok=True)
    _boxplots(per_image, PRIORITY_KPIS, rdir / "priority_kpis.png")
    _boxplots(
        per_image, ["frac_si", "frac_graphite", "frac_pore", "si_contact_pore"], rdir / "area_fractions.png"
    )
    _psd_plot(particles[~particles["touches_border"]], unit, rdir / "si_psd.png")
    _profile_plot(profiles, rdir / "si_profile.png")

    key = [
        k
        for k in PRIORITY_KPIS + ["frac_si", "frac_graphite", "frac_pore", "si_wt_pct_estimate"]
        if k in per_image
    ]
    mean_sd = pd.DataFrame(
        {
            "batch": summary.index,
            "n_images": summary[f"{key[0]}_count"].astype(int).to_numpy(),
            **{
                k: [
                    f"{m:.4g} ± {s:.2g}"
                    for m, s in zip(summary[f"{k}_mean"], summary[f"{k}_std"], strict=False)
                ]
                for k in key
            },
        }
    )
    kw = st[st["comparison"].str.startswith("all")][["kpi", "statistic", "p_value"]]
    flagged = per_image[(per_image["qc_threshold_method"] != "valley") | (per_image["qc_si_peak_n"] < 1.8)][
        ["batch", "image_id", "qc_threshold_method", "qc_si_peak_n", "si_fraction_of_solids"]
    ]

    md = [
        "# Si / graphite BSE segmentation report",
        "",
        f"Images: {len(per_image)}. Size unit: **{unit}**"
        + (" (pixel size unknown: sizes are in pixels)" if unit == "px" else "")
        + ".",
        "",
        "## Priority KPIs per batch (mean ± sd over images)",
        "",
        _md_table(mean_sd),
        "",
        "![priority KPIs](priority_kpis.png)",
        "",
        "## Batch differences (Kruskal-Wallis across batches)",
        "",
        _md_table(kw),
        "",
        "Pairwise Mann-Whitney tests with Cliff's delta are in `kpis/batch_stats.csv`. "
        "With 7-17 images per batch, treat p-values as indicative only.",
        "",
        "## Area fractions",
        "",
        "![area fractions](area_fractions.png)",
        "",
        "## Si particle size distribution (particles touching the image border excluded)",
        "",
        "![Si PSD](si_psd.png)",
        "",
        "## Si fraction vs image row (through-thickness, if rows run through the electrode)",
        "",
        "![Si profile](si_profile.png)",
        "",
        "## QC flags",
        "",
        "Images whose Si threshold did not come from a clear graphite/Si histogram valley, or whose Si peak "
        "is unusually close to graphite (normalised < 1.8, i.e. low Si/graphite contrast; more risk of edge "
        "artefacts being counted as Si):",
        "",
        _md_table(flagged) if len(flagged) else "None.",
        "",
        "## Caveats",
        "",
        "- Binder and carbon black cannot be separated from graphite or pore in BSE; they are not measured.",
        "- Porosity is the least reliable class: pores often show sub-surface material at graphite-like grey.",
        "- `si_wt_pct_estimate` assumes pure Si (2.33 g/cm3) vs graphite (2.26 g/cm3); invalid for SiOx/Si-C.",
        "- Nano-Si would show as a bright haze rather than particles: see `qc_bright_unassigned_fraction`.",
        "",
    ]
    path = rdir / "report.md"
    path.write_text("\n".join(md))
    return path
