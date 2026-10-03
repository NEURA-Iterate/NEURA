# Results: full 31-image run (BSE + ETD/SE + Inlens)

Produced by `python -m anode_qc.cli run --data <Hackathon-Polaron> --out <dir> --workers 4 --mc-runs 0` with the
settings in `config_used.yaml`. Images per batch: Batch_1 = 7, Batch_2 = 7, Batch_3 = 17.
The algorithm-uncertainty Monte Carlo was not run here, so only pixel-ambiguity intervals (`*_pix_lo` / `*_pix_hi`) are included.

| File | Content |
|---|---|
| `kpis/kpis_all_images.csv` | One row per image: the 10 key KPIs with pixel-uncertainty low/high, trust flags |
| `kpis/kpis_all_images_full.csv`, `kpis/per_image.csv` | Every computed column (same data) |
| `kpis/batch_medians.csv` | Median per batch + Kruskal–Wallis p-value for each key KPI |
| `kpis/batch_summary.csv`, `kpis/batch_stats.csv`, `kpis/key_kpis.csv` | Batch summaries and pairwise tests |
| `kpis/si_particles.csv` | One row per Si particle (size, shape, position) |
| `kpis/si_profiles.csv` | Si / pore / binder fraction per row band (top-to-bottom profiles) |
| `report/` | Markdown report and summary plots |
| `qc_examples/` | Threshold histograms for every image, plus 3 downscaled crops: BSE, new labels, ETD, BSE-only labels |

Main finding: Batch_3 differs from Batches 1 and 2 in porosity (14.3 % vs 11.5 / 12.4 %, p = 0.0002), graphite crack density
(67 vs 48 / 51 per Mpx, p = 0.0006) and Si contact (more graphite contact, less binder contact, p ≈ 0.02). No key KPI separates
Batch_1 from Batch_2. Sizes are in pixels (no pixel size in the TIFFs). Binder/CBD values are experimental (texture-based, not validated).
Full masks and overlays (~170 MB) are not committed; rerun the command above to regenerate them.
