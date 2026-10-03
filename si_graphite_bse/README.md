# si_graphite_bse: silicon / graphite detection on BSE images

Labels every pixel of a backscattered-electron (BSE) SEM image as **pore**, **graphite** or **silicon**, then
computes KPIs per image and per batch. In BSE, brightness follows atomic number, so Si (Z=14) appears bright,
graphite (Z=6) mid-grey and pore dark.

## Quick start

```bash
cd si_graphite_bse
pip install -e ".[dev]"
export HF_TOKEN=...              # read access to gabrielgramicelli/NEURA-iterate-hack (private)
sgb download --dest data         # BSE images only (~0.6 GB)
sgb run --data data --out outputs
# outputs/report/report.md is the summary
```

`sgb run --limit 2` processes only the first 2 images; `--config my.yaml` overrides any value in
[`configs/default.yaml`](configs/default.yaml) (unspecified values keep their defaults). Set
`kpis.pixel_size_um` once the pixel size is known: all sizes switch from px to µm and counts to per-mm².

## Algorithm

1. **Load** (`data.py`): channel 0 of the TIFF (grey stored as 3 identical channels); a 2 px border is cropped
   because some files have corrupted outer columns.
2. **Denoise** (`preprocess.py`): Gaussian σ = 1.5 px. The raw histograms are comb-shaped (contrast stretched
   after acquisition), so thresholds are never taken from raw histogram bins.
3. **Normalise per image**: pore floor (0.5th percentile) → 0, graphite peak (mode of the smoothed histogram)
   → 1. Graphite width σ is the narrower half-width of the peak. This removes brightness/contrast differences
   (black level varies 0–24 and the graphite peak 49–63 between images).
4. **Si** (`segment.py`): hysteresis threshold anchored to the graphite peak. The low threshold is the
   histogram valley between the graphite and Si peaks (at least graphite + 3σ); seeds are halfway from the
   valley to the Si peak. If no Si peak exists, graphite + 4σ / + 6σ is used instead (`qc_threshold_method = sigma`).
   Otsu or a free 3-class GMM are deliberately not used: the Si peak is small and they label bright graphite
   edges or pore walls as Si (a naive GMM gave 31 % Si on `Batch_3/img_kbdh4tri`).
5. **Cleanup** (`cleanup.py`), each step aimed at a known artefact:
   - drop thin components with no pixel deeper than 4 px (edge brightening, charging rims);
   - drop components where < 30 % of the pixels reach the seed level (dim artefacts);
   - opening with r = 1, minimum area 30 px, fill holes < 200 px;
   - drop irregular components (solidity < 0.7), which removes bright porous binder/carbon-black mesh.

   The KPI `si_fraction_of_solids_unfiltered` reports the value before cleanup.
6. **Pore**: normalised intensity < 1 − 4σ. **Graphite**: everything else.

## Outputs (`--out`)

| Path | Content |
|---|---|
| `masks/<batch>_<id>_labels.png` | label map, 0 = pore, 1 = graphite, 2 = Si (full resolution) |
| `qc/<batch>_<id>_overlay_small.png` | whole image, Si orange, pore blue (4× downsampled) |
| `qc/<batch>_<id>_crop.png` | raw vs overlay, full-resolution crop with the most Si |
| `qc/<batch>_<id>_hist.png` | normalised histogram with thresholds |
| `kpis/per_image.csv` | all KPIs + QC values per image |
| `kpis/si_particles.csv` | one row per Si particle (area, ECD, centroid, aspect ratio, orientation, border flag) |
| `kpis/si_profiles.csv` | Si fraction of solids in 20 row bands per image |
| `kpis/batch_summary.csv`, `kpis/batch_stats.csv` | per-batch mean/sd/count; Kruskal–Wallis and pairwise Mann–Whitney with Cliff's delta |
| `report/report.md` | summary tables, plots, QC flags |

## KPIs (`kpis.py`)

| Group | KPI | Notes |
|---|---|---|
| Amount | `frac_si`, `frac_graphite`, `frac_pore` | area fractions |
| | **`si_fraction_of_solids`** | Si / (Si + graphite), independent of porosity |
| | `si_wt_pct_estimate` | area ratio × density ratio (2.33 / 2.26); valid only for pure Si |
| Size | **`si_ecd_d10/d50/d90`**, `si_ecd_d50_area_weighted` | equivalent circular diameter, number-based |
| | `si_count_per_mpx` (or `_per_mm2`) | |
| | `graphite_ecd_d50`, `graphite_aspect_ratio_median`, `graphite_orientation_order`, `graphite_orientation_mean_deg` | flakes split by distance-transform watershed; order +1 = all horizontal, 0 = random |
| Distribution | **`si_cv_w256/512/1024`** | coefficient of variation of Si fraction of solids over windows |
| | `si_nn_distance_median`, `si_particles_per_cluster_mean`, `si_fraction_particles_clustered` | clusters = Si particles less than 5 px apart |
| | `si_contact_graphite`, `si_contact_pore` | what surrounds Si, sampled 2–5 px outside each particle |
| | `si_profile_rel_slope` | Si-fraction slope from top to bottom row, relative to the mean |
| Composition | **`si_grey_median_n`**, `si_grey_iqr_n` | Si grey level relative to the graphite peak; a shift suggests a different Si material |
| QC | `qc_*` | normalisation, thresholds, method, Si peak position, `qc_bright_unassigned_fraction` (bright area not in particles: haze / nano-Si / rims) |

**Bold** = priority KPIs for detecting a supplier change.

## Limitations

- Binder and carbon black are indistinguishable from graphite or pore in BSE and are not measured.
- Porosity is the least reliable class: pores often show sub-surface material at graphite-like grey.
- No pixel-size metadata exists in the TIFFs, so sizes are in px until `pixel_size_um` is set.
- **Low-contrast images:** images whose Si peak sits close to graphite (normalised < 1.8, currently
  `Batch_1/img_4ih2ggld` and `img_5n1q8atc`) are flagged in the report. In those images some bright binder
  mesh next to Si particles is still counted as Si.
- The through-thickness profile only means something if image rows run through the electrode thickness.

## Development

```bash
pytest           # synthetic-image tests (known phase fractions, contrast/offset invariance, edge rims)
ruff check . && ruff format --check .
python scripts/explore_histograms.py <data> hist.png   # normalised histograms of all images
```
