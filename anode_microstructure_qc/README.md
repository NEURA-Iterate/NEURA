# anode_microstructure_qc: Si / graphite anode QC from BSE, ETD and Inlens SEM images

Labels every pixel of a backscattered-electron (BSE) SEM image as **pore**, **graphite** or **silicon**, refined
with the ETD and Inlens images of the same field of view into **pore**, **graphite**, **Si**, **binder/carbon-black
(CBD)** and **thin gap**. It then computes the 10 key QC KPIs per image and per batch, each with a
pixel-ambiguity range and an algorithm-choice range ([`docs/KPI_SPEC.md`](docs/KPI_SPEC.md)). In BSE, brightness
follows atomic number, so Si (Z=14) appears bright, graphite (Z=6) mid-grey and pore dark.

## Quick start

```bash
cd anode_microstructure_qc
pip install -e ".[dev]"
export HF_TOKEN=...              # read access to gabrielgramicelli/NEURA-iterate-hack (private)
anode-qc download --dest data         # BSE + ETD/SE + Inlens images
anode-qc run --data data --out outputs --mc-runs 20
# outputs/report/report.md is the summary
```

`anode-qc run --limit 2` processes only the first 2 images; `--bse-only` ignores ETD/Inlens; `--mc-runs 0` and
`--no-pixel-uncertainty` skip the two uncertainty estimates (the Monte Carlo reruns the whole pipeline per
setting and dominates run time); `--config my.yaml` overrides any value in
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

### Multi-detector refinement (`multimodal.py`, used when `<id>_ETD.tif` / `_SE.tif` / `_Inlens.tif` exist)

The three detectors share the same pixel grid (checked: zero shift on all 31 fields).

7. **ETD normalisation** to graphite interiors (graphite ≥ 10 px from its edge = 1). ETD is surface/topography
   sensitive, so open pores are near-black (~0.1 × graphite) and consistent between images.
8. **Pore** = ETD below the histogram valley between pore and graphite; split by width into **wide pores**
   (opening with r = `gap_max_width_px`/2) and **thin gaps** (narrower dark structures plus deep black top-hat
   features on lightly smoothed ETD, which catch 1–2 px gaps).
9. **Si veto**: BSE Si components containing > 15 % ETD-black pixels are porous mesh, not Si.
10. **Binder / carbon-black (experimental)**: porous mesh is 4–6× rougher than graphite interiors in all
    detectors. CBD = local texture (geometric mean over BSE/ETD/Inlens of local SD ÷ graphite-interior SD)
    > 3, with BSE texture > 3 too (BSE does not see topographic relief of graphite surfaces). Rough rims along
    large pores, long gaps and Si edges are ignored unless they adjoin a mesh core. Inlens absolute brightness
    varies strongly between images, so only its texture is used.
11. **QC flags**: polishing-streak severity, charging (saturated pixels), BSE vs multi-detector Si agreement.

### Uncertainty (`uncertainty.py`, `montecarlo.py`)

- **Pixel ambiguity** (noise, blurred edges): per-pixel P(Si) from Gaussian fits of the Si and graphite
  brightness in BSE, P(pore) likewise in ETD; edge blur δ from the 10–90 % width of the brightness profile
  across Si edges. Each KPI is recomputed on confident-only, inclusive, shrunk-by-δ and grown-by-δ label maps:
  `<kpi>_pix_lo` / `_pix_hi`.
- **Algorithm choices** (thresholds, cleanup): `--mc-runs N` reruns the pipeline with N Latin-hypercube draws
  of 18 settings (`PARAM_RANGES`), the same draws for every image. `<kpi>_alg_p05` / `_alg_p95` and the setting
  with the largest |Spearman ρ| (`<kpi>_alg_driver`). `kpis/batch_robustness.csv`: share of runs in which the
  batch difference is significant; robust only if ≥ 95 %.

## Outputs (`--out`)

| Path | Content |
|---|---|
| `masks/<batch>_<id>_labels.png` | label map, 0 = pore, 1 = graphite, 2 = Si, 3 = CBD, 4 = gap (full resolution) |
| `qc/<batch>_<id>_overlay_small.png` | whole image, Si orange, pore blue, CBD green, gap magenta (4× downsampled) |
| `qc/<batch>_<id>_crop.png` | full-resolution crop with the most Si: BSE, overlay; second row ETD, BSE-only labels |
| `qc/<batch>_<id>_hist.png` | normalised histogram with thresholds |
| `kpis/per_image.csv` | all KPIs + QC values per image |
| `kpis/si_particles.csv` | one row per Si particle (area, ECD, centroid, aspect ratio, orientation, border flag) |
| `kpis/si_profiles.csv` | Si fraction of solids in 20 row bands per image |
| `kpis/batch_summary.csv`, `kpis/batch_stats.csv` | per-batch mean/sd/count; Kruskal–Wallis and pairwise Mann–Whitney with Cliff's delta |
| `kpis/key_kpis.csv` | the 10 key KPIs: batch medians, median interval widths, main algorithm driver |
| `kpis/mc_runs.csv`, `kpis/batch_robustness.csv` | algorithm Monte Carlo: KPIs per run and setting; robustness of batch differences |
| `report/report.md` | key KPI table, robustness, per-image intervals, plots, trust flags |

## Key KPIs (`kpis.py`), ranked by QC importance

| Rank | KPI | Columns |
|---|---|---|
| 1 | Si fraction of solids (→ Si wt %) | `si_fraction_of_solids`, `si_wt_pct_estimate` |
| 2 | Porosity + top-to-bottom gradient | `frac_pore` (pore + gap), `porosity_profile_rel_slope` |
| 3 | Si coarse tail D90 (with D50) | `si_ecd_d90`, `si_ecd_d50` (number-weighted ECD; also `si_ecd_d50_area_weighted`) |
| 4 | Si agglomeration | `si_dispersion_index_w512` (window CV ÷ CV of the same particles placed at random; 1 = random, `si_dispersion_random_sd_*` = spread under random placement), `si_clustering_index` (random ÷ observed mean nearest-neighbour distance) |
| 5 | Cracks + Si debonding | `si_crack_density`, `graphite_crack_density` (ETD dark-ridge skeleton length per 10⁴ px² of phase, polishing streaks suppressed), `si_debond_fraction` (gap share of the 0–3 px shell around Si) |
| 6 | Binder/carbon-black + gradient | `cbd_fraction_of_solids`, `cbd_profile_rel_slope` (experimental) |
| 7 | Si material fingerprint | `si_grey_median_n` (Si cores, normalised to graphite) |
| 8 | Si top-to-bottom gradient | `si_profile_rel_slope` |
| 9 | Graphite alignment | `graphite_alignment` (structure tensor of graphite boundaries: +1 horizontal, 0 random, −1 vertical), `graphite_alignment_coherence`, `graphite_alignment_angle_deg` |
| 10 | Si contact | `si_contact_graphite/cbd/pore/gap` (classes 2–5 px outside each particle; diagnostic) |

Supporting values: `frac_*`, `si_ecd_d10`, `si_count_per_mpx`, `si_cv_w*`, `si_fraction_particles_clustered`,
graphite size/shape, and `qc_*` fields (thresholds, edge blur, detector agreement, streaks, charging).

## Foundation-model segmentation (experimental, `dinoseg.py`)

Frozen DINO backbone (DINOv3 ViT-S/16 via `transformers`, falling back to DINOv2-small while the DINOv3
weights are gated) plus a small trained decoder:

1. `scripts/dino_extract_features.py` runs the frozen backbone on BSE, ETD/SE and Inlens separately at native
   resolution in patch-aligned tiles (per-image percentile stretch, so brightness settings do not leak in) and
   caches the patch features (`pip install -e ".[dino]"`).
2. `FusionDecoder` concatenates the three detectors' features, reduces them (1×1 + 3×3 conv), upsamples to pixels
   and refines boundaries with the stretched detector images. Only the decoder is trained.
3. Training targets come from `draft_targets`: interiors of pore / graphite / Si / CBD in a draft label map,
   with edges and gaps marked unknown. Until expert-reviewed labels exist this is the rule-based segmentation,
   so the decoder learns those rules (not ground truth).
4. Decoders are trained per specimen fold (all detector views of a field stay together) and the ensemble's
   spread gives model-disagreement uncertainty; `to_labels` keeps thin ETD gaps from the dedicated detector,
   and the existing `kpis_from_labels` computes KPIs from each member's masks.

## Limitations

- Binder/carbon-black is texture-based and not validated against ground truth: rough graphite surfaces can still
  be counted and smooth binder missed. Without ETD/Inlens it is not measured.
- Pores recessed below the surface but not black in ETD are undercounted. Without ETD, pores come from BSE
  (least reliable: pores often show sub-surface material at graphite-like grey).
- Si grey level is a relative change flag, comparable only at identical acquisition settings; not a chemical
  identification (SiOx needs calibration or EDS).
- The uncertainty ranges cover pixel ambiguity and algorithm choices only, not field-of-view sampling or bias
  against ground truth.
- No pixel-size metadata exists in the TIFFs, so sizes are in px until `pixel_size_um` is set.
- **Low-contrast images:** images whose Si peak sits close to graphite (normalised < 1.8, currently
  `Batch_1/img_4ih2ggld` and `img_5n1q8atc`) are flagged in the report. In those images some bright binder
  mesh next to Si particles is still counted as Si.
- The through-thickness profile only means something if image rows run through the electrode thickness.

## Development

```bash
pytest           # synthetic-image tests (phase fractions, invariance, edge rims, ETD pores/gaps, mesh, KPI indices, uncertainty)
ruff check . && ruff format --check .
python scripts/explore_histograms.py <data> hist.png   # normalised histograms of all images
```
