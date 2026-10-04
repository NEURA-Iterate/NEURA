# Classifier research: balancing rule-based and learned-mask KPIs

Branch `classifier-research` (from `bayesian-kpi-classifier-v2`). Goal: a batch classifier
that does well on **all three** batches, an understanding of the general trends that
separate them, and a better representation of crack density.

All numbers use the **same 17-round rotating hold-out** as `experiments/batch_cv`
(one image per batch held out per round, 51 held-out predictions, 17 per batch, so
overall accuracy equals balanced accuracy). Every label-dependent choice (KPI subset,
weights, LDA, stage features, crack-rim width in the nested run) is made inside each
round on the training images only. Batch priors are uniform.

`fastnb.py` re-implements the diagonal-covariance `BayesianStudentTClassifier` in
vectorised form so many variants can be scored quickly. It reproduces the package
probabilities to 6 decimals and the published rule-based baseline exactly
(37/51, log-loss 0.573, same subset in 15/17 rounds).

## Headline results

| Model (17 rounds, 51 held-out predictions) | Correct | B1 | B2 | B3 | Per image (31) | Log-loss |
|---|---|---|---|---|---|---|
| Rule KPIs, baseline (`batch_cv`) | 37 | 0.71 | 0.53 | 0.94 | 25 | 0.573 |
| Learned-mask KPIs (`learned_masks`) | 39 | 0.82 | 0.71 | 0.76 | 24 | 0.676 |
| Rule KPIs, **batch-balanced subset search** | 39 | 0.71 | 0.65 | 0.94 | 26 | **0.509** |
| **Ensemble: geometric mean of rule + learned classifiers** | **43** | **0.82** | **0.76** | **0.94** | **27** | 0.564 |
| Rule KPIs, crack density = graphite-interior cracks (8 px rim, fixed) + balanced search | 43 | 0.71 | 0.94 | 0.88 | 27 | 0.522 |
| Same, rim width (4/8/12/16 px) chosen inside each round | 39 | 0.65 | 0.71 | 0.94 | 25 | 0.597 |

Round-bootstrap 95% ranges for the gain over the baseline (resampling the 17 rounds):
ensemble +6 [+2, +10]; balanced search +2 [0, +5]; interior cracks (8 px) +6 [+2, +10].

**Recommendation:** use the **rule + learned ensemble** (`final_model.py --how geo`) as the
default classifier. It is the only change that improves B1, B2 and B3 at the same time
without a tuned parameter: the two mask sources make different errors (rule masks keep
B3 sharp through crack density, learned masks separate B1/B2 better through graphite
shape), and averaging log-probabilities keeps both. Arithmetic and geometric averaging
give the same 43/51. Also switch the KPI-subset search to batch-balanced log-loss
(B3 has 17 of 31 images and otherwise dominates the search).

The interior-crack result (43/51) is **promising but not yet trustworthy**: it is sensitive
to the rim width that was picked after seeing results (4 px: 41, 8 px: 43, 12 px: 37,
16 px: 33). When the rim is chosen honestly inside each round the result is 39/51. The
search does prefer interior crack density over the original in 16 of 17 rounds, so it
is a better KPI on training data; fix the rim a priori (4 px) and confirm on new images.

Outputs: `results/runs/final_ensemble/` and `results/runs/final_interior/` contain
`predictions.csv`, `per_image.csv`, `summary.txt` and `per_image_probabilities.png`
(same chart layout as `batch_cv`).

Confidence tiers (top probability only; typicality is not computed in the fast path):
ensemble high 11/11 correct, medium 16/20, review 16/20. The review tier does not
separate right from wrong well for the ensemble; the interior-crack model does better
(high 16/16, medium 13/13, review 14/22).

## What did not help

| Idea | Best result | Note |
|---|---|---|
| Per-KPI weights (non-negative, L2, fit on balanced LOO log-loss) | 33–36/51 | 12 weights from 24 training images overfit |
| Two-stage: B3 vs rest, then B1 vs B2, separate subsets | 31–35/51 | stage 2 has only 12 training images |
| LDA projection (shrinkage) to 2-D, then classifier | 27–34/51 | B1/B2 directions are noise at this sample size |
| One search over all 24 rule + learned KPIs | 34–38/51 | more candidates, more chance picks |
| Hybrid table (learned KPIs + rule crack density) | 35–38/51 | worse than ensembling the two full classifiers |
| Engineered ratios (crack/pore, Si/binder, multi-scale Si CV) | 37–40/51 | no consistent gain |
| Extra crack descriptors in the pool (count, length p90, cracked-particle fraction, CV, k=4/8) | 31–39/51 | the original length density is chosen anyway |
| Tile-level classifier (4 tiles per image, mean tile log-likelihood) | 32–34/51 | tiles are noisier than images; see variance decomposition |

## General trends between batches

Figures in `results/figures/`.

- `kpi_distributions.png`: per-image values by batch, rule (circles) and learned (triangles).
- `decision_regions_rule.png`, `decision_regions_learned.png`, `decision_regions_rule_interior.png`:
  classifier regions for the three key KPI pairs, fit on all images (descriptive).
- `lda_projection.png`: 2-D LDA on all 12 KPIs (descriptive, fit on all images).
- `rule_to_learned_shift.png`: how each image moves when switching rule to learned masks.

1. **B3 is separated by porosity and graphite cracking together.** In the porosity x crack
   density plane B3 sits at high/high; one B3 image (`img_0grcilhi`: highest porosity, 22%, but low crack density, 41) is the
   exception.
   LD1 of the LDA is essentially porosity + crack density + graphite aspect ratio.
2. **B1 vs B2 overlap.** The only consistent separators are Si heterogeneity
   (`si_cv_w256`, B2 higher) and, on learned masks, graphite aspect ratio. Neither gives
   a clean boundary; B1/B2 errors account for most remaining mistakes.
3. **Most KPIs vary more within an image than between batches**
   (`tile_variance_decomposition.png`, 4 vertical tiles per image). Porosity and crack
   density have 32–44% of variance between batches; Si fraction, D50/D90, heterogeneity
   and dispersion have under 7% between batches and 41–73% within one image. B1/B2 separation via Si is therefore limited by
   sampling area: more or larger fields of view per sample should help more than new
   features.
4. **Rule to learned masks:** porosity and crack density drop for every image; B3 cracks
   drop most. Si–graphite contact goes from about 0.75 to 0.2 because the learned model
   puts binder rims around Si, so the learned contact KPI measures something different.

## Crack density: how it is represented

Current definition (`anode_qc.kpis.crack_mask`): Sato ridge filter on ETD at 1/2
resolution, threshold median + k·MAD (k = 6) inside Si ∪ graphite (eroded 2 px), cleaned
and skeletonised. Density = skeleton length in graphite per 1e4 graphite pixels.

Figures:
- `crack_maps.png`: one median-density image per batch: local crack density in 256 px
  windows over the BSE image, and zooms of the densest window with rule (red) and
  learned (cyan) crack skeletons.
- `crack_representations.png` (+ `tables/crack_representation_tests.csv`): nine
  representations per batch with Mann-Whitney p-values.
- `crack_location.png` (+ `tables/crack_location*.csv`): where detected cracks sit.

Findings:
- **Threshold choice does not matter for ranking.** k = 4, 6, 8 give the same batch order
  (B3 > B2 > B1) and similar p-values; only the scale changes.
- **Length density and count density carry the same signal**; mean / p90 crack length
  add a weaker B3 signal (B3 cracks are slightly longer). Orientation (fraction
  horizontal) and the fraction of cracked graphite components do not separate batches
  on rule masks (graphite components merge into large connected regions, so the
  per-particle metrics are not meaningful without particle splitting).
- **About 20% of rule-mask "graphite cracks" lie within 8 px of a Si particle**
  (5% for learned masks). The zooms show these are ridges tracing Si/graphite
  interfaces, not cracks through graphite. Removing cracks within a thin rim of Si or
  pore ("interior crack density") keeps the B3 signal (B2 vs B3 p = 0.004) and
  sharpens B1 vs B2 (p = 0.018 vs 0.053). This is a measured detector artefact; whether
  every removed ridge is a false crack still needs expert review of the zooms.
- **Learned masks lose the B3 crack signal even after removing interface ridges**
  (interior density B2 vs B3 p = 0.29). So the B3 loss is not explained by Si-edge
  ridges alone; why the learned graphite region misses B3 cracks is still open
  (hypothesis: cracked graphite is labelled binder/pore by the learned model).
- Crack density varies strongly across one image (CV over 256 px windows is about
  0.7–0.9) and correlates with porosity, which is why crack density and porosity together
  delimit B3 better than either alone.

Recommended crack KPI set for reporting: interior graphite crack length density
(fixed 4 px rim), crack count density, and the crack map itself for review.

## Limitations

- 31 images (7 / 7 / 17). Many variants were tried on the same images, so the best numbers
  are optimistic; the round bootstrap only captures sampling noise of the held-out
  images, not the selection among variants.
- 25 of 31 images were in the segmentation model's training set (no batch labels used).
- Learned masks are trained on rule pseudo-labels; cracks still come from the ETD
  detector in both pipelines.
- Crack-location statistics use 1/4-resolution labels; the approximate crack density
  reproduces the full-resolution value closely (medians 47.3 vs 47.7 for B1).
- Sizes are in pixels (no pixel size in the TIFFs).

## Reproduce

Inputs: images in `/home/ubuntu/data/neura` (HF `gabrielgramicelli/NEURA-iterate-hack`),
learned FusionHead class maps from `feature-classifier` (`predict_all/*.npy`), rule KPI
table `anode_microstructure_qc/results/kpis/per_image.csv`, learned table
`../learned_masks/learned_per_image.csv`.

```bash
cd bayesian_kpi_classifier/experiments/research
# crack representations, tile KPIs, per-image mask/crack artifacts (~15 min, 4 procs)
python extract_features.py --images DATA --pred-dir PRED --mask-dir MASKS --out OUT --procs 4
# classifier strategies (writes summary.csv, pred__*.csv, chosen__*.csv)
python strategies.py --rule RULE.csv --learned ../learned_masks/learned_per_image.csv --cracks OUT/crack_features.csv --out RUNS
python crack_location.py --mask-dir MASKS --cracks OUT/crack_features.csv --out FIGS
python crack_viz.py --cracks OUT/crack_features.csv --mask-dir MASKS --images DATA --out FIGS
python tiles.py --tiles OUT/tile_kpis.csv --out FIGS
python viz.py --rule RULE.csv --learned ../learned_masks/learned_per_image.csv --out FIGS
# final ensemble chart
python final_model.py --preds RUNS/pred__rule__subset.csv RUNS/pred__learned__subset.csv --how geo --kpis RULE.csv --out FINAL
```

Files: `fastnb.py` (vectorised classifier + rounds), `extract_features.py`,
`strategies.py`, `crack_location.py`, `crack_viz.py`, `tiles.py`, `viz.py`,
`final_model.py`; results in `results/` (figures, tables, per-run predictions).
