# Frozen-encoder embedding test: can a pretrained encoder separate the three batches?

**Question.** Does a frozen, pretrained image encoder give an embedding that
(1) assigns held-out samples to the right batch - in particular Batch 1 vs
Batch 2, which the hand-made KPIs could not separate - and (2) encodes material
structure rather than microscope settings?

**Answer (short).** No on (1): leave-one-sample-out accuracy is the same as with
the 26 KPIs (~0.6 overall) and Batch 1 vs 2 stays inside the shuffled-label
range for every view and every encoder. On (2) the probe is a warning: the
embedding predicts imaging conditions (InLens/SE noise level, mean grey) at
least as well as any material KPI. A larger or fine-tuned encoder trained on
these ~30 samples is not going to make a Batch-1-vs-2 claim more reliable;
the missing piece is a physical descriptor or acquisition metadata.

## Setup

- Encoders (frozen, no training): DINOv2 ViT-S/14 (384-d) and ImageNet
  ResNet-50 (2048-d, penultimate layer). CPU inference, `encoder.py`.
- Each view image is downsampled 2x, robust-normalised to [0, 1] per image,
  tiled into 224 px squares (448 px on the original), each tile encoded,
  and the per-view embedding is the mean over ~70 tiles. The three views are
  three descriptions of one sample, so the unit of analysis stays the sample:
  one row = `sample_id, batch_id, emb__BSE__*, emb__SE__*, emb__InLens__*`.
- Same 31 labelled samples as `reports/kpi_hypothesis_test.md` (7 / 7 / 17;
  one Batch-3 sample has no InLens image).
- Test 1 uses the unchanged leave-one-sample-out harness (`evaluation.py`):
  scaling/centroids refit on the 30 other samples, held-out sample scored
  against the three batch references. Two nearest-reference scorers:
  `cosine` (centre on training mean, L2-normalise each view block, cosine
  distance to the batch mean) and `profile` (median/MAD-scaled Euclidean, as
  for the KPIs). Chance is estimated by repeating the whole LOO with
  permuted labels (30 shuffles).
- Test 2 (`embedding_probe.py`): for each KPI and each imaging `qa__`
  measure, StandardScaler -> PCA(8) -> RidgeCV fitted on 30 samples, predict
  the 31st; report LOO R^2 and Spearman rho. Everything is refit per fold.

## Test 1: can the embedding classify held-out samples?

Full table: `reports/encoder_loo_results.csv` (per view / combined, both
scorers, all 31 samples and the Batch 1 + 2 subset).

| encoder | views | scorer | all 31: LOO acc (shuffle mean / p95) | B1 vs B2 only, n = 14: LOO acc (shuffle p95) |
|---|---|---|---|---|
| DINOv2 | BSE | cosine | **0.68** (0.32 / 0.47) | 0.57 (0.75) |
| DINOv2 | SE | cosine | 0.58 (0.32 / 0.44) | 0.57 (0.83) |
| DINOv2 | InLens | cosine | 0.52 (0.33 / 0.48) | 0.57 (0.79) |
| DINOv2 | all three | cosine | 0.61 (0.32 / 0.48) | 0.64 (0.79) |
| DINOv2 | all three | profile | 0.58 (0.36 / 0.47) | 0.57 (0.79) |
| ResNet-50 | BSE | cosine | 0.61 (0.34 / 0.48) | 0.79 (0.71) |
| ResNet-50 | SE | cosine | 0.61 (0.33 / 0.47) | 0.57 (0.71) |
| ResNet-50 | InLens | cosine | 0.58 (0.34 / 0.52) | 0.57 (0.79) |
| ResNet-50 | all three | cosine | 0.61 (0.33 / 0.48) | 0.64 (0.75) |
| ResNet-50 | all three | profile | 0.58 (0.41 / 0.52) | 0.57 (0.75) |
| *26 hand KPIs (previous report)* | all three | profile | 0.61 (0.37 / 0.57) | 0.50 (chance) |

Confusion matrices for the all-view cosine configurations:

```text
DINOv2                      ResNet-50
true\pred   1   2   3       true\pred   1   2   3
1           4   3   0       1           5   2   0
2           2   4   1       2           4   2   1
3           0   6  11       3           2   3  12
```

- Overall accuracy is 0.58-0.68 for every encoder/view/scorer, i.e. the same
  as the 26 hand KPIs. Above chance, but as before the signal is "Batch 3 vs
  the rest" (11-14 of 17 correct).
- **Batch 1 vs Batch 2 is not separated.** On the 14-sample subset, 15 of the
  16 encoder configurations fall inside the shuffled-label 95 % range. The one
  exception (ResNet-50, BSE only, cosine: 0.79 vs p95 0.71) is one of 16
  comparisons and does not reappear with the other encoder, the other scorer,
  or when the other two views are added, so it should be treated as the
  expected one-in-sixteen false positive, not a finding. The BSE view is
  nevertheless the most useful single view for both encoders, and InLens the
  least useful, which matches the KPI picture.
- Margins: with cosine distances to centroids the best and second-best
  distances differ by a few percent for every sample (all 31 are flagged by
  the 15 % relative-margin rule), so the embedding gives no basis for a
  confident call on any individual sample.
- Preprocessing robustness: re-embedding at 3x instead of 2x downsampling
  (DINOv2, all views, cosine) changes 1/31 assignments (accuracy 0.61 ->
  0.65). The embedding is more stable to this change than the hand KPIs were
  (3/31), but it is stable around a wrong answer for Batch 1 vs 2.

## Test 2: what does the embedding represent?

LOO ridge probe from the all-view embedding to each measurement
(`reports/encoder_probe_dinov2.csv`, `reports/encoder_probe_resnet50.csv`).

| target (LOO R^2, DINOv2 / ResNet-50) | kind |
|---|---|
| `kpi__InLens__dark_fraction__frac` | 0.80 / 0.78 | material (but tracks InLens brightness, see KPI report) |
| `qa__InLens__noise_sigma__8bit` | 0.77 / 0.79 | imaging |
| `kpi__BSE__bright_phase_fraction__frac` | 0.79 / 0.66 | material |
| `kpi__InLens__bright_feature_elongation__ratio` | 0.73 / 0.67 | material |
| `kpi__BSE__pore_fraction__frac` | 0.70 / 0.49 | material |
| `qa__InLens__mean_grey__8bit` | 0.67 / 0.54 | imaging |
| `qa__SE__noise_sigma__8bit` | 0.64 / 0.66 | imaging |
| `qa__BSE__mean_grey__8bit` | 0.64 / 0.58 | imaging |
| `kpi__BSE__solid_chord_vertical__px` (best Batch-3 KPI) | 0.42 / 0.30 | material |
| `kpi__SE__dark_fraction__frac` | -0.10 / 0.14 | material |
| median over all `kpi__` targets | 0.35 / 0.30 | |
| median over all `qa__` targets | 0.40 / 0.36 | |

- The embedding does carry real microstructural information: BSE pore and
  bright-phase fractions are recovered with R^2 0.7-0.8 from DINOv2 (the
  per-view probe confirms this comes from the BSE block, R^2 0.67 / 0.73).
  So "can KPIs be reconstructed from the embedding?" - yes for the
  area-fraction type KPIs, partly for sizes, not for chord lengths or SE
  dark fraction.
- But imaging conditions are recovered at least as well: noise level in every
  detector (R^2 0.6-0.8), mean grey of InLens and BSE (0.55-0.67). The
  InLens block alone predicts InLens mean grey with R^2 0.84 and noise with
  0.82, higher than any material KPI. The median R^2 is higher for `qa__`
  than for `kpi__` targets with both encoders. This is the failure mode of
  learned micrograph features reacting to microscope-induced intensity /
  noise changes, and it means an embedding-based batch decision would be
  partly a detector-settings decision.
- Reconstructing a KPI from the embedding is a probe, not an explanation: the
  KPIs it reconstructs best (InLens dark fraction, BSE bright fraction) are
  not the ones that separate batches, and nothing here shows that the batch
  centroids differ *because* of those KPIs.

## What this means for the team

1. Two independent routes - 26 interpretable KPIs and two frozen generic
   encoders - give the same answer: Batch 3 is recognisably different, Batch 1
   and Batch 2 are not distinguishable from these images on held-out samples.
   Training an encoder on these ~30 samples will not change that; it would
   only let the model memorise the per-session intensity/noise signature.
2. The strongest batch-correlated quantities in both routes are imaging
   descriptors. Before any further modelling, ask the organisers for
   acquisition metadata (session, detector settings, pixel size) and whether
   Batches 1 and 2 are *meant* to be distinguishable. If they are, the
   missing descriptor is probably not visible as a texture at this scale
   (e.g. it may need composition, particle-level statistics at higher
   magnification, or the coating thickness that was stripped from the TIFFs).
3. For the QC product, the embedding is still usable as a *second opinion*
   alongside the KPIs (same CSV contract, same harness), but every
   embedding-based verdict must be reported with the imaging `qa__` check
   next to it, and the Batch-1/2 ambiguity must be surfaced as
   "investigate", never as a confident accept/reject.

## Reproduce

```bash
pip install -r requirements.txt   # torch/torchvision CPU wheels are fine
python build_manifest.py --data-dir /path/to/Hackathon-Polaron --out sample_manifest.csv
python encoder.py --manifest sample_manifest.csv --model dinov2_vits14 --out features_emb_dinov2.csv
python encoder.py --manifest sample_manifest.csv --model resnet50 --out features_emb_resnet50.csv
python evaluation.py --features features_emb_dinov2.csv --feature-prefix emb__ --method cosine --out eval_emb
python evaluation.py --features features_emb_dinov2.csv --feature-prefix emb__BSE__ --method cosine --out eval_emb_bse
python embedding_probe.py --embeddings features_emb_dinov2.csv --targets features_real.csv --out embedding_probe.csv
```

DINOv2 weights/code are fetched by `torch.hub`; the run here used a local
copy in `~/.cache/torch/hub`. CPU time: ~8 s per sample (DINOv2), ~12 s
(ResNet-50).
