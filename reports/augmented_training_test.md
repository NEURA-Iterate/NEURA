# Can aggressive augmentation + a trained encoder separate Batch 1 from Batch 2?

Context: Batch 1 and Batch 2 are *meant* to differ, but hand KPIs, frozen
encoders and population tests (`batch_difference_and_reliability.md`) found no
difference.  This note tests whether training on heavily augmented tiles finds
one.  Everything is evaluated on whole held-out samples: no tile, view or
augmented copy of a held-out sample is ever used in training.  Each pipeline is
also run with batch labels permuted *at the sample level* (Exp C) so a real-label
result can be compared with what the same code achieves on fake labels.

## Setup

* Tiles: 384 px at 2x downsampling (~18 per image, 1710 in total, 31 samples).
* Augmentation (`augment_train.augment`): random flips, 90 degree rotations,
  random crop of 60-100 % rescaled to 224 px ("geometric"); optionally gamma
  0.7-1.4, contrast 0.7-1.3, brightness +-0.1, Gaussian blur sigma 0-1, Gaussian
  noise sigma 0-0.05 ("photometric").  All transforms keep a tile a plausible
  SEM micrograph of the same material.
* Exp A: 6 augmented copies per tile -> frozen DINOv2 ViT-S/14 -> logistic
  regression (C = 0.01) trained on the tiles of the 13 (or 30) training samples;
  held-out verdict = mean tile probability.  20 sample-level label shuffles.
* Exp B: ImageNet ResNet-18 fine-tuned end to end (AdamW, one-cycle 3e-4, label
  smoothing 0.1, 8 epochs, fresh augmentation every step); 7 folds each holding
  out one Batch-1 and one Batch-2 sample.  Label-shuffle runs use the identical
  code with `--shuffle-seed`.

## Exp A: augmented tiles, frozen encoder, trained head

Batch 1 vs 2, n = 14 (`augmented_expA_*_b1_vs_b2.csv`):

| view | aug set | accuracy (no train aug / train aug) | shuffle mean | shuffle p95 |
|---|---|---|---|---|
| BSE | geo+photo | 0.36 / 0.36 | 0.37 | 0.58 |
| BSE | geo | 0.36 / 0.29 | 0.39 | 0.72 |
| SE | geo+photo | 0.57 / 0.50 | 0.47 | 0.71 |
| SE | geo | 0.57 / 0.57 | 0.45 | 0.72 |
| InLens | geo+photo | 0.43 / 0.50 | 0.48 | 0.72 |
| InLens | geo | 0.43 / 0.50 | 0.48 | 0.65 |
| all views | geo+photo | 0.43 / 0.43 | 0.36 | 0.64 |
| all views | geo | 0.43 / 0.57 | 0.38 | 0.65 |

Nothing leaves the shuffled-label range (smallest p = 0.33).  BSE is *below*
chance in every configuration: a held-out sample's tiles look more like the
other batch's training tiles than its own, i.e. within-batch variation is at
least as large as between-batch.  Adding photometric augmentation changed
nothing, so acquisition invariance is not the missing ingredient for the head.

Three batches, n = 31 (`augmented_expA_geo_photo_3batch.csv`): 0.60-0.74 vs
shuffle p95 0.55-0.58, i.e. above chance, but the confusion matrices show it is
entirely Batch 3 (16-17/17 correct) while Batch 2 is 1-2/7 - the same pattern as
the hand KPIs and the frozen encoder.

## Exp B: ResNet-18 fine-tuned end to end on augmented tiles

Batch 1 vs 2, n = 14, per-sample results in `augmented_expB_runs.csv`:

| view | aug set | seed | accuracy | confusion (true -> pred) |
|---|---|---|---|---|
| BSE | geo | 0 | 0.71 | B1 5/7, B2 5/7 |
| BSE | geo | 1 | 0.71 | B1 4/7, B2 6/7 |
| BSE | geo+photo | 0 | 0.57 | B1 5/7, B2 3/7 |
| SE | geo+photo | 0 | 0.71 | B1 5/7, B2 5/7 |
| SE | geo+photo | 1 | 0.57 | B1 3/7, B2 5/7 |
| 3-batch BSE | geo+photo | 0 | 0.61 | called 29/31 "Batch 3" (majority class) |

Averaging the per-sample probabilities of the four geo-BSE / photo-SE runs
gives 10/14 again.  The ranking by p(Batch 2) is: the two Batch-1 samples with
dim Si particles (`img_4ih2ggld`, `img_5n1q8atc`, p = 0.09/0.10) are the most
confidently "Batch 1"; six of seven Batch-2 samples sit at 0.61-0.80;
`img_f1vzngrs` (Batch 1) is called Batch 2 at p = 0.95 in every run; three
Batch-1 samples are undecided at 0.42-0.59.

## Exp C: sample-level label-shuffle control for Exp B

Same code, labels permuted across the 14 samples before fold construction:

| pipeline | shuffled-label accuracies | real-label accuracy |
|---|---|---|
| BSE geo+photo | 0.29, 0.50, 0.64 | 0.57 |
| BSE geo | 0.50, 0.57, 0.43, 0.36, 0.43 | 0.71, 0.71 |
| SE geo+photo | 0.43, 0.50, 0.50, 0.14, 0.57, 0.43 | 0.71, 0.57 |

With 14 samples one correct call is 0.07 accuracy; the shuffled runs span
0.14-0.64 and the real-label runs 0.57-0.71.  The best real result (BSE geo,
0.71 on two seeds) exceeds all five BSE-geo shuffled runs, but with five
permutations that is only an empirical p of 0.17; against a binomial(14, 0.5)
null 10/14 has one-sided p = 0.09.  The other configurations sit inside their
shuffle range.  This is weakly suggestive, not evidence, and it is not stable
across seeds or augmentation sets.

## What the fine-tuned network actually uses

Spearman correlation of the ensemble p(Batch 2) with sample-level descriptors
(n = 14): **BSE mean grey (`qa__`) rho = -0.82**, Si grey relative to graphite
+0.69, BSE pore count +0.53, SE mean grey -0.53, BSE noise sigma -0.51.  The
strongest correlate is an acquisition descriptor, not a material KPI, which is
consistent with photometric augmentation *lowering* accuracy (it removes the
cue) and with the two confident Batch-1 calls being the two samples whose Si
particles are anomalously dim.  Robust per-image normalisation does not remove
this cue because it preserves the relative brightness of phases.

## Conclusions

1. Aggressive augmentation plus a trained head on a frozen encoder: chance for
   Batch 1 vs 2, in every view and augmentation set (Exp A).
2. End-to-end fine-tuning on augmented tiles: 0.57-0.71, seed-dependent, within
   or at the edge of the label-shuffle null, and the signal it does use is BSE
   brightness, which is an imaging setting rather than a material property.
   Adding the photometric augmentation that would make the network blind to
   brightness pushes it back toward 0.5.
3. Augmentation cannot manufacture independent samples: 7 vs 7 whole samples
   remains the limiting factor, and the shuffle null is correspondingly wide.
4. Together with the earlier experiments (hand KPIs, frozen encoders,
   population tests, tile-level and Si-particle probes) the data do not contain
   a Batch 1 vs Batch 2 difference that any of these methods can detect at this
   sample size.  If the two batches are known to differ, the next step is to
   learn *what* differs (formulation vs processing) and target that measurement,
   and to obtain pixel size / session metadata so that brightness cues can be
   separated from material cues.
