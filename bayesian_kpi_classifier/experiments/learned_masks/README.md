# Learned-mask KPI experiment

This experiment recomputes the per-image KPI table from FusionHead predictions, then
runs the same 17-round rotating classifier evaluation used for the rule-based KPI
table.

## KPI recomputation

`learned_kpis.py` uses the holdout Fusion checkpoint's uint8 argmax maps in class
order `pore`, `graphite`, `si`, `binder`. Each map aligns with the border-cropped
image returned by `anode_qc.data.load_bse` (`border_px=2`); there is no additional
offset. The map classes are translated to anode_qc labels. Si components smaller
than `Config.cleanup.si_min_area_px` are relabelled as binder.

The script recomputes KPIs with `anode_qc.kpis_from_labels`. Crack densities still
use the ETD crack detector, evaluated within the learned phase regions. Rule-table
`trust_flags` are carried into the output unchanged. In `rule` mode, the script
reproduces the existing KPI table exactly on two images.

From `bayesian_kpi_classifier/`, with `anode_qc` installed, run:

```bash
python experiments/learned_masks/learned_kpis.py \
  --images <image-directory> --mode learned \
  --pred-dir <directory-of-image_id-npy-maps> \
  --base <rule-per_image.csv> \
  --out experiments/learned_masks/learned_per_image.csv --procs 5
python experiments/batch_cv/rotating_cv.py \
  --kpis experiments/learned_masks/learned_per_image.csv \
  --out experiments/learned_masks/results/learned
```

`--mode rule` recomputes the baseline without prediction maps. `--ids` can restrict
the run to selected image IDs. The paths are CLI arguments; the script has no
machine-specific data paths.

## Classifier results

The learned-KPI classifier uses the same 17 rounds as the rule-KPI comparison:

| KPI inputs | Correct | Mean log-loss | Per-image correct | Batch_1 | Batch_2 | Batch_3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule masks | 37/51 (0.725) | 0.573 | 25/31 | 0.706 | 0.529 | 0.941 |
| Learned masks | 39/51 (0.765) | 0.676 | 24/31 | 0.824 | 0.706 | 0.765 |

Learned-mask confusion counts (rows are true batches, columns are predictions):

| True \ Predicted | Batch_1 | Batch_2 | Batch_3 |
| --- | ---: | ---: | ---: |
| Batch_1 | 14 | 3 | 0 |
| Batch_2 | 2 | 12 | 3 |
| Batch_3 | 1 | 3 | 13 |

| Confidence tier | Predictions | Accuracy |
| --- | ---: | ---: |
| High | 13 | 1.00 |
| Medium | 24 | 0.583 |
| Review | 14 | 0.857 |

Selected subsets were `frac_pore|graphite_aspect_ratio_median|si_cv_w256` in 12/17
rounds and `frac_pore|graphite_crack_density|graphite_aspect_ratio_median` in 5/17.
The copied per-round outputs are in `results/learned/`; rule-KPI outputs are in
`results/rule/`.

Learned graphite crack density decreased in Batch_3 (median 67 to 43, near Batch_2
at 41), removing its main separator. Porosity and `si_cv_w256` retain nearly the
same rank (Spearman ≈0.96); graphite aspect ratio changed most (ρ=0.61).

## Batch differences and trust flags

The batch-difference table uses rule masks and Mann-Whitney tests. Batch_3 is more
porous (11.5/12.4/14.3%), has more graphite cracks (48/51/67), and slightly more
elongated graphite (1.59/1.61/1.64), with p ≤ 0.016. Between Batch_1 and Batch_2,
only `si_cv_w256` differs (1.81 vs 2.02, p=0.038). No difference was found in Si
fraction, Si D50/D90, dispersion, binder fraction, graphite alignment, or Si cracks.
Across 36 tests, about two false positives are expected at 0.05.

`low_si_contrast` marks `qc_si_peak_n < 1.8` or a non-valley threshold; the flagged
Batch_1 images are `img_4ih2ggld` and `img_5n1q8atc`. `charging` marks more than 5%
saturated BSE/Inlens pixels; the flagged Batch_3 images are `img_hawkfj64`,
`img_mgxahqnk`, and `img_xgj4xftb`. Flags are retained, not used to exclude images.

## Caveats

This is a hybrid pipeline: phase masks come from a pseudo-label-trained model, while
crack detection remains rule-based. The masks are not expert annotations; 25/31
images were used to train the segmentation model, although no batch labels were
used. Differences of about two predictions are within noise, so the learned-KPI
result does not establish superior batch classification.
