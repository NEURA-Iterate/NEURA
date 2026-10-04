# Rotating held-out batch CV

This experiment evaluates a 12-KPI physics pool (`BASE` in `rotating_cv.py`) over 17
rotating rounds. Each round holds out one image per batch. Batch_3 has 17 images and
each is held out once; Batch_1 and Batch_2 have 7 images each, so those images repeat
two or three times. Within each round, `rank_kpi_subsets` searches subsets of up to
three KPIs using only the training rows. The selected subset is evaluated with
diagonal covariance.

From `bayesian_kpi_classifier/`, run:

```bash
python experiments/batch_cv/rotating_cv.py \
  --kpis <path-to-anode_qc-per_image.csv> \
  --out <output-directory>
```

The default `--kpis` path is
`anode_microstructure_qc/results/kpis/per_image.csv` relative to the repository
root. That generated KPI table may not exist in a fresh checkout; first produce it
with `anode_qc`, or pass another compatible per-image KPI CSV. The default output is
`experiments/batch_cv/results/`; the script writes `predictions.csv`, `per_image.csv`,
`summary.txt`, and `per_image_probabilities.png`.

## Results

The run produced **37/51** correct predictions (accuracy **0.725**), with mean
log-loss **0.573** versus the uniform baseline **1.099**. Per-image averaged
probabilities classified **25/31** images correctly.

| Batch | Accuracy |
| --- | ---: |
| Batch_1 | 0.706 |
| Batch_2 | 0.529 |
| Batch_3 | 0.941 |

| Tier | Predictions | Accuracy |
| --- | ---: | ---: |
| High | 10 | 1.00 |
| Medium | 16 | 0.75 |
| Review | 25 | 0.60 |

`frac_pore|graphite_crack_density|si_cv_w256` was selected in **15/17** rounds.

## Feature-selection experiments

Two alternatives were tried and rejected. A six-step funnel (physics and reliability
filters, Spearman redundancy filtering at |ρ| > 0.8, Kruskal-Wallis ranking, 60%
bootstrap stability, then subset search up to three KPIs) reached **26/51** and
**17/31** correct with log-loss **0.95**. Pairwise per-batch-pair ranking reached
**23/51**, **16/31**, and log-loss **1.02**. These selection experiments are not
committed in this repository.

Small-sample rankings favor Batch_3-versus-rest KPIs and split the Batch_1/Batch_2
Si-heterogeneity signal across correlated variants (`si_cv` at 256/512/1024 and Si
elongation p90); none passed the stability filter. The physics pool plus
training-only subset search is therefore retained. `porosity_detector_disagreement`
is a QC metric and remains excluded pending a check that it is not an acquisition
artefact.

The pool was hand-picked after exploring the same **31 images** (7/7/17 across
batches), so the result is somewhat optimistic.
