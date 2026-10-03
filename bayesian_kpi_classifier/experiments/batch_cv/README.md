# Rotating held-out batch CV

This experiment holds out one image from each batch per round for 17 rounds. Every
Batch_3 image is held out once, while Batch_1 and Batch_2 images are held out two or
three times. In each round, `rank_kpi_subsets` selects up to three KPIs using only the
training rows; the selected subset is then evaluated with the diagonal classifier.

From `bayesian_kpi_classifier/`, run:

```bash
python experiments/batch_cv/rotating_cv.py
```

By default, the script reads `anode_microstructure_qc/results/kpis/per_image.csv`
from the repository root and writes `predictions.csv`, `per_image.csv`, `summary.txt`,
and `per_image_probabilities.png` under `experiments/batch_cv/results/`. Override
the input or output directory with `--kpis` or `--out`.

The run produced 37/51 correct predictions (accuracy 0.725), mean log-loss 0.573,
and 25/31 correct classifications from per-image averaged probabilities. The subset
`frac_pore|graphite_crack_density|si_cv_w256` was selected in 15/17 folds. Confidence
tiers were high for 10 predictions, medium for 16, and review for 25.

The tier is `review` when the predicted batch probability is below 0.6 or its
typicality p-value is below 0.05; otherwise it is `high` when the probability is at
least 0.9, and `medium` otherwise.

The 12-KPI pool was hand-picked after exploring these same 31 images, so this
evaluation is somewhat optimistic. Automatic univariate feature-selection funnels
did worse (26/51 and 23/51), because they drop `si_cv_w256`, the B1-vs-B2 separator.
`porosity_detector_disagreement` is a QC metric and is deliberately excluded pending a
check that it is not an acquisition artefact.
