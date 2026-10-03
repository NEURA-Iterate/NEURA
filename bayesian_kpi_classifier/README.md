# neura-uq

`neura-uq` assigns an SEM sample to one of several reference batches using a Bayesian
Student-t classifier. KPI values can be transformed (for example, fractions with a
logit and sizes with a log), and uncertainty from measurement SDs or perturbed
segmentation runs is included in prediction. Batch profiles are regularized toward
shared pooled covariance, which is useful when each batch has only a few reference
samples.

## CSV formats

The reference CSV requires `batch` and `sample_id` columns, followed by numeric KPI
columns. Any KPI may have a corresponding `<kpi>_sd` column. When reference SD columns
are present, their uncertainty is used to estimate how much of the observed reference
spread is measurement noise.

A one-row sample CSV contains the KPI values and may include `<kpi>_sd` columns:

```csv
si_frac,porosity,d50,agglom,si_frac_sd,porosity_sd,d50_sd,agglom_sd
0.122,0.275,5.55,0.57,0.006,0.012,0.08,0.035
```

A sample CSV with multiple rows instead represents repeated perturbed-segmentation
runs; each row is one run and each KPI column is a measurement. The classifier uses
the transformed run values to estimate the sample's measurement uncertainty.

## CLI

From the repository root, install the project and optional development tools with:

```bash
cd bayesian_kpi_classifier
python -m pip install -e '.[dev]'
```

Classify using the example CSVs:

```bash
neura-classify \
  --reference examples/reference_example.csv \
  --sample examples/sample_example.csv \
  --kpis si_frac,porosity,d50,agglom \
  --transforms logit,logit,log,log \
  --covariance diag \
  --prior-strength 2 \
  --loo
```

If omitted, KPIs default to all non-metadata, non-SD columns from the reference file,
transforms default to `identity`, covariance defaults to `full`, and prior strength
defaults to `2`. Add `--json` for probabilities, typicality p-values, flags, and
per-KPI evidence as JSON.

## Python

```python
import numpy as np
from neura_uq import BayesianStudentTClassifier, Measurement

X = np.array([[0.10, 5.0], [0.11, 5.3], [0.15, 6.4], [0.14, 6.1]])
labels = ["batch_a", "batch_a", "batch_b", "batch_b"]
model = BayesianStudentTClassifier(
    kpis=["si_frac", "d50"],
    transforms=["logit", "log"],
    covariance="diag",
).fit(X, labels)
prediction = model.predict(Measurement.from_sd([0.11, 5.2], [0.004, 0.1]))
print(prediction.summary())
```

## Interpreting the output

- The posterior is relative to the supplied batches and always sums to 1; it does not
  establish that the sample belongs to any of them.
- Typicality p-values answer “does it fit at all?” for each batch. They complement the
  posterior, which will always select relative probabilities even for a poor fit.
- Per-KPI evidence is exactly additive only with `covariance="diag"` and diagonal
  measurement noise. With full covariance, KPI contributions should not be interpreted
  as an exact additive decomposition.
- Typicality p-values use a moment-matched approximation to the predictive distribution.
- `rank_kpi_subsets` selects and evaluates subsets on the same data, so its best score
  is optimistic.
- With about seven samples per batch, check probability calibration and separation with
  leave-one-out evaluation rather than relying on a single predicted probability.

Run `python examples/synthetic_demo.py` for a reproducible demonstration, including
uncertain measurements, an outlier, leave-one-out metrics, KPI separation, and ranked
KPI subsets.
