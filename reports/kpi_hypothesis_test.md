# Hypothesis test: "a KPI vector is an interpretable embedding that can assign a sample to batch 1/2/3"

Date: 2026-10-03. Branch `devin/1791039901-kpi-hypothesis`. Data: the three labelled
Drive batches (Batch_1 = 7 samples, Batch_2 = 7, Batch_3 = 17; one Batch-3 sample,
`img_xgj4xftb`, is missing its InLens view because the Drive download of that file kept failing).

## What was tested

Exactly the procedure in the hypothesis:

1. `kpis.py` extracts 26 material KPIs per sample (BSE: pore fraction, bright-phase
   fraction, pore/particle size d50, pore elongation, solid chord lengths H/V and
   their ratio, orientation coherence, dominant angle, phase contrast; SE: dark
   fraction/size/count, edge density, roughness; InLens: bright-edge fraction and
   skeleton length, dark fraction, edge density, coherence) plus 18 `qa__` imaging
   descriptors (size, mean/std grey, noise sigma per view). Units are px / fractions:
   the TIFFs carry no pixel size.
2. `evaluation.py` holds out one whole sample at a time, fits median/MAD scaling and
   per-batch median profiles on the other 30, assigns the held-out sample to the
   closest profile, records the gap to the runner-up, and lists mistakes and drivers.
3. `kpi_separation.py` checks, per KPI, whether within-batch spread is smaller than
   the between-batch differences.

## Result: not supported with these KPIs

| Feature set | LOO accuracy | Label-shuffle chance (mean / 95th pct) | Flagged low-confidence (rel. gap < 0.15) |
|---|---|---|---|
| 26 material KPIs, profile distance | **0.61** (19/31) | 0.37 / 0.57 | 17/31 |
| 26 material KPIs, diagonal Gaussian | 0.65 (20/31) | 0.39 / 0.55 | 17/31 |
| 18 imaging `qa__` descriptors only | 0.45 | 0.35 / 0.57 | 17/31 |
| KPIs + qa (44) | 0.58 | 0.39 / 0.55 | 15/31 |
| Batch 1 vs 2 only, 26 KPIs (7 + 7) | **0.50** | 0.51 / 0.79 | 12/14 |

Confusion matrix (material KPIs, rows = true, cols = predicted):

```
      1  2   3
1     2  5   0
2     3  3   1
3     0  3  14
```

- Batch 1 and Batch 2 are indistinguishable: 5 of 7 Batch-1 samples go to Batch 2 and
  3 of 7 Batch-2 samples go to Batch 1. With only these two batches the accuracy is
  exactly chance.
- Batch 3 is mostly recognisable (14/17), which is where all the above-chance signal
  comes from. Both scorers (0.61, 0.65) sit just above the 95th percentile of the
  shuffled-label distribution (0.55-0.57): real but marginal evidence.
- 17 of 31 assignments have a relative gap < 0.15 between best and second-best batch,
  so by the hypothesis's own rule most calls are "low confidence".
- Robustness: re-extracting KPIs with a different downsampling factor (2 -> 3) changes
  the assigned batch for 3/31 samples and accuracy drops to 0.55. Assignments are not
  stable under a trivial preprocessing change.

### Per-KPI separation (`kpi_separation.csv`, effect size = between-batch range of medians / pooled within-batch MAD)

| KPI | effect size | Kruskal p | univariate LOO acc | median B1 / B2 / B3 |
|---|---|---|---|---|
| InLens dark fraction | 5.4 | 0.011 | 0.39 | 0.202 / 0.208 / 0.508 |
| BSE solid chord, vertical (px) | 2.9 | 0.007 | 0.65 | 212 / 196 / 186 |
| InLens bright-edge skeleton length (px/Mpx) | 2.0 | 0.032 | 0.71 | 16.7k / 15.6k / 13.4k |
| SE dark-region d50 (px) | 1.9 | 0.043 | 0.58 | 135 / 155 / 183 |
| SE dark fraction | 1.3 | 0.002 | 0.68 | 0.119 / 0.124 / 0.159 |
| BSE pore fraction | 0.5 | 0.086 | 0.45 | 0.091 / 0.096 / 0.099 |
| BSE bright-phase fraction | 0.5 | 0.429 | 0.19 | 0.067 / 0.060 / 0.060 |
| BSE pore d50 (px) | 0.4 | 0.567 | 0.13 | 117 / 127 / 117 |

Every KPI with a between-batch effect places Batch 3 apart; none separates Batch 1
from Batch 2. The physically most interpretable KPIs (pore fraction, bright-phase
fraction, pore size d50) show no batch effect at all; 26 features at 31 samples also
means the uncorrected p-values above are not strong evidence.

### The Batch-3 signal is confounded with acquisition

Imaging descriptors differ between batches as much as the material KPIs do: InLens
noise sigma (4.4 / 3.7 / 3.0 8-bit, Kruskal p = 0.001), SE noise sigma (p = 0.001), SE
mean grey (p = 0.006); imaging descriptors alone predict the batch at 0.45. Within Batch 3 the InLens images fall into visibly different
settings (mean grey 65-137, noise 1.9-4.4) and `InLens dark fraction` tracks InLens
mean grey (Spearman rho = -0.67) and noise (-0.61); 15 of the 26 material KPIs have
|rho| > 0.5 with at least one imaging descriptor. A third of Batch 3 shows InLens
platelet interiors as nearly black with bright rims (see `img_kbdh4tri`), which could
be material (conductivity / surface chemistry) or detector settings; the data cannot
tell, and a classifier trained on it would partly learn the microscope session.

## What this means for the team

- The hypothesis's own success criterion ("KPIs vary less within each batch than they
  differ between batches") fails for Batch 1 vs 2 and is only weakly, and confoundedly,
  met for Batch 3 with this first KPI set. The method is sound; the measurements do not
  yet carry the information.
- Treat `qa__` descriptors as a gate: report imaging drift separately from material
  drift, otherwise a baseline-vs-incoming comparison will flag detector changes as
  material changes.
- Where better measurement is needed: pore/bright-phase fractions are nearly identical
  across batches at this segmentation, so either the batches genuinely do not differ in
  those properties or finer KPIs are needed (platelet size distribution from a proper
  particle segmentation, crack/delamination counts, binder/carbon-black texture at full
  resolution). This is the gap an encoder embedding would have to fill, evaluated through
  the same `evaluation.py` splits.
- Ask the organisers for pixel size and acquisition settings per image, and whether
  Batch 3 was imaged in more than one session.

## Reproduce

```bash
python build_manifest.py --data-dir <drive folder> --out sample_manifest.csv
python kpis.py --manifest sample_manifest.csv --out features_real.csv
python evaluation.py --features features_real.csv --feature-prefix kpi__ --out eval_real --shuffles 50
python evaluation.py --features features_real.csv --feature-prefix qa__  --out eval_real_qa
python evaluation.py --features features_real.csv --feature-prefix kpi__ --method gaussian --out eval_real_gauss
python kpi_separation.py --features features_real.csv --out kpi_separation.csv
```
