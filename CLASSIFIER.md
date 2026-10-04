# Batch classifier: summary

**Goal:** given a new sample (one location with its BSE, ETD and Inlens images), give the probability that it belongs to each batch, P(B1), P(B2) and P(B3).

**Best solution: the combined classifier.**

## How it works

1. **Two sets of phase masks** (Si, graphite, pore and binder) for each sample:
   - **Rule masks:** rule-based segmentation (`anode_microstructure_qc/`).
   - **Learned masks:** frozen DINOv2-small plus a small trained decoder (`feature_classifier/`).
2. **KPIs:** the same 12 KPIs are computed from each set of masks, giving one row per sample. Crack density uses the ETD crack detector in both cases.
3. **Two classifiers:** a Bayesian Student-t classifier is trained on each KPI table (`bayesian_kpi_classifier/`). Each one picks its own best 3 KPIs, usually:
   - Rule: porosity, graphite crack density and Si heterogeneity.
   - Learned: porosity, graphite aspect ratio and Si heterogeneity.
4. **Combine:** P = sqrt(P_rule × P_learned), rescaled to sum to 1.

The rule classifier picks out B3 well through cracks; the learned classifier separates B1 from B2 better through graphite shape. Combining them keeps both strengths.

## Results

Each of the 17 validation rounds holds out 1 sample per batch, which gives 51 held-out predictions.

| Classifier | Correct (of 51) | B1 | B2 | B3 | Samples correct (of 31) |
|---|---|---|---|---|---|
| Rule KPIs | 37 | 0.71 | 0.53 | 0.94 | 25 |
| Learned KPIs | 39 | 0.82 | 0.71 | 0.76 | 24 |
| **Combined** | **43** | **0.82** | **0.76** | **0.94** | **27** |

## What separates the batches

- **B3:** more porous and more cracked graphite.
- **B1 vs B2:** they overlap; Si heterogeneity and graphite shape separate them, weakly.

## Caveats

- Small data: 7, 7 and 17 samples. Confirm on new samples.
- The learned masks were trained to copy the rule masks; expert annotations are needed to validate them.

## Where things are

- Combined classifier and research: `bayesian_kpi_classifier/experiments/research/` (`final_model.py`, `README.md`)
- Learned masks: `feature_classifier/README.md`
- Classifier package: `bayesian_kpi_classifier/README.md`
