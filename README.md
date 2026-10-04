# NEURA: SEM anode sample → batch classifier

Given one **sample** (one field of view imaged by three detectors, **BSE**, **ETD/SE** and **Inlens**), NEURA returns
P(Batch 1), P(Batch 2), P(Batch 3) with uncertainty ranges, a QC decision against **Batch 3 (the baseline)**, and the
physical reasons behind the call. Each sample is Si/graphite anode material; the three images of a sample are always used
together.

**Results summary:** [CLASSIFIER.md](CLASSIFIER.md). **Run the app:** `bash integration/run.sh` (UI + API on port 8000).

## Pipeline

```text
BSE + ETD/SE + Inlens (one sample)
 ├─ Rule branch     anode_microstructure_qc   thresholds + cleanup → pore / graphite / Si / binder / gap masks
 ├─ Learned branch  feature_classifier        frozen DINOv2-small + trained FusionHead → pore / graphite / Si / binder masks
 │                  (both)                    → same KPIs from each mask set (cracks from the ETD crack detector)
 ├─ Classifiers     bayesian_kpi_classifier   Bayesian Student-t on 3 KPIs per branch → P_rule, P_learned
 ├─ Combine                                   P ∝ sqrt(P_rule × P_learned)
 └─ App             integration               FastAPI + Svelte: probabilities, ranges, reasons, baseline QC, history
```

## Data

- **Dataset:** private Hugging Face dataset `gabrielgramicelli/NEURA-iterate-hack` (needs `HF_TOKEN`): 31 labelled samples,
  Batch 1: 7, Batch 2: 7, Batch 3: 17. `anode-qc download --dest data` fetches it.
- **Images:** ~2044 × 6994 px TIFFs, grey stored as 3 identical channels. There's no pixel-size metadata, so sizes are in px.
  The three detectors share one pixel grid (zero shift).
- **Test runs:** classified test samples (TIFFs, `result.json`, rendered images) plus `history.csv` are in
  [`integration/test_runs/`](integration/test_runs/README.md).

## 1. Preprocessing

- **Classifier path** (`anode_qc/data.py`, `preprocess.py`):
  1. Keep channel 0 and crop a 2 px border (corrupt edge columns).
  2. Gaussian denoise, σ = 1.5 px.
  3. Normalise each image so the pore floor → 0 and the graphite peak → 1, which removes brightness and contrast drift.
  4. ETD is normalised to graphite interiors.
- **Tile dataset path** (root `normalize.py`, `prepare_dataset.py`), a separate graphite-anchored normalisation:
  1. Graphite mask, then subtract the black level.
  2. Fit a smooth graphite-brightness surface to remove uneven lighting.
  3. Apply a fixed per-detector scale learned from the baseline folder (`profile.json`).
  4. Cut 512 × 512 tiles and export them as CSV.

  ```bash
  pip install numpy scipy pillow
  python3 prepare_dataset.py OUT_DIR Batch_3 Batch_1 Batch_2   # baseline folder first
  ```
- **DINO path:** per-image percentile stretch of each detector before the backbone (see 3).

## 2. Rule-based segmentation and KPIs (`anode_microstructure_qc/`)

1. **Si:** BSE hysteresis threshold anchored to the graphite peak (Si is bright, high Z).
2. **Cleanup:** remove edge rims, dim artefacts and porous mesh (solidity < 0.7).
3. **Pores and gaps:** from ETD (open pores are near-black), split into wide pores and thin gaps.
4. **Binder/CBD:** high texture across all three detectors. This step is experimental.
5. **KPIs** (`kpis.py`, spec in [`docs/KPI_SPEC.md`](anode_microstructure_qc/docs/KPI_SPEC.md)):
   - Si fraction of solids and porosity;
   - Si D50/D90;
   - Si heterogeneity `si_cv_w256` (CV of Si fraction over 256 px windows);
   - dispersion and clustering;
   - Si and graphite crack density (ETD dark-ridge skeleton length per 10⁴ px² of phase);
   - graphite aspect ratio (watershed-split particles, median of ellipse major ÷ minor axis);
   - alignment and Si contacts.
6. **Uncertainty and trust:**
   - pixel-ambiguity ranges and algorithm-choice Monte Carlo ranges;
   - trust flags (`low_si_contrast`, `charging`) mark measurement quality, not batch.

```bash
cd anode_microstructure_qc && pip install -e ".[dev]"
anode-qc run --data data --out outputs --mc-runs 20   # outputs/kpis/per_image.csv, outputs/report/report.md
```

## 3. DINO features and fine-tuned decoder (`feature_classifier/`)

1. **Backbone:** `facebook/dinov2-small`, **frozen**. It runs on each detector in 518 px tiles with 56 px overlap, giving
   384-channel features per 14 px patch, cached as `{image_id}.npy` (HWC, float16).
2. **Decoder (the only trained part):** `FusionHead` (< 0.5 M params):
   - DINO features projected to 64 channels and upsampled;
   - plus a 32-channel CNN on the raw detector pixels for sharp edges;
   - output: pore / graphite / Si / binder.
3. **Labels:** pseudo-labels from the rule segmentation, with a 2 px ignored boundary band. The decoder learns the rules;
   it is **not** validated against expert masks.
4. **Split:** 25 train / 3 val / 3 test images (one of each per batch), giving `fusion_holdout25.pt`.
   - Test mIoU vs rule masks: 0.81–0.85.
   - Per-class IoU: pore 0.98, graphite 0.93, Si 0.94, binder 0.47.
5. **Learned KPIs:** the same `anode_qc` KPI code applied to the learned masks
   (`bayesian_kpi_classifier/experiments/learned_masks/`).

```bash
python -m phaseseg.features --images IMG --out FEAT --tile 518 --overlap 56
python -m phaseseg.train --images IMG --features FEAT --labels pseudo --split holdout --head fusion --epochs 15 --out RUN
```

The alternative `anode_qc/dinoseg.py` (DINOv3/DINOv2 `FusionDecoder`) is experimental.

## 4. Classifier (`bayesian_kpi_classifier/`, package `neura_uq`)

- **Model:** `BayesianStudentTClassifier`, a generative per-batch Student-t on transformed KPIs (log/logit), diagonal
  covariance, uniform priors.
- **Outputs:**
  - posterior P(batch), a soft EM-like responsibility;
  - per-batch typicality p-value, with `outlier` and `ambiguous` flags;
  - additive per-KPI log-evidence (the "why").
- **KPIs per branch** (most often chosen in validation):
  - Rule: `frac_pore`, `graphite_crack_density`, `si_cv_w256`;
  - Learned: `frac_pore`, `graphite_aspect_ratio_median`, `si_cv_w256`.
- **Combined:** geometric mean of the two posteriors, renormalised. Each branch gets half a vote, so evidence they share
  isn't double-counted.
- **Confidence:** `high` if P ≥ 0.9, `medium` if P ≥ 0.6, `review` otherwise. Ranges are 5–95 % from 200 stratified
  bootstrap refits (bootstrap ranges, not Bayesian credible intervals).

**Validation** (`experiments/batch_cv/rotating_cv.py`): 17 rounds, each holding out 1 sample per batch, giving 51 held-out
predictions. KPI subset selection happens inside each round.

| Classifier | Correct / 51 | B1 | B2 | B3 | Samples / 31 |
|---|---|---|---|---|---|
| Rule | 37 | 0.71 | 0.53 | 0.94 | 25 |
| Learned (DINO masks) | 39 | 0.82 | 0.71 | 0.76 | 24 |
| **Combined** | **43** | **0.82** | **0.76** | **0.94** | **27** |

**What separates the batches:**
- **B3:** more porous, with more cracked graphite. Easy to separate.
- **B1 vs B2:** overlap. Weak signals:
  - Si heterogeneity (higher in B2);
  - DINO-mask graphite crack density (higher in B2);
  - Si–graphite contact.

**Things that did not help:** per-KPI weights, two-stage classifiers, LDA and tile-level classifiers all overfit. Details in
[`experiments/research/README.md`](bayesian_kpi_classifier/experiments/research/README.md).

## 5. App (`integration/`)

- **Backend:** `integration/backend/neura_app`, FastAPI.
  - `pipeline.py` runs the rule branch and the DINO branch in parallel, renders the masks, cracks and KPI explanation images.
  - `model.py` holds the final classifiers, fit on all 31 samples. The decoder is reused, not retrained.
  - Endpoints: `/api/classify`, `/api/jobs/{id}`, `/api/history`, `/api/history.csv`, `/api/overview`.
  - Every run is saved under `$NEURA_JOBS_DIR/<job_id>/`.
- **Frontend:** `integration/frontend`, Svelte 5 + TypeScript. Tabs:
  - **Classify:** upload 3 TIFFs to get probabilities and ranges, the rule-vs-DINO split, KPI reasons, and a **QC decision**
    vs the baseline (Accept if P(B3) ≥ 0.7 and its range stays above 0.5; Reject if ≤ 0.3 and below 0.5; otherwise
    Investigate). Also shows scatter plots with batch regions and masks next to a representative sample per batch.
  - **History:** reopen past runs, CSV export.
  - **Overview:** validation results, feature charts, pipeline explanation.
- **Run:** `bash integration/run.sh`. About 1.5–3 min per sample on CPU.

  | Variable | Default |
  |---|---|
  | `NEURA_DATA_DIR` | `/home/ubuntu/data/neura` |
  | `NEURA_JOBS_DIR` | `/home/ubuntu/data/runs/app_jobs` |
  | `PORT` | `8000` |
  | `HF_TOKEN` | needed only to download the dataset |
- **GPU deploy:** Hugging Face Space (Docker, CUDA) in `integration/deploy/hf_space/`.

## Repository map

| Path | What |
|---|---|
| `normalize.py`, `prepare_dataset.py` | graphite-anchored normalisation + 512 px tile export |
| `anode_microstructure_qc/` | rule segmentation, KPIs, uncertainty, QC report (`anode-qc` CLI) |
| `feature_classifier/` | DINOv2 feature cache, FusionHead/linear decoders, training/eval, Modal app |
| `bayesian_kpi_classifier/` | `neura_uq` classifier, CV experiments, learned-mask KPIs, research |
| `integration/` | backend, frontend, artifacts (`fusion_holdout25.pt`, KPI tables), test runs, HF deploy |
| `CLASSIFIER.md` | one-page classifier summary |

## Limitations

- **Small data:** 7 / 7 / 17 samples. B1 vs B2 calls are fragile, so confirm them on new samples.
- **Pseudo-labels:** the learned masks copy the rule masks, so expert masks are needed to validate them.
- **Units:** sizes are in px until a pixel size is known.
- **Probabilities are relative:** they're relative to the three known batches. Check the typicality and outlier flags for
  samples that fit none of them.
