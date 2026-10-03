# Microscopy sample batch assignment

Assign each of nine unseen **samples** to one of three known batches (1, 2, or 3). 
A sample contains three microscopy views/channels. 
The priority is a repeatable prediction with physical reasons for each assignment.

## Architecture

```text
Labelled samples (batches 1/2/3) + nine unlabelled samples
    -> preprocessing.py -> processed three-view samples + IDs
    -> kpis.py -> physically meaningful features per sample
       [optional: encoder.py -> pretrained embeddings per sample]
    -> comparison.py -> score each sample against all three batch profiles
       [optional: em_model.py -> per-batch likelihoods, if data supports it]
    -> qc_report.py -> predicted batch, three scores, margin, KPI reasons

evaluation.py: hold out complete samples to check accuracy and stability
analyse_batch.py: run the same fixed pipeline on all nine test samples
```

**Core method:** build a reference profile for each known batch from training samples, then assign each test sample to the closest profile using a small set of physically grounded KPIs. Normalize features using training data only. Show which channels and measurements favoured the winning batch over the runner-up. The encoder and EM branches are supporting evidence only if whole-sample validation shows they help.

## Shared contract: one row means one sample

- **Input manifest:** start from [sample_manifest.template.csv](sample_manifest.template.csv). It has `sample_id`, optional `batch_id`, and three named view paths; add acquisition metadata columns (especially pixel size/magnification) when available. `batch_id` is 1/2/3 for reference samples and blank for test samples. Do not assume the nine test samples are evenly divided across batches.
- **Processed sample:** keep the three views separately under the same `sample_id`; stack into `[N, 3, H, W]` only if they are spatially compatible. Patches retain their parent `sample_id`.
- **Feature table:** exactly one row per sample, keyed by `sample_id`, with named, unit-bearing KPI columns. Aggregate any patch measurements within their parent sample. An encoder, if used, returns one embedding per sample under the same key.
- **Prediction table:** one row per test sample with `predicted_batch`, three batch distances/scores, winner-versus-runner-up margin, `review_flag`, and the top physical feature contributions. A small margin means ambiguity; it is not automatically a calibrated probability.

## First working baseline

1. Check that IDs are unique, each sample has all three expected views, labels are valid, and acquisition settings are recorded. The KPI owner then selects a **small** set of measurements that make physical sense for each view.
2. Fit preprocessing and feature scaling on labelled reference samples only. Build a robust reference profile (for example, a median KPI vector) for each batch. Give each view an explicit weight so a view with more KPI columns does not dominate by accident.
3. Compute a transparent, scaled distance from each test sample to each batch profile. Choose the smallest distance and report the feature contributions that separate it from the runner-up.
4. Refit this entire process while holding out one **whole sample** at a time. Record predictions, confusion matrix, and margins. If there are too few labelled samples per batch to do this, report that limit instead of claiming validated accuracy.
5. Freeze the feature list, preprocessing settings, weights, and model versions before classifying the nine unseen samples. Save the settings and predictions together so a second run can reproduce them.

The EM and encoder owners can work in parallel against the same sample IDs and feature table. Add either branch to the final score only if whole-sample validation improves results or resolves a clear failure of the KPI baseline. Do not train a transformer from scratch on this dataset.

These Python files are interface placeholders only; no pipeline is implemented yet.

## Evaluation harness (`evaluation.py`)

Sample-level leave-one-out: every labelled sample (all its views and patches,
already aggregated into one row) is held out in turn; feature scaling and one
profile per batch are fitted on the remaining samples only; the held-out sample
is scored against every batch.

```bash
pip install -r requirements.txt
python make_fake_features.py                 # temporary KPI table -> features_fake.csv
python evaluation.py --features features_fake.csv --out eval_out
python -m pytest tests
```

Input: a CSV with `sample_id`, `batch_id` (blank for unseen test samples, which
are ignored) and numeric feature columns. Columns named
`kpi__<view>__<name>__<unit>` are grouped so each view carries equal total weight
(`--view-weights BSE=1,SE=1,InLens=0.5` to change). Any other numeric columns
(e.g. `emb_0..emb_N` from `encoder.py`) are accepted as-is, so swapping the
temporary CSV for real KPIs or embeddings needs no code change.

Outputs in `--out`: `predictions.csv` (true/predicted batch, `score_batch_*`,
margin, `review_flag` for small margins, top contributing features),
`confusion_matrix.csv`, `mistakes.csv`, `summary.json` (accuracy, counts per
batch, warnings, label-shuffle chance baseline).

A batch that would have fewer than `--min-train-per-batch` (default 2) samples
left after holding one out is reported in `warnings`, and its samples are marked
`evaluable=False` rather than scored against a profile they cannot have.

Scorers are pluggable via `--method`: `profile` (robust-scaled distance to the
per-batch median) and `gaussian` (diagonal Gaussian, shrunk variances; the
stand-in for the EM branch). A new approach only needs `fit(X, y)`,
`scores(x)` and `contributions(x, a, b)`.
