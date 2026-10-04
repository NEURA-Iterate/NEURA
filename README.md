# NEURA MicroNet microscopy batch classifier

This repository prepares, trains, evaluates, and serves a three-class microscopy classifier for the known specimen batches **Batch 1**, **Batch 2**, and **Batch 3**. Each complete cross-section has BSE, ETD, and Inlens TIFFs. The model is a ResNet50 initialized from NASA MicroNet weights and trained in two stages on Modal.

## Data inventory and labels

The authoritative class label is the batch folder, not an image filename or TIFF annotation.

| Raw folder | Groups | Complete BSE/ETD/Inlens groups | Excluded SE groups | Class ID | Train / val / test groups |
|---|---:|---:|---:|---:|---:|
| `Batch_1` | 7 | 7 | 0 | 0 | 4 / 1 / 2 |
| `Batch_2` | 7 | 6 | 1 | 1 | 4 / 1 / 1 |
| `Batch_3` | 17 | 14 | 3 | 2 | 8 / 3 / 3 |
| **Total** | **31** | **27** | **4** | — | **16 / 5 / 6** |

The four incomplete groups have an `_SE.tif` file where `_ETD.tif` is required; they are inventoried but excluded from staging/training. They are `Batch_2/img_rxax5ozo` and `Batch_3/img_utfgcjfa`, `Batch_3/img_vc2whyaq`, and `Batch_3/img_x77cy643`. SE is not treated as an ETD substitute.

| Folder | Class ID | User-facing class |
|---|---:|---|
| `Batch_1` | 0 | Batch 1 |
| `Batch_2` | 1 | Batch 2 |
| `Batch_3` | 2 | Batch 3 |

Raw images are uint8 RGB TIFFs with identical channels except for occasional corrupted edge columns. Widths are 6960, 6996, or 7000 pixels; heights range from 1612 to 2316 pixels. Preparation uses non-overlapping 512×512 tiles from the top-left; edge remainders are discarded. The dimensions therefore yield either **39 or 52 crops per detector image** (117 or 156 CSV crops per complete three-detector set).

Splits are group-isolated, seeded with 42, and fixed at **Batch 1: 4/1/2, Batch 2: 4/1/1, Batch 3: 8/3/3** (train/val/test). The split manifest is `artifacts/split_manifest.csv`; its SHA256 is `578bb7c103a6cf8671bd690549e6b07c5b8dabaece9bc4662a1ac2c4ee98c7d5`.

The shared normalization calibration set is **`img_9luzk4jm`**, the first sorted complete Batch 3 train set:

| Detector | Profile value |
|---|---:|
| BSE | 0.44326052069664 |
| ETD | 0.37828272581100464 |
| Inlens | 0.4544069170951843 |

The profile SHA256 is `118f8e474622d6bd67d60042b997d5948201a65ede0dc1f49e4b603ec383acab`. Its source hashes, staged paths, preparation script hashes, and provenance are in `artifacts/run-20261004-100739/profile_provenance.json`.

MicroNet weights are downloaded from [`jstuckner/microscopy-resnet50-micronet`](https://huggingface.co/jstuckner/microscopy-resnet50-micronet), revision `c810b294acda2461775542841e6d0ba824ec46a9`. The loaded weights SHA256 is `4c906df94c224efa267e59311244900a5633549aeda5e11f5ecfade6f30bafd2`.

## Pipeline

1. **Inventory, split, and stage:** inventory/hash source detector TIFFs locally, assign complete cross-section groups to the fixed splits, and copy only complete sets into neutral `Batch 1`/`Batch 2`/`Batch 3` folders on `neura-data`.
2. **Prepare:** run the existing, unchanged `prepare_dataset.py`. It normalizes each whole three-detector set before tiling. Batch 3 train is processed first to fit the shared profile; validation and test reuse it. Profile provenance is stored under `/runs/<run_id>/`.
3. **Crop metadata/cache:** validate crop filenames, shape, finite range and decimal precision; check floor-based crop counts; write per-split `crop_manifest.csv`, float32 `crops.npy`, completion markers, and train previews.
4. **Stage A — SimSiam:** adapt the MicroNet encoder on **Batch 3 train only**, select a non-collapsed checkpoint using Batch 3 validation, and export the encoder.
5. **Stage B — classifier:** train three-way classifiers on Batch 1/2 plus Batch 3 rehearsal. Warmup freezes the encoder; fine-tuning unfreezes the configured parts. Encoder BatchNorm running statistics remain frozen.
6. **Selection and locked test:** compare both initializations on validation using equal-weighted group-mean crop loss; save a frozen `model_selection.json`. Evaluate both models on the mixed test split only after selection.
7. **Raw DA-v1 and live demo:** separately evaluate the selected model on raw images, measure parity with prepared crops, and run neutral-filename raw/CSV live predictions before joining labels.

### V1 and V2 configurations

| Configuration | Stage A | Stage B | GPU / input preprocessing |
|---|---|---|---|
| V1 | 10 epochs, batch 8 pairs, train `input_conv` + `layer4` | 5 warmup + up to 25 fine-tune epochs; microbatch 4, accumulation 4; `full_encoder=False` | L4; `InputPrep("none")` |
| V2 (`tag=v2`) | 20 epochs, batch 32 pairs, all encoder layers trainable | 5 warmup + up to 40 fine-tune epochs; microbatch 16, accumulation 1; full encoder, patience 10 | B200; ImageNet normalization in both stages |

Both use seed 42, the same balanced crop sampler and declared spatial augmentations. BN running statistics remain frozen. V2 writes to separate `stage_a_v2/` and `stage_b_*_v2/` directories; it does not overwrite V1. Stage B refuses a Stage A export whose `input_prep` mode differs.

### Raw mode is a separate variant

The official `normalize.py` path **requires the BSE companion image** to construct its graphite mask and uses a separate scale per detector. A lone BSE/ETD/Inlens TIFF cannot reproduce that path. Single-image raw mode therefore uses the explicitly separate, detector-agnostic **DA-v1** variant: it fits a pooled calibration scale and computes the graphite mask from that image itself. It is not equivalent to prepared-CSV mode; measured raw/prepared parity is poor (see results).

## Tested commands

The local environment used Python 3.13, Modal 1.6.1, NumPy 2.4.3, pandas 3.0.1, Pillow 12.2.0, SciPy 1.15.1, pytest 8.4.2, and a local CPU Torch 2.11/torchvision 0.26 pair for model-layer tests. The pinned `requirements-modal.txt` targets the Modal Python 3.11 image; installing it locally is **not** required. Data tooling needs Modal, NumPy, pandas, Pillow, SciPy, and pytest; model-layer tests additionally import Torch/torchvision, scikit-learn, and matplotlib.

The local dependency install and test commands run in this session were:

```bash
.venv/Scripts/python.exe -m pip install modal numpy pandas pillow scipy pytest
.venv/Scripts/python.exe -m pytest -q tests
```

Result: **15 passed**. The model tests used the compatible local CPU Torch/torchvision noted above; Modal uses the pinned image versions below.

These Modal execution commands were run (UTF-8 is needed for Modal’s console output on this Windows host):

```bash
PYTHONUTF8=1 modal run modal_app.py::upload
PYTHONUTF8=1 modal run modal_app.py::split
PYTHONUTF8=1 modal run --detach modal_app.py::prepare_cli --run-id run-20261004-100739
PYTHONUTF8=1 modal run modal_app.py::smoke --run-id run-20261004-100739
PYTHONUTF8=1 modal run modal_app.py::smoke --run-id run-20261004-100739 --gpu B200 --tag b200
PYTHONUTF8=1 modal run --detach modal_app.py::pipeline --run-id run-20261004-100739
PYTHONUTF8=1 modal run --detach modal_app.py::pipeline --run-id run-20261004-100739 --tag v2 --gpu B200
PYTHONUTF8=1 modal run --detach modal_app.py::live_demo_cli --run-id run-20261004-100739 --model stage_b_micronet --n 12 --seed 42
PYTHONUTF8=1 modal run modal_app.py::fetch --run-id run-20261004-100739
```

The following standalone entrypoint parsers were checked with `--help`:

```bash
PYTHONUTF8=1 modal run modal_app.py::adapt --help
PYTHONUTF8=1 modal run modal_app.py::train --help
PYTHONUTF8=1 modal run modal_app.py::eval_cli --help
PYTHONUTF8=1 modal run modal_app.py::eval_raw --help
PYTHONUTF8=1 modal run modal_app.py::predict --help
PYTHONUTF8=1 modal run modal_app.py::live_demo_cli --help
PYTHONUTF8=1 modal run modal_app.py::fetch --help
PYTHONUTF8=1 modal run modal_app.py::prepare_cli --help
```

The stand-alone Stage A/Stage B/evaluation jobs were executed as child functions of the pipeline; the v1 and v2 pipelines above ran through evaluation, raw parity, and live demo. Their corresponding single-stage invocation forms are:

```bash
PYTHONUTF8=1 modal run --detach modal_app.py::adapt --run-id RUN_ID [--resume] [--tag v2] [--gpu B200]
PYTHONUTF8=1 modal run --detach modal_app.py::train --run-id RUN_ID --init stage_a|stage_a_v2|micronet [--resume] [--tag v2] [--microbatch 16 --accum 1 --full-encoder --input-prep imagenet] [--gpu B200]
PYTHONUTF8=1 modal run --detach modal_app.py::eval_cli --run-id RUN_ID --model MODEL --split val|test [--gpu B200]
PYTHONUTF8=1 modal run --detach modal_app.py::eval_raw --run-id RUN_ID --model MODEL [--tag v2] [--gpu B200]
PYTHONUTF8=1 modal run --detach modal_app.py::predict --run-id RUN_ID --model MODEL --path FILE --mode csv|raw [--gpu B200]
PYTHONUTF8=1 modal run --detach modal_app.py::live_demo_cli --run-id RUN_ID --model MODEL --n 12 --seed 43 [--tag v2] [--gpu B200]
PYTHONUTF8=1 modal run modal_app.py::fetch --run-id RUN_ID [--with-checkpoints]
```

`NEURA_GPU` sets the Modal GPU default at app import time (default `L4`); per-invocation `--gpu` overrides it, and the pipeline forwards the selected GPU to its GPU child functions. Stage A and Stage B expose `--resume`. The v2 pipeline command above was run with the declared B200 configuration; no additional tuned configuration was run.

A single CSV prediction returns `predicted_batch`, a `probabilities` mapping for Batch 1/2/3, `n_crops: 1`, `mode: "csv"`, and `warning: null`. Raw mode adds tile positions, per-crop probabilities, a `fallback_used` flag, and a warning identifying DA-v1 and the parity report.

## Results

All values below are from the fetched JSON/JSONL files under `artifacts/run-20261004-100739/`; paths are given in each subsection. Confusion matrices use rows=true and columns=predicted in Batch 1/2/3 order.

### Stage A and MicroNet loading

| Config | Best epoch | Best val loss | z_std range | Off-diagonal cosine range | Collapse |
|---|---:|---:|---:|---:|---|
| V1 | 5 | −0.967667 | 0.02026–0.02155 | 0.04175–0.15320 | None |
| V2 | 10 | −0.982515 | 0.01529–0.02139 | 0.04061–0.50870 | None |

V2 trained all 23,508,032 encoder parameters while keeping BN running stats frozen. Its z standard deviation remained above the collapse threshold (`0.5/sqrt(2048) ≈ 0.01105`), and off-diagonal cosine remained below 0.9.

| Epoch | V2 train loss | V2 val loss | z_std | Offdiag cosine |
|---:|---:|---:|---:|---:|
| 1 | −0.7600 | −0.8820 | 0.02010 | 0.1605 |
| 2 | −0.9140 | −0.9356 | 0.02117 | 0.0655 |
| 3 | −0.9452 | −0.9563 | 0.02139 | 0.0406 |
| 4 | −0.9210 | −0.9164 | 0.02131 | 0.0606 |
| 5 | −0.9398 | −0.9209 | 0.02116 | 0.0498 |
| 6 | −0.9163 | −0.9205 | 0.02028 | 0.1279 |
| 7 | −0.9599 | −0.9648 | 0.02040 | 0.0947 |
| 8 | −0.9357 | −0.8614 | 0.01964 | 0.1575 |
| 9 | −0.9340 | −0.9741 | 0.01846 | 0.2588 |
| 10 | −0.9817 | **−0.9825** | 0.01693 | 0.4080 |
| 11 | −0.9866 | −0.9824 | 0.01770 | 0.3551 |
| 12 | −0.9294 | −0.9069 | 0.02096 | 0.0579 |
| 13 | −0.9013 | −0.9180 | 0.02013 | 0.1555 |
| 14 | −0.9326 | −0.9169 | 0.01985 | 0.1284 |
| 15 | −0.9503 | −0.9450 | 0.02017 | 0.1304 |
| 16 | −0.9687 | −0.9758 | 0.02112 | 0.0752 |
| 17 | −0.9783 | −0.9776 | 0.01982 | 0.1811 |
| 18 | −0.9301 | −0.9368 | 0.01529 | 0.5087 |
| 19 | −0.9219 | −0.9552 | 0.01704 | 0.3795 |
| 20 | −0.8894 | −0.8887 | 0.01847 | 0.1907 |

The pinned MicroNet checkpoint loaded strictly: 318 tensors, zero non-FC missing/unexpected keys, and exact checks on `conv1.weight`, `layer1.0.conv1.weight`, and `layer4.2.conv3.weight`. Weight SHA256: `4c906df94c224efa267e59311244900a5633549aeda5e11f5ecfade6f30bafd2`. Reports: `stage_a/summary.json`, `stage_a/metrics.jsonl`, `stage_a_v2/summary.json`, `stage_a_v2/metrics.jsonl`.

### Stage B model cards and validation selection

| Config/model | Selected epoch | Card val group loss | Val source-image accuracy / macro-F1 | Val accuracy BSE / ETD / Inlens |
|---|---:|---:|---:|---:|
| V1 `stage_b_stage_a` | 11 | 0.914021 | 0.533 / 0.389 | 0.60 / 0.60 / 0.40 |
| V1 `stage_b_micronet` | 21 | 0.886578 | 0.533 / 0.329 | 0.80 / 0.40 / 0.40 |
| V2 `stage_b_stage_a_v2` | 32 | 0.458695 | 0.867 / 0.750 | 0.80 / 1.00 / 0.80 |
| V2 `stage_b_micronet_v2` | 19 | 0.498635 | 0.867 / 0.750 | 0.80 / 0.80 / 0.80 |

Each card records the runtime GPU, Torch/CUDA versions, preprocessing mode, and config. Per-epoch selection metrics are in each model’s `metrics.jsonl`. V1 VAL selected `stage_b_micronet` (0.885986 vs 0.913860); V2 VAL selected **`stage_b_stage_a_v2`** (0.458766 vs 0.499203). Selection used VAL only.

### Locked TEST results

| Config/model | Accuracy | Balanced accuracy | Macro-F1 | Recall Batch 1 / 2 / 3 | Overall confusion matrix |
|---|---:|---:|---:|---:|---|
| V1 `stage_b_stage_a` | 0.500 | 0.333 | 0.231 | 0.000 / 0.000 / 1.000 | `[[0,1,5],[0,0,3],[0,0,9]]` |
| V1 `stage_b_micronet` | 0.667 | 0.500 | 0.472 | 0.500 / 0.000 / 1.000 | `[[3,0,3],[0,0,3],[0,0,9]]` |
| V2 `stage_b_stage_a_v2` | 0.389 | 0.278 | 0.296 | 0.167 / 0.000 / 0.667 | `[[1,4,1],[2,0,1],[2,1,6]]` |
| V2 `stage_b_micronet_v2` | 0.556 | 0.426 | 0.428 | 0.500 / 0.000 / 0.778 | `[[3,3,0],[2,0,1],[2,0,7]]` |

Per-detector image accuracy:

| Config/model | BSE | ETD | Inlens |
|---|---:|---:|---:|
| V1 Stage A init | 0.500 | 0.500 | 0.500 |
| V1 MicroNet init | 0.667 | 0.500 | 0.833 |
| V2 Stage A init | 0.333 | 0.333 | 0.500 |
| V2 MicroNet init | 0.500 | 0.667 | 0.500 |

The full JSONs contain per-detector balanced accuracy, macro-F1, per-class precision/recall/F1/support, and labelled confusion matrices:
`evaluation/stage_b_stage_a/test/metrics.json`, `evaluation/stage_b_micronet/test/metrics.json`, `evaluation/stage_b_stage_a_v2/test/metrics.json`, and `evaluation/stage_b_micronet_v2/test/metrics.json`.

| Config/model | 95% accuracy CI | 95% macro-F1 CI |
|---|---:|---:|
| V1 Stage A init | [0.500, 0.500] | [0.222, 0.240] |
| V1 MicroNet init | [0.611, 0.722] | [0.407, 0.528] |
| V2 Stage A init | [0.111, 0.611] | [0.083, 0.419] |
| V2 MicroNet init | [0.278, 0.833] | [0.222, 0.602] |

All intervals use 2,000 test-group bootstrap samples, stratified within batch, seed 42. The test has six independent groups (Batch 1=2, Batch 2=1, Batch 3=3), so uncertainty is substantial. The majority baseline is **Batch 3**, with train crop counts B1=585, B2=624, B3=1,131; test image accuracy **0.500** and macro-F1 **0.222**. Each model’s five neutral-filename checks passed with identical argmax and max probability difference 0.0 (tolerance `1e-4`).

### Raw DA-v1 parity, raw evaluation, and live demo

Raw evaluation uses the validation-selected model for each version: V1 MicroNet init and V2 Stage A init.

| Version | Raw variant | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| V1 | Whole-image DA | 0.556 | 0.389 | 0.326 |
| V1 | Independent cutout DA | 0.611 | 0.444 | 0.407 |
| V2 | Whole-image DA | 0.556 | 0.426 | 0.413 |
| V2 | Independent cutout DA | 0.556 | 0.426 | 0.413 |

Parity compares DA-v1 tiles against official prepared CSV tiles (858 crops). Each cell is MAE / max absolute difference / fraction within 0.01:

| Version | Detector | Whole-image DA | Cutout DA |
|---|---|---|---|
| V1 | BSE | 0.0342 / 0.078 / 0.0625 | 0.0467 / 0.817 / 0.1252 |
| V1 | ETD | 0.0317 / 0.096 / 0.1093 | 0.0377 / 0.986 / 0.2025 |
| V1 | Inlens | 0.0646 / 0.943 / 0.1591 | 0.1156 / 0.992 / 0.1628 |
| V2 | BSE | 0.0342 / 0.078 / 0.0625 | 0.0467 / 0.817 / 0.1252 |
| V2 | ETD | 0.0317 / 0.096 / 0.1093 | 0.0377 / 0.986 / 0.2025 |
| V2 | Inlens | 0.0646 / 0.943 / 0.1591 | 0.1156 / 0.992 / 0.1628 |

V2 used the flat-surface fallback for one Inlens cutout; the fallback flag is included in `raw_parity_v2.csv` and `raw_eval_v2/predictions.csv`. Both raw variants had zero Batch 2 recall. Full raw metrics are in `raw_eval/metrics.json` and `raw_eval_v2/metrics.json`; parity reports are `raw_parity.json` and `raw_parity_v2.json`.

| Version | Live-demo seed | Raw accuracy | Prepared-CSV accuracy |
|---|---:|---:|---:|
| V1 | 42 | 0.583 | 0.667 |
| V2 | 43 | 0.500 | 0.333 |

V2 live-demo queries (`live_demo_v2.json`):

| Query | Raw prediction | CSV prediction | Truth |
|---|---|---|---|
| query_01.tif | Batch 3 | Batch 3 | Batch 1 |
| query_02.tif | Batch 3 | Batch 3 | Batch 3 |
| query_03.tif | Batch 2 | Batch 2 | Batch 1 |
| query_04.tif | Batch 3 | Batch 3 | Batch 2 |
| query_05.tif | Batch 2 | Batch 2 | Batch 1 |
| query_06.tif | Batch 3 | Batch 3 | Batch 3 |
| query_07.tif | Batch 3 | Batch 3 | Batch 3 |
| query_08.tif | Batch 1 | Batch 1 | Batch 3 |
| query_09.tif | Batch 3 | Batch 3 | Batch 3 |
| query_10.tif | Batch 3 | Batch 2 | Batch 3 |
| query_11.tif | Batch 1 | Batch 2 | Batch 1 |
| query_12.tif | Batch 1 | Batch 1 | Batch 3 |

## Findings and limitations

- No broad, reliable superiority to the majority baseline is established for these specimens. V1 MicroNet has a numerically higher point estimate and a bootstrap interval above the baseline point, but that interval is based on only six groups and one Batch 2 group; it is not reliable evidence of generalization.
- Batch 2 recall is **zero for all four prepared-CSV models**; the test contains only one independent Batch 2 section.
- Six independent test groups make uncertainty large. Treat image/crop metrics as specimen-specific, not population-level validation.
- V2 train CE fell below approximately 0.5 and VAL source-image accuracy reached 0.867, while TEST accuracy was 0.389 / 0.556 depending on init. This is consistent with overfitting and/or section-level variance.
- Raw DA-v1 parity is poor, especially for Inlens; raw-mode accuracies should not be treated as equivalent to prepared-CSV performance.
- Softmax probabilities are not calibrated.
- Conclusions are scoped to these three specimens and their grouped fields of view.

## Artifact map

Run outputs are **not committed to git** (`artifacts/` is ignored, as are the raw `Batch_*` and external eval folders); regenerate them locally with `PYTHONUTF8=1 modal run modal_app.py::fetch --run-id run-20261004-100739`. Fetched results live under `artifacts/run-20261004-100739/`:

- `source_manifest.csv`, `split_manifest.csv`, `prepare_summary.json`, profile/provenance, and preparation logs.
- `smoke/report.json` (V1) and `smoke_b200/report.json` (B200 v2 image smoke).
- `stage_a/`, `stage_b_stage_a/`, `stage_b_micronet/` (V1) and `stage_a_v2/`, `stage_b_stage_a_v2/`, `stage_b_micronet_v2/` (V2): summaries, metrics JSONL, model cards, exports.
- `model_selection.json`, `model_selection_v2.json`.
- `evaluation/<model>/{val,test}/`: metrics, predictions CSVs, confusion-matrix PNGs, and previews.
- `raw_eval/`, `raw_eval_v2/`, `raw_parity.json`/CSV, `raw_parity_v2.json`/CSV, `live_demo.json`, `live_demo_v2.json`, and pipeline summaries.

The Modal volumes remain populated: `neura-data` mounts at `/data` (raw data, split manifests, staging, prepared CSVs/manifests/caches); `neura-hf-cache` mounts at `/hf`; `neura-runs` mounts at `/runs` (profiles, logs, checkpoints, model cards, metrics, predictions). `fetch` omits `*.pt` unless `--with-checkpoints` is set; checkpoints remain in the volume. The prepared run is `/data/prepared/run-20261004-100739/{train,val,test}/`.

Tests: `python -m pytest -q tests`.

## Earlier KPI/profile baseline (historical)

# Microscopy sample batch assignment

**Current best classifier and results: see [CLASSIFIER.md](CLASSIFIER.md).**

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

To run data pipeline:

```
pip install numpy scipy pillow
python3 prepare_dataset.py Result Batch_3 Batch_1 Batch_2 
```
