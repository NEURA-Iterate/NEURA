# Feature-based phase segmentation

```text
Hugging Face dataset → Modal /data volume
        → BSE + ETD/SE + Inlens triplet
        → frozen facebook/dinov2-small (384 channels, 14 px patches)
        → cached 384-channel features from 518 px tiles / 56 px overlap
        → LinearProbe or FusionHead trained on pixel labels
        → full-resolution pore / graphite / Si / binder masks
        → anode_qc KPI calculations
```

The DINOv2-small backbone is frozen; training updates only the segmentation head.
`LinearProbe` is a 1×1 classifier over patch features. `FusionHead` has fewer than
0.5M parameters and fuses DINO features with a full-resolution CNN on the raw
detector triplet. The 14 px patch stride is too coarse for reliable Si edges and
particle sizes on its own; the image branch restores full-resolution boundary
detail. The four output classes are `pore`, `graphite`, `si`, and `binder`, mapped
to anode_qc `PORE`, `GRAPHITE`, `SI`, and `CBD`. Cracks are not predicted.

## Labels and limitations

The current pseudo-labels come from `anode_qc.pipeline.segment_image`. They are a
bootstrap only: a head trained on rule-based masks can at best reproduce those
rules, not validate them. `anode_qc.uncertainty.class_probabilities` returns separate
P(Si) and P(pore) maps rather than four-class probabilities, so pseudo-label pixels
within 2 px of a mapped phase boundary are marked `255` (ignore) instead. Expert
scribbles can be loaded from grayscale PNG or NPY files with `255` for unlabelled
pixels and should replace pseudo-labels for defensible KPIs. Binder labels need
expert review. Cracks are out of scope and need a separate full-resolution treatment.
Pseudo-label maps can be reused with `--label-cache DIR`; the cache stores uint8 maps
keyed by label settings.

Classes are ordered `pore`, `graphite`, `si`, `binder`; anode_qc `PORE` and `GAP`
map to pore, `GRAPHITE` to graphite, `SI` to Si, and `CBD` to binder.

## Precomputed features

The native cache is one `{image_id}.npy` per image, HWC with shape
`(ceil(H/14), ceil(W/14), 384)`, `float16`; `image_id` is the BSE stem used in
`per_image.csv` (without `_BSE.tif`). A shared `meta.json` records `model_id`,
`upsample`, `tile`, `overlap`, `channels`, `patch_size`, input/backbone
normalisation, and feature layout. The loader also accepts a colleague's
`{image_id}.npz` with an array named `features` in HWC or CHW order and validates
its grid against the border-cropped image shape and patch size. `load_features` is
the single adapter point if the handoff format changes.

## Local usage

From the repository root, install CPU PyTorch, the existing QC package, and this
package with development tools:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers
python -m pip install -e ./anode_microstructure_qc
python -m pip install -e 'feature_classifier[dev]'
```

After the dataset is present locally, extract cached features and train one model
with the deterministic holdout split:

```bash
python -m phaseseg.features \
  --images /path/to/images --out /path/to/features \
  --tile 518 --overlap 56 --batch-size 8 --device cpu
python -m phaseseg.train \
  --images /path/to/images --features /path/to/features \
  --labels pseudo --label-cache /path/to/label-cache \
  --split holdout --head linear --out /path/to/linear \
  --epochs 10 --device cpu
python -m phaseseg.evaluate \
  --images /path/to/images --features /path/to/features \
  --labels pseudo --label-cache /path/to/label-cache \
  --checkpoint /path/to/linear/model.pt \
  --split-json /path/to/linear/split.json --split-role test \
  --tile-size 224 --overlap 56 --device cpu --out /path/to/evaluation
```

For FusionHead, use `--head fusion --epochs 15 --crops-per-image 16` and evaluate
its `model.pt` with the same split. Use `--labels DIR` with `{image_id}.png` or
`{image_id}.npy` scribble maps for expert labels. Training selects the best epoch
using validation images; test images are not loaded during training.

## Holdout split and results

The fixed-seed `--split holdout` uses one test and one validation image per batch,
with the remaining images for training (25/3/3):

| Role | Batch_1 | Batch_2 | Batch_3 |
| --- | --- | --- | --- |
| Test | `img_f1vzngrs` | `img_r17byphk` | `img_hzumfsms` |
| Validation | `img_fzrt2k6r` | `img_b3esycq1` | `img_hawkfj64` |

On the 8-CPU VM, without a GPU, DINO feature extraction took 706.85 s total
(22.5 s/image on average); filling the label cache took 2.6 min. LinearProbe
training took 1 min. Fusion training measured 92.5 s/epoch; 15 epochs were about
23 min. Full-image Fusion prediction for all 31 images took 906 s.

**Important:** these scores compare predictions with rule-derived pseudo-labels.
They measure reproduction of the rules, not segmentation correctness; expert
annotations are needed to assess correctness.

| Test image | Pixel accuracy: linear → Fusion | Mean IoU: linear → Fusion |
| --- | ---: | ---: |
| Batch_1 | 0.88 → 0.94 | 0.68 → 0.85 |
| Batch_2 | 0.87 → 0.94 | 0.63 → 0.81 |
| Batch_3 | 0.87 → 0.95 | 0.64 → 0.83 |

Fusion per-class IoU was pore 0.98, graphite 0.93, Si 0.94, and binder 0.47. Linear
per-class IoU was Si 0.70, pore 0.73, graphite 0.86, and binder 0.31.

Fusion KPI values below are predicted versus rule-based, in test-image order
Batch_1 / Batch_2 / Batch_3. Fractions are percentages; Si D50 is in pixels.

| KPI | Batch_1 | Batch_2 | Batch_3 |
| --- | ---: | ---: | ---: |
| Si fraction | 6.7 vs 6.4% | 4.2 vs 3.7% | 4.8 vs 4.5% |
| Pore fraction | 6.4 vs 5.6% | 14.3 vs 15.6% | 18.8 vs 21.2% |
| Filtered Si D50 | 41.5 vs 47.7 | 20.9 vs 32.9 | 44.1 vs 50.8 |

Filtered Si D50 applies the same 30 px minimum-area cleanup as the rule-based KPI
path; `si_ecd_d50_raw` is also reported. Linear-probe D50 values were 16–20 px.

## Modal

`modal_app.py` defines `neura-phaseseg`, mounts the existing `neura-data` volume at
`/data`, downloads the Hugging Face dataset to `/data/images`, and runs the same
feature-extraction and training module CLIs on a T4. Images/features/runs are read
or written only on that volume; no dataset is uploaded from the local machine.

```bash
modal secret create huggingface HF_TOKEN=...
modal run feature_classifier/phaseseg/modal_app.py::download_dataset
modal run feature_classifier/phaseseg/modal_app.py::smoke
modal run feature_classifier/phaseseg/modal_app.py::extract_features --batch-size 8
modal run feature_classifier/phaseseg/modal_app.py::train --epochs 10 --folds 5
```

`download_dataset` reads `HF_TOKEN` from the existing Modal secret named
`huggingface`; create it with the command above only when the token is available.
The smoke function downloads public DINOv2-small weights, warms up with two
518×518 forwards at batch size 8, and times 20 more to report per-tile throughput.
It does not access the private image dataset. Feature extraction streams each
image's tiles in configurable batches, writes that image's feature file, then moves
to the next image. Sliding-window evaluation retains only the current image's
probability accumulator and one model tile on the GPU. The Modal T4 path is
configured, but the workspace currently needs a payment method to run GPU functions.
