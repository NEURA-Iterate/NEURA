# Feature-based phase segmentation

```text
Hugging Face dataset → Modal /data volume
        → BSE + ETD/SE + Inlens
        → frozen DINOv2-small
        → cached 384-channel patch features
        → LinearProbe or FusionHead trained on pixel labels
        → full-resolution pore / graphite / Si / binder masks
        → anode_qc KPI calculations
```

The DINOv2-small backbone is frozen; training updates only the small segmentation
head. `LinearProbe` is a 1×1 classifier over patch features. `FusionHead` projects
those features and fuses them with a full-resolution CNN branch. DINOv2-small has a
14 px patch stride, which is too coarse for reliable Si edges and particle sizes on
its own; the image branch restores full-resolution boundary detail.

## Labels and limitations

The current pseudo-labels come from `anode_qc.pipeline.segment_image`. They are a
bootstrap only: a head trained on rule-based masks can at best reproduce those
rules, not validate them. `anode_qc.uncertainty.class_probabilities` returns separate
P(Si) and P(pore) maps rather than four-class probabilities, so pseudo-label pixels
within 2 px of a mapped phase boundary are marked `255` (ignore) instead. Expert
scribbles can be loaded from grayscale PNG or NPY files with `255` for unlabelled
pixels and should replace pseudo-labels for defensible KPIs. Binder labels need
expert review. Cracks are out of scope and need a separate full-resolution treatment.

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

After the dataset is present locally, extract cached features, train five
image-stratified folds, then evaluate one fold's held-out images:

```bash
python -m phaseseg.features --images /path/to/images --out /path/to/features
python -m phaseseg.train \
  --images /path/to/images --features /path/to/features \
  --labels pseudo --head fusion --folds 5 --out /path/to/runs --epochs 10 --device auto
python -m phaseseg.evaluate \
  --images /path/to/images --features /path/to/features \
  --labels pseudo --checkpoint /path/to/runs/fold_0.pt --out /path/to/evaluation
```

Use `--labels DIR` with `{image_id}.png` or `{image_id}.npy` scribble maps for
expert labels. Training splits by image and stratifies by batch; crops and tiles
from a single image never cross the train/validation boundary.

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
probability accumulator and one model tile on the GPU.
