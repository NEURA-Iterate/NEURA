# Do the batches differ as populations? Permutation tests, KPI reliability, detection limits

Follow-up to `kpi_hypothesis_test.md` and `encoder_embedding_test.md`, which
asked "can one held-out sample be assigned to its batch?" (a weak question at
n = 7 per batch). This note asks the QC question directly: **do Batch 1 and
Batch 2 differ at all, within what detection limit, and which KPIs are even
reliable enough to say?**

## 0. Metadata and data bar: nothing recoverable

Every TIFF carries only 16 standard tags, `Software = tifffile.py`,
`ImageDescription = {"shape": [...]}`; no Zeiss `CZ_SEM`, FEI, XMP or ImageJ
blocks, and no burned-in data bar (no constant rows/columns at any image
edge). Pixel size, EHT, WD, detector settings, scan speed and date are gone.
**All size KPIs stay in px and acquisition session cannot be reconstructed;
the organisers have to supply this.** The three detector files per sample are
named `_BSE`, `_ETD`/`_SE`, `_Inlens` - InLens + ETD is a Zeiss/FEI mix of
names, which does not help either.

## 1. Two-sample energy-distance permutation tests (sample level)

`batch_difference_test.py`: energy distance between the two sets of
sample-level vectors (each column robust-scaled on the pooled pair, clipped at
+-5; U-statistic form), p-value from relabelling - all 3432 splits for 7 v 7,
5000 random splits for 7 v 17. Also the self-split null: energy distance
between random halves of one batch in the same scaled space (the "no change"
band; note 3 v 3 halves are very noisy).

| feature block | 1 vs 2 (7 v 7) | 1 vs 3 (7 v 17) | 2 vs 3 (7 v 17) |
|---|---|---|---|
| 26 material `kpi__` | ED 0.47, **p = 0.31** | ED 1.71, p = 0.020 | ED 1.65, p = 0.014 |
| 18 imaging `qa__` | ED 0.57, p = 0.21 | ED 1.66, p = 0.044 | ED 2.63, p = 0.013 |
| DINOv2 embedding (1152-d) | ED 2.57, p = 0.26 | ED 12.2, p = 0.0004 | ED 8.1, p = 0.004 |
| Batch-3 self-split 95th pct (8 v 8), kpi / qa / emb | - | 1.8 / 2.2 / 5.0 | 1.6 / 2.4 / 5.7 |

Per-KPI univariate tests (`batch_difference_per_kpi.csv`): 1 vs 2 has 2/26
KPIs below p = 0.05 uncorrected (expected ~1.3 by chance) and 0 after
Bonferroni; 1 vs 3 has 12/26 (2 after Bonferroni: BSE vertical solid chord,
SE edge-pixel fraction); 2 vs 3 has 8/26 (1: InLens bright-feature
elongation).

Reading:

- **Batch 1 and Batch 2 are indistinguishable** by every block - hand KPIs,
  imaging descriptors and a generic deep embedding all give p = 0.2-0.3 with
  the exact permutation distribution. This is consistent with Batch 2 being
  "acceptable variation" of the same material, and also consistent with the
  two being imaged in one session. It is not evidence that the KPIs are weak.
- **Batch 3 differs from both** (p = 0.0004-0.02) and the 1v3 / 2v3 energy
  distances sit at or just above the Batch-3 self-split 95th percentile for
  KPIs, clearly above it for the embedding. But the imaging block differs
  just as significantly (p = 0.013-0.044), so part of the Batch-3 signal is
  acquisition; a verdict on Batch 3 must say "material and acquisition both
  differ" until matched-settings images exist.

## 2. KPI reliability (split-half, `kpi_reliability.py`)

Each KPI recomputed on the left and right halves of every image;
reliability = 1 - within-sample half-variance / total variance (ICC-like),
plus the Spearman correlation of the two halves across the 31 samples.

| KPI | reliability | rho(halves) | verdict |
|---|---|---|---|
| BSE phase contrast bright vs graphite (8-bit) | 0.95 | 0.78 | reliable (but a grey-level KPI - acquisition-sensitive) |
| BSE bright-particle count (/Mpx) | 0.90 | 0.63 | reliable |
| InLens dark fraction | 0.88 | 0.77 | reliable, but tracks InLens brightness |
| **BSE pore fraction** | **0.86** | 0.76 | reliable |
| BSE solid chord, vertical (px) | 0.80 | 0.67 | reliable |
| InLens edge density | 0.79 | 0.82 | reliable |
| InLens bright-feature elongation | 0.69 | 0.69 | usable |
| SE edge-pixel fraction | 0.69 | 0.67 | usable |
| BSE bright-phase fraction | 0.67 | 0.19 | half-to-half rank unstable (few large particles) |
| BSE pore d50 (px) | 0.65 | 0.57 | usable |
| ... | | | |
| InLens bright-edge skeleton length | 0.43 | 0.44 | weak |
| **SE dark-region d50** | **0.30** | 0.04 | noise - drop |
| **SE dark fraction** | **0.27** | 0.31 | noise - drop |
| BSE dominant angle | 0.26 | 0.12 | noise - drop |
| BSE bright-particle d50 | 0.25 | 0.28 | noise - drop |
| BSE solid-chord anisotropy | 0.19 | 0.30 | noise - drop |
| SE dark-region count | 0.07 | -0.11 | noise - drop |

Full table: `kpi_reliability.csv`. Consequence for the earlier KPI report:
two of its "top separators" for Batch 3 - SE dark fraction and SE dark-region
d50 - have reliability < 0.31 and should be discarded; the SE dark-region
segmentation (Otsu on the ETD image) is not measuring a stable quantity. The
BSE phase-fraction and chord KPIs, and the InLens dark fraction, are the
ones worth keeping and calibrating.

## 3. Detection limits for Batch 1 vs Batch 2 (`min_detectable_shift_b1_vs_b2.csv`)

With 7 v 7 samples, a two-sided 5 % test has 80 % power for a mean shift of
about 1.6 pooled within-batch SD. In the units of the reliable KPIs:

| KPI | Batch 1 mean | Batch 2 mean | observed shift | min detectable shift (7 v 7) |
|---|---|---|---|---|
| BSE pore fraction | 0.082 | 0.096 | +0.014 | 0.026 (32 % of B1) |
| BSE pore d50 (px) | 112 | 130 | +18 | 42 (38 %) |
| BSE solid chord, vertical (px) | 224 | 192 | -32 | 48 (22 %) |
| BSE bright-particle count (/Mpx) | 19.6 | 10.6 | -9.0 | 19.3 (98 %) |
| BSE phase contrast (8-bit) | 41.9 | 43.7 | +1.8 | 6.5 (15 %) |
| InLens edge density (/px) | 0.0223 | 0.0223 | 0.000 | 0.003 (12 %) |

So the honest QC statement for Batch 2 is: *"materially indistinguishable
from Batch 1 within our detection limits: porosity shift < ~0.03 absolute,
pore d50 shift < ~40 px, platelet (vertical chord) shift < ~50 px; imaging
conditions not significantly different either."* Some observed shifts
(bright-particle count -46 %, vertical chord -14 %) are large in relative
terms but inside the noise at n = 7; they are the first things to re-measure
if more Batch-2 samples arrive, not things to tune features towards.

## 4. What changes in the plan

1. Stop trying to separate 1 from 2. Treat Batch 1 (or 1 + 2 pooled, 14
   samples) as the baseline and build the accept band from its self-split
   null; Batch 3 is the "changed" example to calibrate the investigate/reject
   bands against.
2. Trim the KPI block to the reliable, geometry-based ones (BSE pore
   fraction, pore d50, solid chords, bright-particle count; InLens edge
   density / elongation) and move grey-level quantities (phase contrast,
   InLens dark fraction) into the acquisition block or make them invariant
   (histogram-match per detector before segmentation).
3. Particle-level distributions (pore / platelet size, not per-sample
   medians) with a hierarchical bootstrap are the next power gain; the
   per-sample reliability numbers above say which segmentations are stable
   enough to do that on.
4. Ask the organisers: pixel size, detector settings / session per image,
   and whether Batches 1 and 2 are intended to be the same material.

## Reproduce

```bash
python batch_difference_test.py --features features_real.csv --out batch_difference.csv
python batch_difference_test.py --features features_real.csv --feature-prefix qa__ --out batch_difference_qa.csv
python batch_difference_test.py --features features_emb_dinov2.csv --feature-prefix emb__ --out batch_difference_dinov2.csv
python kpi_reliability.py --manifest sample_manifest.csv --out kpi_reliability.csv
```
