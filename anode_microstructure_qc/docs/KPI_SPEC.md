# Key QC KPIs: interpretability and uncertainty

Ten KPIs, ranked by how much they matter for a QC accept/reject or lot-change decision on Si–graphite
anodes. Uncertainty played no part in the ranking.

## Shared rules

- **Interpretability:** every number has an image overlay of the exact pixels / particles behind it and a
  plain-language definition.
- **Uncertainty:** every KPI is reported as `value [pixel lo – hi] [algorithm p5 – p95]` with its main driver.
  - *Pixel ambiguity* (root cause: noise and edge blur). Each pixel gets a class probability from the
    per-image brightness distribution of each class (BSE for Si, ETD for pore). The edge blur width δ is
    measured across real Si edges. The KPI is recomputed on 4 label maps: confident pixels only (P > 0.95),
    inclusive (P > 0.05), objects shrunk by δ, objects grown by δ. The range over these maps is the interval.
  - *Algorithm choices* (root cause: settings with no single correct value). The full pipeline is rerun
    30× per image, with settings drawn at random within plausible ranges (same draws for every image).
    The interval is the 5th–95th percentile. The main driver is the setting most correlated with the KPI.
  - A batch difference counts as **robust** only if it is significant (p < 0.05) in ≥ 95 % of runs.
- **Trust flags** per image: Si/graphite contrast, polishing streaks, charging, BSE vs multi-detector
  agreement. A flagged image's KPIs are shown but excluded from batch decisions.

## KPIs

| Rank | KPI | QC decision it drives | Interpretability | Uncertainty |
|---|---|---|---|---|
| 1 | **Si fraction of solids (→ Si wt %)** | Recipe right? Capacity on spec? Accept/reject the lot. | Si overlay; histogram with the Si threshold relative to the graphite peak; mesh regions removed by ETD shown separately. | Pixel: blurred Si edges. Algorithm: Si threshold, cleanup, mesh filter. Wt % is labelled "assumes pure Si". |
| 2 | **Porosity + top-to-bottom gradient** | Calendering/density on spec? | ETD pore overlay; porosity per row band; BSE vs ETD pore disagreement. | Pixel: pore walls, ETD shadowing. Algorithm: ETD pore threshold, gap width. Gradient valid only if rows run through the thickness. Recessed pores showing material underneath are a stated undercount. |
| 3 | **Si coarse tail D90 (with D50)** | Si supplier/milling change? Fade risk. | Size distribution; the top-10 % particles highlighted; per-particle table. | Pixel: edge ±δ (worse for small particles). Algorithm: minimum size (D50), merging of touching particles (D90). In px until the pixel size is known; 2D, number-weighted, not comparable with laser-diffraction D90. |
| 4 | **Si agglomeration (dispersion index)** | Slurry mixing OK? | Si density map per window next to one random-placement example; clusters outlined. | Pixel: small particles appearing/vanishing. Algorithm: minimum size, window size. Random-baseline spread reported; index within it = "not distinguishable from random". |
| 5 | **Cracks + Si debonding gaps** | Processing/calendering damage? | Cracks and gaps drawn on the ETD image; per-particle debond % in the table. | Pixel: 1–2 px gaps at the blur limit (dominant). Algorithm: gap width/depth, crack threshold. Polishing streaks suppressed and flagged. |
| 6 | **Binder/carbon-black amount + gradient** | Binder migration during drying? | Binder overlay on the Inlens image; per-row-band plot. | Pixel: gradual mesh boundaries. Algorithm: texture thresholds. Experimental until visually validated on ≈ 10 crops. |
| 7 | **Si material fingerprint (grey vs graphite)** | Silent Si material change: trigger EDS? | Si-core brightness histogram against the graphite peak; value against lot history. | Pixel: edge pixels (Si cores used), graphite-reference noise. Algorithm: smoothing, floor. A relative change flag, not chemical identification; only comparable at the same microscope settings. |
| 8 | **Si top-to-bottom gradient** | Si settling/segregation during coating? | Si per row band. | As #1, per band. Rows must run through the thickness. |
| 9 | **Graphite alignment** | Ion path too winding (tortuous) for fast charge? | Orientation rose plot. | Pixel: noisy boundaries bias it towards 0. Algorithm: structure-tensor scale. Coherence reported so "random" and "unmeasurable" aren't confused. |
| 10 | **Si contact (graphite / binder / pore / gap)** | Failure diagnosis. | Contact ring coloured by neighbour class. | Inherits #2, #5 and #6. Diagnostic, not pass/fail. |

## Settings varied for algorithm uncertainty (default in brackets)

| Setting | Range | KPIs |
|---|---|---|
| Denoise σ | 1.0–2.0 px [1.5] | all |
| Si threshold offset from valley | ±0.05 normalised [0] | 1, 3, 4, 7, 8, 10 |
| Si seed position | 0.3–0.7 [0.5] | 1, 3, 4 |
| Rim filter depth | 3–5 px [4] | 1, 3, 4 |
| Min seed fraction | 0.2–0.4 [0.3] | 1, 3, 4 |
| Opening radius | 0–2 px [1] | 1, 3, 4 |
| Min Si area | 20–50 px [30] | 3, 4 |
| Min solidity | 0.6–0.8 [0.7] | 1, 3, 4 |
| Mesh filter (max internal pore fraction) | 0.10–0.25 [0.15] | 1 |
| ETD pore threshold offset | ±0.05 [0] | 2, 5, 10 |
| Max gap width | 2–6 px [4] | 2, 5, 10 |
| Gap depth | 0.25–0.45 [0.35] | 2, 5, 10 |
| Crack threshold k | 4–8 [6] | 5 |
| Binder hole / edge density thresholds | ±30 % | 6, 10 |
| Structure-tensor scale | 2–6 px [4] | 9 |
| Contact ring inner / outer | 1–3 / 4–7 px [2 / 5] | 10 |

## Not measurable from these images

- Coating thickness.
- Delamination at the current collector, unless it's in frame.
- Absolute sizes, until the pixel size is known.

Field-of-view sampling, physical assumptions and bias against ground truth are not quantified (by decision).
