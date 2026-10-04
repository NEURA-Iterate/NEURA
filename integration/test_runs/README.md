# Test runs

Test samples classified with the integration app, one folder per sample:

- `<sample>_BSE.tif`, `<sample>_ETD.tif` (or `_SE`), `<sample>_Inlens.tif`: the uploaded detector images
- `result.json`: the full classification result (probabilities, intervals, KPIs, explanations)
- `job.json`: run metadata
- `*.png` / `*.jpg`: rendered masks, cracks and KPI images

`history.csv` summarises all runs (one row per sample). To make them appear in the app's History tab, copy the folders into `$NEURA_JOBS_DIR`.
