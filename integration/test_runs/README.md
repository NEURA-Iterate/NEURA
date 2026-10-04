# Test runs

Test samples classified with the integration app, one folder per sample:

- `<sample>_BSE.tif`, `<sample>_ETD.tif` (or `_SE`), `<sample>_Inlens.tif`: the uploaded detector images
- `result.json`: the full classification result (probabilities, intervals, KPIs, explanations)
- `job.json`: run metadata
- `*.png` / `*.jpg`: rendered masks, cracks and KPI images

`history.csv` summarises all runs (one row per sample). To show a run in the app's History tab, copy its folder to `$NEURA_JOBS_DIR/<job_id>`, using the `job_id` column of `history.csv`: the app uses the folder name as the job ID and image links depend on it.
