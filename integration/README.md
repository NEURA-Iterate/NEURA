# NEURA integration

The backend reuses the validated 25-sample Fusion holdout decoder and fits the two Bayesian KPI classifiers on all 31 samples. It does not retrain the decoder or regenerate the saved masks.

From the repository root, assemble compact artifacts and the representative panels from the existing outputs:

```bash
cd integration/backend
python -m pip install -e '.[dev]'
python -m neura_app.build_artifacts
uvicorn neura_app.api:app --host 0.0.0.0 --port 8000
```

The API reads `NEURA_DATA_DIR` (default `/home/ubuntu/data/neura`) and stores job files under `NEURA_JOBS_DIR` (default `/home/ubuntu/data/runs/app_jobs`). To run the Svelte frontend:

```bash
cd integration/frontend
npm install
npm run dev
```
