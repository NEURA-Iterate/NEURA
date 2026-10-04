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

## Run it in a Devin session

From the repository root, run `bash integration/run.sh`. The launcher builds the frontend and serves both the UI and API on port 8000 by default (`PORT` overrides it). DINOv2 weights are fetched from Hugging Face on first use. If the private sample dataset is missing, set `HF_TOKEN` to allow its download; the token is not printed.

| Environment variable | Default |
| --- | --- |
| `NEURA_DATA_DIR` | `/home/ubuntu/data/neura` |
| `NEURA_JOBS_DIR` | `/home/ubuntu/data/runs/app_jobs` |
| `NEURA_CACHE_DIR` | `<parent of NEURA_JOBS_DIR>/cache` |
| `PORT` | `8000` |

The History tab shows saved classifications; export the same history with `GET /api/history.csv`.
