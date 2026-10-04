from __future__ import annotations

import csv
import io
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
from fastapi.testclient import TestClient
from neura_app.api import create_app
from neura_app.model import BASE, BATCHES


class StubPipeline:
    def run(
        self,
        *,
        image: Any,
        sample_id: str,
        job_dir: Path,
        job_id: str,
        known_batch: str | None,
        progress: Any = None,
    ) -> dict[str, Any]:
        if progress is not None:
            progress("stub inference", 0.8)
        probabilities = {"Batch_1": 0.8, "Batch_2": 0.15, "Batch_3": 0.05}
        return {
            "sample_id": sample_id,
            "known_batch": known_batch,
            "timings": {"stub": 0.01},
            "trust_flags": "test_flag",
            "prediction": {
                "predicted": "Batch_1",
                "probabilities": probabilities,
                "interval": {
                    "Batch_1": [0.7, 0.9],
                    "Batch_2": [0.08, 0.22],
                    "Batch_3": [0.01, 0.09],
                },
                "tier": "high",
                "ambiguous": False,
                "outlier": False,
            },
            "classifiers": {
                "rule": {"probabilities": probabilities},
                "learned": {"probabilities": probabilities},
            },
        }


def _tiff_bytes(shape: tuple[int, int]) -> bytes:
    stream = io.BytesIO()
    tifffile.imwrite(stream, np.full(shape, 100, dtype=np.uint8))
    return stream.getvalue()


def _write_overview_data(root: Path) -> tuple[Path, Path, Path, Path]:
    artifacts = root / "artifacts"
    validation = root / "validation"
    figures = root / "figures"
    data_dir = root / "data"
    for path in (artifacts, validation / "strat_v1", validation / "final_ensemble", figures):
        path.mkdir(parents=True, exist_ok=True)

    rule_rows = []
    learned_rows = []
    for index, batch in enumerate(BATCHES):
        image_id = f"sample-{index}"
        values = {
            kpi: (0.1 + index * 0.05 if "frac_" in kpi or "contact" in kpi else 1.0 + index * 0.1)
            for kpi in BASE
        }
        rule_rows.append(
            {"image_id": image_id, "batch": batch, "trust_flags": "", **values}
        )
        learned_rows.append({"image_id": image_id, "batch": batch, **values})
    pd.DataFrame(rule_rows).to_csv(artifacts / "rule_per_image.csv", index=False)
    pd.DataFrame(learned_rows).to_csv(artifacts / "learned_per_image.csv", index=False)

    subset_rows = []
    for table in ("rule", "learned"):
        subset_rows.append(
            {
                "table": table,
                "strategy": "subset",
                "correct": 2,
                "n": 3,
                "acc_Batch_1": 1.0,
                "acc_Batch_2": 0.0,
                "acc_Batch_3": 1.0,
                "image_correct": 2,
                "log_loss": 0.7,
            }
        )
    pd.DataFrame(subset_rows).to_csv(validation / "strat_v1" / "summary.csv", index=False)

    prediction_rows = []
    per_image_rows = []
    for index, row in enumerate(rule_rows):
        batch = row["batch"]
        probs = [0.8 if candidate == batch else 0.1 for candidate in BATCHES]
        prediction_rows.append(
            {
                "round": 0,
                "image_id": row["image_id"],
                "true": batch,
                **{f"P_{candidate}": probs[j] for j, candidate in enumerate(BATCHES)},
                "pred": batch,
                "correct": True,
                "tier": "high",
            }
        )
        per_image_rows.append(
            {
                "image_id": row["image_id"],
                "true": batch,
                **{f"P_{candidate}": probs[j] for j, candidate in enumerate(BATCHES)},
                "tiers_seen": "high",
                "pred": batch,
                "correct": True,
                "trust_flags": "",
            }
        )
    pd.DataFrame(prediction_rows).to_csv(
        validation / "final_ensemble" / "predictions.csv", index=False
    )
    pd.DataFrame(per_image_rows).to_csv(
        validation / "final_ensemble" / "per_image.csv", index=False
    )

    sample_dir = data_dir / "Batch_1"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("BSE", "ETD", "Inlens"):
        tifffile.imwrite(sample_dir / f"demo_{suffix}.tif", np.full((16, 20), 100, dtype=np.uint8))
    return artifacts, validation, figures, data_dir


def _app(
    tmp_path: Path,
    pipeline: Any | None = None,
    jobs_dir: Path | None = None,
):
    artifacts, validation, figures, data_dir = _write_overview_data(tmp_path)
    app = create_app(
        pipeline=pipeline or StubPipeline(),
        data_dir=data_dir,
        jobs_dir=jobs_dir or tmp_path / "jobs",
        artifacts_dir=artifacts,
        validation_dir=validation,
        figures_dir=figures,
        repo_root=tmp_path / "repo",
    )
    return app


def _upload_files(
    bse_shape: tuple[int, int] = (16, 20),
    etd_shape: tuple[int, int] = (16, 20),
    inlens_shape: tuple[int, int] = (16, 20),
) -> dict[str, tuple[str, bytes, str]]:
    return {
        "bse": ("sample_BSE.tif", _tiff_bytes(bse_shape), "image/tiff"),
        "etd": ("sample_ETD.tif", _tiff_bytes(etd_shape), "image/tiff"),
        "inlens": ("sample_Inlens.tif", _tiff_bytes(inlens_shape), "image/tiff"),
    }


def _wait_for_job(client: TestClient, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["status"] in {"done", "error"}:
            return state
        time.sleep(0.01)
    raise AssertionError(f"Job {job_id} did not finish")


def test_overview_contains_required_sections_and_validation_note(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path)) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        response = client.get("/api/overview")
    assert response.status_code == 200
    overview = response.json()
    assert {
        "classes",
        "n_per_batch",
        "kpi_meta",
        "classifiers",
        "validation",
        "training",
        "batch_stats",
        "figures",
        "representatives",
        "legend",
    } <= overview.keys()
    assert {"summary", "per_sample", "confusion", "tiers", "note"} <= overview["validation"].keys()
    assert "25-sample" in overview["validation"]["note"]
    assert "all 31 samples" in overview["validation"]["note"]
    assert set(overview["classifiers"]) == {"rule", "learned"}


def test_upload_job_lifecycle_and_demo_known_batch(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()

    class GatedPipeline(StubPipeline):
        def run(self, **kwargs: Any) -> dict[str, Any]:
            started.set()
            if not release.wait(timeout=3):
                raise TimeoutError("Test pipeline gate timed out")
            return super().run(**kwargs)

    with TestClient(_app(tmp_path, GatedPipeline())) as client:
        response = client.post(
            "/api/classify",
            data={"sample_id": "uploaded-sample"},
            files=_upload_files(),
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        initial = client.get(f"/api/jobs/{job_id}").json()
        assert initial["status"] in {"queued", "running"}
        assert started.wait(timeout=2)
        running = client.get(f"/api/jobs/{job_id}").json()
        assert running["status"] == "running"
        release.set()
        uploaded_job = _wait_for_job(client, job_id)
        assert uploaded_job["status"] == "done"
        assert uploaded_job["result"]["sample_id"] == "uploaded-sample"
        assert uploaded_job["result"]["known_batch"] is None

        demo = client.post("/api/classify-demo", json={"image_id": "demo"})
        assert demo.status_code == 200
        demo_job = _wait_for_job(client, demo.json()["job_id"])
        assert demo_job["status"] == "done"
        assert demo_job["result"]["known_batch"] == "Batch_1"
        assert client.get("/api/demo-samples").json() == [
            {"image_id": "demo", "batch": "Batch_1"}
        ]


def test_upload_validation_rejects_missing_and_mismatched_detectors(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path)) as client:
        missing = client.post(
            "/api/classify",
            files={"bse": ("sample.tif", _tiff_bytes((16, 20)), "image/tiff")},
        )
        assert missing.status_code == 422

        mismatch = client.post(
            "/api/classify",
            files=_upload_files(etd_shape=(15, 20)),
        )
    assert mismatch.status_code == 422
    assert "Detector shape mismatch" in mismatch.json()["detail"]


def test_job_history_survives_app_restart(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "persistent-jobs"
    with TestClient(_app(tmp_path, jobs_dir=jobs_dir)) as client:
        response = client.post("/api/classify-demo", json={"image_id": "demo"})
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        first_state = _wait_for_job(client, job_id)
        assert first_state["status"] == "done"

    job_metadata = json.loads((jobs_dir / job_id / "job.json").read_text())
    assert job_metadata["source"] == "demo"
    assert job_metadata["status"] == "done"

    with TestClient(_app(tmp_path, jobs_dir=jobs_dir)) as client:
        restored = client.get(f"/api/jobs/{job_id}").json()
        assert restored["status"] == "done"
        assert restored["result"]["sample_id"] == "demo"
        assert restored["sample_id"] == "demo"
        history = client.get("/api/history").json()
    assert len(history) == 1
    assert history[0]["sample_id"] == "demo"
    assert history[0]["source"] == "demo"
    assert history[0]["predicted"] == "Batch_1"


def test_history_csv_has_expected_header_and_completed_upload(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path)) as client:
        response = client.post(
            "/api/classify",
            data={"sample_id": "sample-csv"},
            files=_upload_files(),
        )
        job_id = response.json()["job_id"]
        assert _wait_for_job(client, job_id)["status"] == "done"
        history_csv = client.get("/api/history.csv")

    assert history_csv.status_code == 200
    assert history_csv.headers["content-disposition"] == (
        "attachment; filename=neura_history.csv"
    )
    reader = csv.DictReader(io.StringIO(history_csv.text))
    assert reader.fieldnames == [
        "job_id",
        "created_at",
        "sample_id",
        "source",
        "known_batch",
        "status",
        "predicted",
        "P_Batch_1",
        "P_Batch_2",
        "P_Batch_3",
        "P_Batch_1_lo",
        "P_Batch_1_hi",
        "P_Batch_2_lo",
        "P_Batch_2_hi",
        "P_Batch_3_lo",
        "P_Batch_3_hi",
        "tier",
        "ambiguous",
        "outlier",
        "trust_flags",
        "rule_P_Batch_1",
        "rule_P_Batch_2",
        "rule_P_Batch_3",
        "learned_P_Batch_1",
        "learned_P_Batch_2",
        "learned_P_Batch_3",
    ]
    rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["sample_id"] == "sample-csv"
    assert rows[0]["source"] == "upload"
    assert rows[0]["P_Batch_1"] == "0.8"
    assert rows[0]["P_Batch_1_lo"] == "0.7"
    assert rows[0]["rule_P_Batch_1"] == "0.8"
    assert rows[0]["learned_P_Batch_3"] == "0.05"


def test_job_route_rejects_invalid_job_ids(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path)) as client:
        assert client.get("/api/jobs/not-hex").status_code == 404
        assert client.get("/api/jobs/..%2Fx").status_code == 404


def test_interrupted_job_is_reported_after_restart(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "interrupted-jobs"
    job_id = "f" * 32
    job_dir = jobs_dir / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "sample_id": "interrupted-sample",
                "known_batch": None,
                "source": "upload",
                "created_at": "2025-01-01T00:00:00Z",
                "status": "running",
                "step": "classifying",
                "error": None,
            }
        )
    )
    with TestClient(_app(tmp_path, jobs_dir=jobs_dir)) as client:
        response = client.get(f"/api/jobs/{job_id}")
        history = client.get("/api/history").json()
    assert response.status_code == 200
    assert response.json()["status"] == "error"
    assert response.json()["error"] == (
        "Interrupted: the server restarted before this job finished."
    )
    assert history[0]["status"] == "error"
    assert history[0]["sample_id"] == "interrupted-sample"
    assert all(
        history[0][field] is None
        for field in (
            "predicted",
            "probabilities",
            "interval",
            "tier",
            "ambiguous",
            "outlier",
        )
    )


def test_legacy_result_only_job_is_in_history(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "legacy-jobs"
    job_id = "a" * 32
    job_dir = jobs_dir / job_id
    job_dir.mkdir(parents=True)
    result = StubPipeline().run(
        image=None,
        sample_id="legacy-sample",
        job_dir=job_dir,
        job_id=job_id,
        known_batch="Batch_2",
    )
    (job_dir / "result.json").write_text(json.dumps(result))

    with TestClient(_app(tmp_path, jobs_dir=jobs_dir)) as client:
        history = client.get("/api/history").json()
    assert len(history) == 1
    assert history[0]["job_id"] == job_id
    assert history[0]["sample_id"] == "legacy-sample"
    assert history[0]["known_batch"] == "Batch_2"
    assert history[0]["source"] == "demo"
    assert history[0]["status"] == "done"
