from __future__ import annotations

import json
import os
import re
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Annotated, Any

import numpy as np
import pandas as pd
import tifffile
from anode_qc.data import BSEImage, find_bse_images
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .model import BASE, BATCHES, KPI_META, SOURCE_KPIS, DualSourceClassifier
from .pipeline import LEGEND, InferencePipeline

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = REPO_ROOT / "integration" / "artifacts"
VALIDATION_DIR = (
    REPO_ROOT
    / "bayesian_kpi_classifier"
    / "experiments"
    / "research"
    / "results"
    / "runs"
)
FIGURES_DIR = (
    REPO_ROOT
    / "bayesian_kpi_classifier"
    / "experiments"
    / "research"
    / "results"
    / "figures"
)
_SAFE_SAMPLE_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


class DemoRequest(BaseModel):
    image_id: str


def _number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _text(value: Any) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def _read_representatives(artifacts_dir: Path) -> dict[str, dict[str, Any]]:
    representatives = {}
    for batch in BATCHES:
        directory = artifacts_dir / "representatives" / batch
        metadata_path = directory / "kpis.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        representatives[batch] = {
            "image_id": metadata["image_id"],
            "images": {
                name: f"/api/files/artifacts/representatives/{batch}/{name}.png"
                for name in (
                    "bse",
                    "rule_overlay",
                    "learned_overlay",
                    "cracks",
                    "zoom_bse",
                    "zoom_rule",
                    "zoom_learned",
                    "zoom_cracks",
                )
            },
            "kpis": metadata["kpis"],
        }
    return representatives


def _summary_from_row(row: pd.Series) -> dict[str, Any]:
    return {
        "correct": int(row["correct"]),
        "n": int(row["n"]),
        "acc_Batch_1": float(row["acc_Batch_1"]),
        "acc_Batch_2": float(row["acc_Batch_2"]),
        "acc_Batch_3": float(row["acc_Batch_3"]),
        "image_correct": int(row["image_correct"]),
        "log_loss": float(row["log_loss"]),
    }


def _combined_summary(validation_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    ensemble = validation_dir / "final_ensemble"
    predictions = pd.read_csv(ensemble / "predictions.csv")
    per_image = pd.read_csv(ensemble / "per_image.csv")
    probability_columns = [f"P_{batch}" for batch in BATCHES]
    truth_index = predictions["true"].map({batch: index for index, batch in enumerate(BATCHES)})
    probabilities = predictions[probability_columns].to_numpy(dtype=float)
    selected = probabilities[np.arange(len(predictions)), truth_index.to_numpy()]
    batch_accuracy = predictions.groupby("true")["correct"].mean().to_dict()
    summary = {
        "correct": int(predictions["correct"].sum()),
        "n": len(predictions),
        **{
            f"acc_{batch}": float(batch_accuracy[batch])
            for batch in BATCHES
        },
        "image_correct": int(per_image["correct"].sum()),
        "log_loss": float(-np.log(np.clip(selected, 1e-15, 1.0)).mean()),
    }
    return summary, predictions, per_image


def build_overview(
    *,
    artifacts_dir: Path = ARTIFACTS_DIR,
    validation_dir: Path = VALIDATION_DIR,
    figures_dir: Path = FIGURES_DIR,
) -> dict[str, Any]:
    rule = pd.read_csv(artifacts_dir / "rule_per_image.csv")
    learned = pd.read_csv(artifacts_dir / "learned_per_image.csv")
    if set(rule["image_id"]) != set(learned["image_id"]):
        raise ValueError("Rule and learned KPI artifacts do not contain the same samples")
    learned_by_id = learned.set_index("image_id")
    rule_by_id = rule.set_index("image_id")
    training = []
    for image_id, rule_row in rule_by_id.iterrows():
        learned_row = learned_by_id.loc[image_id]
        training.append(
            {
                "image_id": str(image_id),
                "batch": str(rule_row["batch"]),
                "trust_flags": _text(rule_row.get("trust_flags", "")),
                "rule": {kpi: _number(rule_row.get(kpi)) for kpi in BASE},
                "learned": {kpi: _number(learned_row.get(kpi)) for kpi in BASE},
            }
        )

    batch_stats: dict[str, dict[str, dict[str, dict[str, float | None]]]] = {}
    for source, table in (("rule", rule), ("learned", learned)):
        batch_stats[source] = {}
        for kpi in BASE:
            batch_stats[source][kpi] = {}
            for batch, group in table.groupby("batch"):
                values = group[kpi].dropna().to_numpy(dtype=float)
                batch_stats[source][kpi][str(batch)] = {
                    "median": float(np.median(values)) if values.size else None,
                    "p25": float(np.percentile(values, 25)) if values.size else None,
                    "p75": float(np.percentile(values, 75)) if values.size else None,
                }

    subset_rows = pd.read_csv(validation_dir / "strat_v1" / "summary.csv")
    summary_rows: dict[str, Any] = {}
    for source, table_name in (("rule", "rule"), ("learned", "learned")):
        rows = subset_rows.loc[
            (subset_rows["table"] == table_name) & (subset_rows["strategy"] == "subset")
        ]
        if rows.empty:
            raise ValueError(f"Missing {table_name} subset row in rotating validation summary")
        summary_rows[source] = _summary_from_row(rows.iloc[0])
    combined, predictions, per_image = _combined_summary(validation_dir)

    rounds = predictions.groupby("image_id").size()
    per_sample = []
    for row in per_image.to_dict(orient="records"):
        per_sample.append(
            {
                "image_id": row["image_id"],
                "true": row["true"],
                **{column: float(row[column]) for column in (f"P_{batch}" for batch in BATCHES)},
                "n_rounds": int(rounds.get(row["image_id"], 0)),
                "trust_flags": _text(row.get("trust_flags")),
                "pred": row.get("pred"),
                "correct": bool(row["correct"]),
                "tier": row.get("tiers_seen"),
            }
        )

    confusion: dict[str, dict[str, int]] = {
        batch: {predicted: 0 for predicted in BATCHES} for batch in BATCHES
    }
    for row in predictions.groupby(["true", "pred"]).size().reset_index(name="count").to_dict(orient="records"):
        confusion[str(row["true"])][str(row["pred"])] = int(row["count"])
    tiers = []
    for tier, group in predictions.groupby("tier", sort=False):
        tiers.append(
            {
                "tier": str(tier),
                "n": len(group),
                "accuracy": float(group["correct"].mean()),
            }
        )

    figure_list = [
        {
            "name": path.stem,
            "title": path.stem.replace("_", " ").strip().title(),
            "url": f"/api/figures/{path.name}",
        }
        for path in sorted(figures_dir.glob("*.png"))
    ]
    return {
        "classes": list(BATCHES),
        "n_per_batch": {
            str(batch): int(count)
            for batch, count in rule["batch"].value_counts().sort_index().items()
        },
        "kpi_meta": KPI_META,
        "classifiers": {
            source: {"kpis": list(SOURCE_KPIS[source])} for source in ("rule", "learned")
        },
        "validation": {
            "summary": {
                "rule": summary_rows["rule"],
                "learned": summary_rows["learned"],
                "combined": combined,
            },
            "per_sample": per_sample,
            "confusion": confusion,
            "tiers": tiers,
            "note": (
                "The validated 25-sample holdout decoder was reused. The rule and learned "
                "Bayesian classifiers were fitted on all 31 samples."
            ),
        },
        "training": training,
        "batch_stats": batch_stats,
        "figures": figure_list,
        "representatives": _read_representatives(artifacts_dir),
        "legend": LEGEND,
    }


def _tiff_shape(path: Path) -> tuple[int, int]:
    try:
        with tifffile.TiffFile(path) as tiff:
            series = tiff.series[0]
            shape = series.shape
            dtype = series.dtype
    except (OSError, IndexError, ValueError, tifffile.TiffFileError) as error:
        raise ValueError(f"{path.name} is not a readable TIFF: {error}") from error
    if dtype != np.dtype(np.uint8):
        raise ValueError(f"{path.name} must be uint8, got {dtype}")
    if len(shape) == 3:
        shape = shape[:2]
    if len(shape) != 2:
        raise ValueError(f"{path.name} must be a grayscale TIFF, got shape {shape}")
    return int(shape[0]), int(shape[1])


def _save_upload(upload: UploadFile, target: Path) -> None:
    upload.file.seek(0)
    with target.open("wb") as destination:
        shutil.copyfileobj(upload.file, destination)


def create_app(
    *,
    pipeline: Any | None = None,
    data_dir: str | Path | None = None,
    jobs_dir: str | Path | None = None,
    artifacts_dir: str | Path = ARTIFACTS_DIR,
    validation_dir: str | Path = VALIDATION_DIR,
    figures_dir: str | Path = FIGURES_DIR,
    repo_root: str | Path = REPO_ROOT,
) -> FastAPI:
    data_path = Path(data_dir or os.environ.get("NEURA_DATA_DIR", "/home/ubuntu/data/neura"))
    jobs_path = Path(jobs_dir or os.environ.get("NEURA_JOBS_DIR", "/home/ubuntu/data/runs/app_jobs"))
    artifacts_path = Path(artifacts_dir)
    validation_path = Path(validation_dir)
    figures_path = Path(figures_dir)
    root_path = Path(repo_root)
    jobs_path.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if application.state.pipeline is None:
            rule_table = pd.read_csv(artifacts_path / "rule_per_image.csv")
            learned_table = pd.read_csv(artifacts_path / "learned_per_image.csv")
            cache_dir = Path("/home/ubuntu/data/runs/final")
            classifier = DualSourceClassifier(
                rule_table,
                learned_table,
                bootstrap_replicates=200,
                seed=0,
                bootstrap_cache=cache_dir / "dual_source_bootstrap.pkl",
            )
            application.state.pipeline = InferencePipeline(
                classifier,
                artifacts_path / "fusion_holdout25.pt",
                Path("/home/ubuntu/data/features/meta.json"),
            )
        yield
        application.state.executor.shutdown(wait=False, cancel_futures=False)

    app = FastAPI(
        title="NEURA sample classifier",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.pipeline = pipeline
    app.state.jobs = {}
    app.state.jobs_lock = threading.Lock()
    app.state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="neura-job")
    app.state.data_dir = data_path
    app.state.jobs_dir = jobs_path
    app.state.artifacts_dir = artifacts_path
    app.state.validation_dir = validation_path
    app.state.figures_dir = figures_path
    app.state.repo_root = root_path

    def update_job(job_id: str, step: str, progress: float) -> None:
        with app.state.jobs_lock:
            state = app.state.jobs[job_id]
            state.update(step=step, progress=max(0.0, min(float(progress), 1.0)))

    def run_job(job_id: str, image: BSEImage, sample_id: str, known_batch: str | None) -> None:
        with app.state.jobs_lock:
            app.state.jobs[job_id].update(status="running", step="starting", progress=0.01)
        try:
            result = app.state.pipeline.run(
                image=image,
                sample_id=sample_id,
                job_dir=jobs_path / job_id,
                job_id=job_id,
                known_batch=known_batch,
                progress=lambda step, progress: update_job(job_id, step, progress),
            )
            result["sample_id"] = sample_id
            result["known_batch"] = known_batch
            with app.state.jobs_lock:
                app.state.jobs[job_id].update(
                    status="done",
                    step="done",
                    progress=1.0,
                    result=result,
                )
        except Exception as error:  # noqa: BLE001
            with app.state.jobs_lock:
                app.state.jobs[job_id].update(
                    status="error",
                    step="error",
                    progress=1.0,
                    error=str(error),
                )

    def enqueue(image: BSEImage, sample_id: str, known_batch: str | None) -> str:
        job_id = uuid.uuid4().hex
        with app.state.jobs_lock:
            app.state.jobs[job_id] = {
                "status": "queued",
                "step": "queued",
                "progress": 0.0,
            }
        app.state.executor.submit(run_job, job_id, image, sample_id, known_batch)
        return job_id

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        try:
            return build_overview(
                artifacts_dir=artifacts_path,
                validation_dir=validation_path,
                figures_dir=figures_path,
            )
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @app.post("/api/classify")
    async def classify(
        bse: Annotated[UploadFile, File()],
        etd: Annotated[UploadFile, File()],
        inlens: Annotated[UploadFile, File()],
        sample_id: Annotated[str | None, Form()] = None,
    ) -> dict[str, str]:
        chosen_id = sample_id or f"sample_{uuid.uuid4().hex[:10]}"
        if not _SAFE_SAMPLE_ID.fullmatch(chosen_id):
            raise HTTPException(status_code=422, detail="sample_id may contain only letters, digits, '_' and '-'")
        job_id = uuid.uuid4().hex
        job_dir = jobs_path / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        bse_path = job_dir / f"{chosen_id}_BSE.tif"
        etd_path = job_dir / f"{chosen_id}_ETD.tif"
        inlens_path = job_dir / f"{chosen_id}_Inlens.tif"
        try:
            _save_upload(bse, bse_path)
            _save_upload(etd, etd_path)
            _save_upload(inlens, inlens_path)
            shapes = {
                "BSE": _tiff_shape(bse_path),
                "ETD": _tiff_shape(etd_path),
                "Inlens": _tiff_shape(inlens_path),
            }
            if len(set(shapes.values())) != 1:
                raise ValueError(f"Detector shape mismatch: {shapes}")
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        image = BSEImage(chosen_id, "Uploaded", bse_path)
        with app.state.jobs_lock:
            app.state.jobs[job_id] = {
                "status": "queued",
                "step": "queued",
                "progress": 0.0,
            }
        app.state.executor.submit(run_job, job_id, image, chosen_id, None)
        return {"job_id": job_id}

    @app.post("/api/classify-demo")
    def classify_demo(request: DemoRequest) -> dict[str, str]:
        samples = {sample.image_id: sample for sample in find_bse_images(data_path)}
        image = samples.get(request.image_id)
        if image is None:
            raise HTTPException(status_code=404, detail=f"Sample {request.image_id!r} was not found")
        return {"job_id": enqueue(image, image.image_id, image.batch)}

    @app.get("/api/demo-samples")
    def demo_samples() -> list[dict[str, str]]:
        return [
            {"image_id": image.image_id, "batch": image.batch}
            for image in find_bse_images(data_path)
        ]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        with app.state.jobs_lock:
            state = app.state.jobs.get(job_id)
            if state is None:
                raise HTTPException(status_code=404, detail="Job not found")
            return dict(state)

    @app.get("/api/files/{asset_path:path}")
    def files(asset_path: str) -> FileResponse:
        parts = PurePosixPath(asset_path).parts
        if len(parts) < 2 or parts[0] not in {"jobs", "artifacts"}:
            raise HTTPException(status_code=404, detail="File not found")
        base = jobs_path if parts[0] == "jobs" else artifacts_path
        relative = Path(*parts[1:])
        target = (base / relative).resolve()
        if not target.is_relative_to(base.resolve()) or not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(target)

    @app.get("/api/figures/{name}")
    def figure(name: str) -> FileResponse:
        if Path(name).name != name or not name.endswith(".png"):
            raise HTTPException(status_code=404, detail="Figure not found")
        target = figures_path / name
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Figure not found")
        return FileResponse(target)

    dist_dir = root_path / "integration" / "frontend" / "dist"
    if dist_dir.is_dir():
        app.mount("/", StaticFiles(directory=dist_dir, html=True), name="frontend")
    else:

        @app.get("/", include_in_schema=False)
        def root() -> dict[str, str]:
            return {"status": "ok", "service": "NEURA sample classifier"}

    return app


app = create_app()
