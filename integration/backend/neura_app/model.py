from __future__ import annotations

import ast
import hashlib
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from neura_uq import BayesianStudentTClassifier, Measurement
from neura_uq.classifier import transform

from . import REPO_ROOT

BATCHES = ("Batch_1", "Batch_2", "Batch_3")
SOURCE_KPIS = {
    "rule": ("frac_pore", "graphite_crack_density", "si_cv_w256"),
    "learned": ("frac_pore", "graphite_aspect_ratio_median", "si_cv_w256"),
}
SOURCE_LABELS = {"rule": "rule masks", "learned": "DINO masks"}
KPI_LABELS = {
    "frac_pore": ("Porosity", "%"),
    "graphite_crack_density": ("Graphite crack density", "crack px per 10⁴ graphite px"),
    "graphite_aspect_ratio_median": ("Graphite aspect ratio (median)", ""),
    "si_cv_w256": ("Si heterogeneity (CV, 256 px windows)", ""),
    "si_contact_graphite": ("Si–graphite contact", "%"),
    "si_fraction_of_solids": ("Si fraction of solids", "%"),
    "si_ecd_d50": ("Si particle size D50", "px"),
    "si_ecd_d90": ("Si particle size D90", "px"),
    "si_crack_density": ("Si crack density", ""),
    "si_dispersion_index_w512": ("Si dispersion index (512 px)", ""),
    "cbd_fraction_of_solids": ("Binder fraction of solids", "%"),
    "graphite_alignment": ("Graphite alignment", ""),
}


def _load_base_transforms() -> dict[str, str]:
    source = REPO_ROOT / "bayesian_kpi_classifier" / "experiments" / "batch_cv" / "rotating_cv.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "BASE" for target in statement.targets
        ):
            base = ast.literal_eval(statement.value)
            if not isinstance(base, dict):
                break
            return base
    raise ValueError(f"Could not read BASE transforms from {source}")


BASE = _load_base_transforms()
KPI_META = {
    kpi: {"label": KPI_LABELS[kpi][0], "unit": KPI_LABELS[kpi][1], "transform": kind}
    for kpi, kind in BASE.items()
}


class DualSourceClassifier:
    def __init__(
        self,
        rule_table: pd.DataFrame,
        learned_table: pd.DataFrame,
        bootstrap_replicates: int = 200,
        seed: int = 0,
        bootstrap_cache: str | Path | None = None,
    ) -> None:
        if bootstrap_replicates < 1:
            raise ValueError("bootstrap_replicates must be positive")
        self.tables = {
            "rule": rule_table.copy().reset_index(drop=True),
            "learned": learned_table.copy().reset_index(drop=True),
        }
        self.bootstrap_replicates = bootstrap_replicates
        self.seed = seed
        for source, table in self.tables.items():
            required = {"image_id", "batch", *BASE}
            missing = sorted(required.difference(table.columns))
            if missing:
                raise ValueError(f"{source} KPI table is missing columns: {', '.join(missing)}")
            if set(table["batch"].astype(str)) != set(BATCHES):
                raise ValueError(f"{source} KPI table must contain all batches {BATCHES}")

        self.models = {
            source: self._fit(source, SOURCE_KPIS[source])
            for source in ("rule", "learned")
        }
        for model in self.models.values():
            if not np.allclose(list(model.log_prior_.values()), -np.log(len(BATCHES))):
                raise ValueError("The combined classifier requires uniform batch priors")

        self.bootstrap_models: dict[tuple[str, tuple[str, ...]], list[BayesianStudentTClassifier]] = {}
        self.bootstrap_cache_path = Path(bootstrap_cache) if bootstrap_cache is not None else None
        self.cache_signature = self._cache_signature()
        cached = self._load_bootstrap_cache()
        started = time.perf_counter()
        if cached is None:
            for source in ("rule", "learned"):
                self._get_bootstrap_models(source, SOURCE_KPIS[source])
            if time.perf_counter() - started > 20:
                self._save_bootstrap_cache()
        else:
            self.bootstrap_models = cached

    def _cache_signature(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(self.bootstrap_replicates).encode())
        digest.update(str(self.seed).encode())
        for source in ("rule", "learned"):
            table = self.tables[source]
            columns = ["image_id", "batch", *BASE]
            digest.update(pd.util.hash_pandas_object(table[columns], index=False).values.tobytes())
        return digest.hexdigest()

    def _load_bootstrap_cache(self) -> dict[tuple[str, tuple[str, ...]], list[BayesianStudentTClassifier]] | None:
        if self.bootstrap_cache_path is None or not self.bootstrap_cache_path.is_file():
            return None
        try:
            with self.bootstrap_cache_path.open("rb") as stream:
                signature, models = pickle.load(stream)
            if signature == self.cache_signature:
                return models
        except (OSError, EOFError, pickle.PickleError, ValueError, TypeError):
            return None
        return None

    def _save_bootstrap_cache(self) -> None:
        if self.bootstrap_cache_path is None:
            return
        self.bootstrap_cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.bootstrap_cache_path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            pickle.dump((self.cache_signature, self.bootstrap_models), stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(self.bootstrap_cache_path)

    def _fit(
        self, source: str, kpis: tuple[str, ...], table: pd.DataFrame | None = None
    ) -> BayesianStudentTClassifier:
        data = self.tables[source] if table is None else table
        values = data[list(kpis)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{source} training table has missing or non-finite values for {kpis}")
        classifier = BayesianStudentTClassifier(
            list(kpis),
            [BASE[kpi] for kpi in kpis],
            covariance="diag",
        )
        return classifier.fit(values, data["batch"].astype(str).to_numpy())

    def _fit_bootstrap_models(self, source: str, kpis: tuple[str, ...]) -> list[BayesianStudentTClassifier]:
        table = self.tables[source]
        by_batch = {batch: table.loc[table["batch"] == batch] for batch in BATCHES}
        if any(len(group) < 2 for group in by_batch.values()):
            raise ValueError("Each batch needs at least two samples for stratified bootstrap refits")
        source_offset = 0 if source == "rule" else 1_000_003
        rng = np.random.default_rng(self.seed + source_offset)
        models = []
        for _ in range(self.bootstrap_replicates):
            pieces = []
            for batch in BATCHES:
                group = by_batch[batch]
                indices = rng.integers(0, len(group), size=len(group))
                while np.unique(indices).size < 2:
                    indices = rng.integers(0, len(group), size=len(group))
                pieces.append(group.iloc[indices])
            sample = pd.concat(pieces, ignore_index=True)
            models.append(self._fit(source, kpis, sample))
        return models

    def _get_bootstrap_models(
        self, source: str, kpis: tuple[str, ...]
    ) -> list[BayesianStudentTClassifier]:
        key = (source, tuple(kpis))
        if key not in self.bootstrap_models:
            self.bootstrap_models[key] = self._fit_bootstrap_models(source, tuple(kpis))
        return self.bootstrap_models[key]

    @staticmethod
    def _interval(probabilities: np.ndarray) -> dict[str, list[float]]:
        return {
            batch: [
                float(np.percentile(probabilities[:, index], 5)),
                float(np.percentile(probabilities[:, index], 95)),
            ]
            for index, batch in enumerate(BATCHES)
        }

    @staticmethod
    def _finite_value(values: dict[str, Any], kpi: str) -> float | None:
        value = values.get(kpi)
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    @staticmethod
    def _prediction(
        model: BayesianStudentTClassifier, values: dict[str, float]
    ):
        return model.predict(Measurement.from_sd([values[kpi] for kpi in model.kpis]))

    def _bootstrap_probabilities(
        self,
        source: str,
        kpis: tuple[str, ...],
        values: dict[str, float],
    ) -> np.ndarray:
        models = self._get_bootstrap_models(source, kpis)
        return np.asarray(
            [
                [self._prediction(model, values).probabilities[batch] for batch in BATCHES]
                for model in models
            ],
            dtype=float,
        )

    def _baseline(self, sample_kpis: dict[str, dict[str, Any]], typicality: dict[str, dict[str, float]]) -> dict:
        rows = []
        for source in ("rule", "learned"):
            table = self.tables[source]
            batch3 = table.loc[table["batch"] == "Batch_3"]
            for kpi, kind in BASE.items():
                value = self._finite_value(sample_kpis[source], kpi)
                ref = batch3[kpi].to_numpy(dtype=float)
                ref = ref[np.isfinite(ref)]
                transformed = transform(ref[:, None], [kind]).ravel()
                median = float(np.median(ref)) if ref.size else None
                p05 = float(np.percentile(ref, 5)) if ref.size else None
                p95 = float(np.percentile(ref, 95)) if ref.size else None
                z = None
                percentile = None
                direction = "typical"
                if value is not None and ref.size:
                    transformed_value = float(transform(np.asarray([[value]]), [kind])[0, 0])
                    sd = float(transformed.std(ddof=1)) if transformed.size > 1 else 0.0
                    if sd > 0:
                        z = (transformed_value - float(transformed.mean())) / sd
                        direction = "higher" if z > 1 else ("lower" if z < -1 else "typical")
                    percentile = float(100 * np.mean(ref <= value))
                rows.append(
                    {
                        "source": source,
                        "kpi": kpi,
                        "label": KPI_META[kpi]["label"],
                        "value": value,
                        "median": median,
                        "p05": p05,
                        "p95": p95,
                        "z": None if z is None or not np.isfinite(z) else float(z),
                        "percentile": percentile,
                        "direction": direction,
                    }
                )
        return {
            "batch": "Batch_3",
            "typicality": {source: typicality[source]["Batch_3"] for source in ("rule", "learned")},
            "consistent_with_baseline": all(
                typicality[source]["Batch_3"] >= 0.05 for source in ("rule", "learned")
            ),
            "kpis": rows,
        }

    @staticmethod
    def _display(kpi: str, value: float) -> str:
        unit = KPI_META[kpi]["unit"]
        if unit == "%":
            return f"{value * 100:.1f}%"
        return f"{value:.3g}"

    def _explanations(self, contributions: list[dict], top: str, runner_up: str) -> list[str]:
        explanations = []
        for contribution in contributions:
            if len(explanations) == 5:
                break
            favored = contribution["favours"]
            if not favored:
                continue
            other = runner_up if favored == top else top
            source = contribution["source"]
            kpi = contribution["kpi"]
            display = contribution["display"]
            table = self.tables[source]
            medians = {
                batch: float(table.loc[table["batch"] == batch, kpi].median())
                for batch in (favored, other)
            }
            name = {batch: batch.replace("_", " ") for batch in (favored, other)}
            explanations.append(
                f"{KPI_META[kpi]['label']} {display} ({SOURCE_LABELS[source]}) favours {name[favored]} over "
                f"{name[other]}: typical {name[favored]} is {self._display(kpi, medians[favored])}, "
                f"{name[other]} {self._display(kpi, medians[other])}."
            )
        return explanations

    def classify(
        self,
        rule_kpis: dict[str, Any],
        learned_kpis: dict[str, Any],
    ) -> dict:
        supplied = {"rule": rule_kpis, "learned": learned_kpis}
        warnings: list[str] = []
        predictions = {}
        active_values = {}
        for source, values in supplied.items():
            active = tuple(
                kpi for kpi in SOURCE_KPIS[source] if self._finite_value(values, kpi) is not None
            )
            missing = sorted(set(SOURCE_KPIS[source]) - set(active))
            if not active:
                raise ValueError(f"{source} classifier has no usable KPI values")
            for kpi in missing:
                warnings.append(
                    f"Missing KPI {kpi} for {SOURCE_LABELS[source]}; refit without it for this sample."
                )
            active_values[source] = {kpi: float(values[kpi]) for kpi in active}
            model = self.models[source] if active == SOURCE_KPIS[source] else self._fit(source, active)
            predictions[source] = self._prediction(model, active_values[source])

        source_intervals = {}
        bootstrap_probabilities = {}
        for source in ("rule", "learned"):
            active = tuple(predictions[source].kpis)
            bootstrap_probabilities[source] = self._bootstrap_probabilities(
                source, active, active_values[source]
            )
            source_intervals[source] = self._interval(bootstrap_probabilities[source])

        probabilities = {
            batch: float(
                np.sqrt(predictions["rule"].probabilities[batch] * predictions["learned"].probabilities[batch])
            )
            for batch in BATCHES
        }
        total = sum(probabilities.values())
        probabilities = {batch: value / total for batch, value in probabilities.items()}
        order = sorted(BATCHES, key=probabilities.__getitem__, reverse=True)
        top, runner_up = order[:2]

        combined_bootstrap = np.sqrt(
            bootstrap_probabilities["rule"] * bootstrap_probabilities["learned"]
        )
        combined_bootstrap /= combined_bootstrap.sum(axis=1, keepdims=True)
        combined_interval = self._interval(combined_bootstrap)

        contributions = []
        for source in ("rule", "learned"):
            prediction = predictions[source]
            for kpi in prediction.kpis:
                per_batch = {
                    batch: 0.5 * prediction.kpi_log_likelihood[batch][kpi]
                    for batch in BATCHES
                }
                evidence = per_batch[top] - per_batch[runner_up]
                contributions.append(
                    {
                        "source": source,
                        "kpi": kpi,
                        "label": KPI_META[kpi]["label"],
                        "value": float(supplied[source][kpi]),
                        "display": self._display(kpi, float(supplied[source][kpi])),
                        "per_batch": per_batch,
                        "favours": top if evidence > 0 else (runner_up if evidence < 0 else ""),
                        "evidence": float(evidence),
                    }
                )
        contributions.sort(key=lambda item: abs(item["evidence"]), reverse=True)

        typicality = {
            source: predictions[source].typicality_pvalue for source in ("rule", "learned")
        }
        source_results = {}
        for source in ("rule", "learned"):
            prediction = predictions[source]
            source_results[source] = {
                "kpis": {
                    kpi: self._finite_value(supplied[source], kpi)
                    for kpi in BASE
                },
                "probabilities": prediction.probabilities,
                "interval": source_intervals[source],
                "typicality": prediction.typicality_pvalue,
                "evidence": prediction.kpi_evidence,
            }
        max_probability = probabilities[top]
        return {
            "sample_id": "",
            "known_batch": None,
            "timings": {},
            "warnings": warnings,
            "trust_flags": "",
            "prediction": {
                "predicted": top,
                "runner_up": runner_up,
                "probabilities": probabilities,
                "interval": combined_interval,
                "tier": "high" if max_probability >= 0.9 else ("medium" if max_probability >= 0.6 else "review"),
                "ambiguous": max_probability < 0.7,
                "outlier": any(max(values.values()) < 0.01 for values in typicality.values()),
            },
            "classifiers": source_results,
            "contributions": contributions,
            "explanation": self._explanations(contributions, top, runner_up),
            "baseline": self._baseline(supplied, typicality),
        }
