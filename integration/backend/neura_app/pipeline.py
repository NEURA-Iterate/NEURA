from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from anode_qc.cli import load_detectors
from anode_qc.config import Config
from anode_qc.data import BSEImage
from anode_qc.kpis import compute_kpis, crack_mask, kpis_from_labels
from anode_qc.pipeline import segment_image
from anode_qc.qc import densest_si_crop
from anode_qc.report import trust_flags
from anode_qc.segment import CBD, GAP, GRAPHITE, PORE, SI
from anode_qc.uncertainty import pixel_intervals
from experiments.learned_masks.learned_kpis import LEARNED_TO_ANODE
from phaseseg.backbone import Dinov2Backbone
from phaseseg.data import load_triplet
from phaseseg.evaluate import sliding_window_probabilities
from phaseseg.features import extract_tiled_features
from phaseseg.heads import build_head
from PIL import Image
from skimage.morphology import remove_small_objects

from .model import BASE, DualSourceClassifier

LEGEND = {
    "si": {"label": "Si", "color": "#ff8c00"},
    "graphite": {"label": "Graphite", "color": "#59636e"},
    "pore": {"label": "Pore", "color": "#1e3cff"},
    "binder": {"label": "Binder", "color": "#28c828"},
    "gap": {"label": "Gap", "color": "#ff00ff"},
    "crack": {"label": "Crack", "color": "#ff3030"},
}

_MASK_COLORS = {
    SI: (255, 140, 0),
    GRAPHITE: (89, 99, 110),
    PORE: (30, 60, 255),
    CBD: (40, 200, 40),
    GAP: (255, 0, 255),
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _display_rgb(raw: np.ndarray, labels: np.ndarray | None = None, factor: int = 1) -> np.ndarray:
    gray = raw[::factor, ::factor]
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    if labels is None:
        return rgb.astype(np.uint8)
    small_labels = labels[::factor, ::factor]
    for label, color in _MASK_COLORS.items():
        mask = small_labels == label
        rgb[mask] = rgb[mask] * 0.55 + np.asarray(color, dtype=np.float32) * 0.45
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _save_panel(path: Path, pixels: np.ndarray) -> None:
    if path.stem.endswith("cracks"):
        gray_levels = np.linspace(0, 255, 15).round().astype(np.uint8)
        gray = pixels.astype(np.float32).mean(axis=2)
        palette_indices = np.argmin(
            np.abs(gray[..., None] - gray_levels.astype(np.float32)),
            axis=2,
        ).astype(np.uint8)
        crack_pixels = (
            (pixels[..., 0] == 255)
            & (pixels[..., 1] == 48)
            & (pixels[..., 2] == 48)
        )
        palette_indices[crack_pixels] = 15
        panel = Image.fromarray(palette_indices, mode="P")
        palette = np.column_stack(
            [
                np.append(gray_levels, 255),
                np.append(gray_levels, 48),
                np.append(gray_levels, 48),
            ]
        ).astype(np.uint8)
        panel.putpalette(np.pad(palette.ravel(), (0, 768 - palette.size)).tolist())
    else:
        panel = Image.fromarray(pixels).quantize(colors=16, dither=Image.Dither.NONE)
    panel.save(path, format="PNG", optimize=True, bits=4, compress_level=9)


def render_panels(
    raw: np.ndarray,
    rule_labels: np.ndarray,
    learned_labels: np.ndarray,
    etd_n: np.ndarray,
    cfg: Config,
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    factor = max(1, math.ceil(raw.shape[1] / 1400))
    names = {
        "bse": "bse.png",
        "rule_overlay": "rule_overlay.png",
        "learned_overlay": "learned_overlay.png",
        "cracks": "cracks.png",
        "zoom_bse": "zoom_bse.png",
        "zoom_rule": "zoom_rule.png",
        "zoom_learned": "zoom_learned.png",
        "zoom_cracks": "zoom_cracks.png",
    }
    _save_panel(output_dir / names["bse"], _display_rgb(raw, factor=factor))
    _save_panel(
        output_dir / names["rule_overlay"],
        _display_rgb(raw, rule_labels, factor=factor),
    )
    _save_panel(
        output_dir / names["learned_overlay"],
        _display_rgb(raw, learned_labels, factor=factor),
    )

    rule_cracks = crack_mask(
        etd_n,
        (rule_labels == SI) | (rule_labels == GRAPHITE),
        cfg.multimodal.crack_k,
    )
    learned_cracks = crack_mask(
        etd_n,
        (learned_labels == SI) | (learned_labels == GRAPHITE),
        cfg.multimodal.crack_k,
    )
    crack_panels = [
        _display_rgb(raw, factor=factor),
        _display_rgb(raw, factor=factor),
    ]
    for panel, cracks in zip(crack_panels, (rule_cracks, learned_cracks), strict=True):
        full = np.repeat(np.repeat(cracks, 2, axis=0), 2, axis=1)[: raw.shape[0], : raw.shape[1]]
        visible = full[::factor, ::factor]
        panel[visible] = (255, 48, 48)
    crack_height = max(panel.shape[0] for panel in crack_panels)
    crack_panel = np.concatenate(
        [
            np.pad(panel, ((0, crack_height - panel.shape[0]), (0, 0), (0, 0)))
            for panel in crack_panels
        ],
        axis=0,
    )
    _save_panel(output_dir / names["cracks"], crack_panel)

    y, x = densest_si_crop(rule_labels == SI, size=(600, 600))
    crop = (slice(y, y + 600), slice(x, x + 600))
    raw_crop = raw[crop]
    rule_crop = rule_labels[crop]
    learned_crop = learned_labels[crop]
    rule_crack_full = np.repeat(np.repeat(rule_cracks, 2, axis=0), 2, axis=1)[: raw.shape[0], : raw.shape[1]]
    learned_crack_full = np.repeat(
        np.repeat(learned_cracks, 2, axis=0), 2, axis=1
    )[: raw.shape[0], : raw.shape[1]]
    _save_panel(output_dir / names["zoom_bse"], _display_rgb(raw_crop))
    _save_panel(output_dir / names["zoom_rule"], _display_rgb(raw_crop, rule_crop))
    _save_panel(output_dir / names["zoom_learned"], _display_rgb(raw_crop, learned_crop))
    zoom_crack_panels = [_display_rgb(raw_crop), _display_rgb(raw_crop)]
    for panel, cracks in zip(
        zoom_crack_panels,
        (rule_crack_full[crop], learned_crack_full[crop]),
        strict=True,
    ):
        panel[cracks] = (255, 48, 48)
    zoom_cracks = np.concatenate(zoom_crack_panels, axis=0)
    _save_panel(output_dir / names["zoom_cracks"], zoom_cracks)
    return names


def map_learned_labels(prediction: np.ndarray, cfg: Config) -> np.ndarray:
    labels = LEARNED_TO_ANODE[prediction]
    si = labels == SI
    specks = si & ~remove_small_objects(si, min_size=cfg.cleanup.si_min_area_px)
    labels = labels.copy()
    labels[specks] = CBD
    return labels


class InferencePipeline:
    def __init__(
        self,
        classifier: DualSourceClassifier,
        checkpoint_path: str | Path,
        feature_meta_path: str | Path,
    ) -> None:
        torch.set_num_threads(os.cpu_count() or 1)
        self.classifier = classifier
        self.cfg = Config()
        metadata = json.loads(Path(feature_meta_path).read_text(encoding="utf-8"))
        self.tile = int(metadata["tile"])
        self.overlap = int(metadata["overlap"])
        self.feature_batch_size = int(metadata["batch_size"])
        self.patch_size = int(metadata["patch_size"])
        self.backbone = Dinov2Backbone(
            metadata["model_id"],
            upsample=int(metadata.get("upsample", 1)),
            device="cpu",
        ).eval()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.head = build_head(
            checkpoint["head"],
            checkpoint["in_dim"],
            checkpoint["n_classes"],
        )
        self.head.load_state_dict(checkpoint["state_dict"])
        self.head.eval()

    @staticmethod
    def _notify(progress: Callable[[str, float], None] | None, step: str, value: float) -> None:
        if progress is not None:
            progress(step, value)

    def run(
        self,
        image: BSEImage,
        sample_id: str,
        job_dir: str | Path,
        job_id: str,
        known_batch: str | None = None,
        progress: Callable[[str, float], None] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        output_dir = Path(job_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timings: dict[str, float] = {}

        self._notify(progress, "rule analysis", 0.05)
        step = time.perf_counter()
        raw, etd, inlens = load_detectors(image, self.cfg)
        if etd is None or inlens is None:
            raise ValueError("BSE, ETD (or SE), and Inlens detector files are all required")
        if etd.shape != raw.shape:
            raise ValueError(f"ETD shape {etd.shape} does not match BSE shape {raw.shape}")
        if inlens.shape != raw.shape:
            raise ValueError(f"Inlens shape {inlens.shape} does not match BSE shape {raw.shape}")
        rule_result = segment_image(raw, self.cfg, etd, inlens)
        rule_kpis, _, _ = compute_kpis(
            rule_result,
            self.cfg.kpis,
            self.cfg.multimodal.crack_k,
        )
        if self.cfg.uncertainty.pixel:
            etd_n = rule_result.mm.etd_n if rule_result.mm is not None else None
            rule_kpis.update(
                pixel_intervals(
                    rule_result,
                    rule_kpis,
                    lambda labels: kpis_from_labels(
                        labels,
                        rule_result.n,
                        self.cfg.kpis,
                        etd_n,
                        self.cfg.multimodal.crack_k,
                    )[0],
                    self.cfg.uncertainty,
                    self.cfg.cleanup.si_min_area_px,
                )
            )
        flags = str(trust_flags(pd.DataFrame([rule_kpis])).iloc[0])
        timings["rule"] = time.perf_counter() - step

        self._notify(progress, "feature extraction", 0.30)
        step = time.perf_counter()
        triplet = load_triplet(image, border_px=self.cfg.preprocess.border_px)
        features = extract_tiled_features(
            self.backbone,
            triplet,
            tile=self.tile,
            overlap=self.overlap,
            batch_size=self.feature_batch_size,
        ).astype(np.float16)
        timings["features"] = time.perf_counter() - step

        self._notify(progress, "mask prediction", 0.65)
        step = time.perf_counter()
        probabilities = sliding_window_probabilities(
            self.head,
            triplet,
            features,
            tile_size=224,
            overlap=56,
            patch_size=self.patch_size,
            device="cpu",
        )
        prediction = probabilities.argmax(axis=0).astype(np.uint8)
        if prediction.shape != raw.shape:
            raise ValueError(
                f"Predicted mask shape {prediction.shape} does not match loaded sample shape {raw.shape}"
            )
        learned_labels = map_learned_labels(prediction, self.cfg)
        etd_n = rule_result.mm.etd_n if rule_result.mm is not None else None
        if etd_n is None:
            raise ValueError("ETD or SE data is required for learned crack KPIs")
        learned_kpis, _, _ = kpis_from_labels(
            learned_labels,
            rule_result.n,
            self.cfg.kpis,
            etd_n,
            self.cfg.multimodal.crack_k,
        )
        timings["prediction_and_learned_kpis"] = time.perf_counter() - step

        self._notify(progress, "classifying", 0.83)
        step = time.perf_counter()
        classification = self.classifier.classify(rule_kpis, learned_kpis)
        timings["classification"] = time.perf_counter() - step

        self._notify(progress, "creating panels", 0.91)
        step = time.perf_counter()
        image_files = render_panels(
            raw,
            rule_result.labels,
            learned_labels,
            etd_n,
            self.cfg,
            output_dir,
        )
        image_urls = {
            key: f"/api/files/jobs/{job_id}/{filename}" for key, filename in image_files.items()
        }
        timings["visuals"] = time.perf_counter() - step
        timings["total"] = time.perf_counter() - started

        kpis = {"rule": {}, "learned": {}}
        for kpi in BASE:
            rule_value = rule_kpis.get(kpi)
            learned_value = learned_kpis.get(kpi)
            kpis["rule"][kpi] = {
                "value": rule_value,
                "lo": rule_kpis.get(f"{kpi}_pix_lo"),
                "hi": rule_kpis.get(f"{kpi}_pix_hi"),
            }
            kpis["learned"][kpi] = {"value": learned_value}
        result = {
            "sample_id": sample_id,
            "known_batch": known_batch,
            "timings": timings,
            "warnings": classification["warnings"],
            "trust_flags": flags,
            "prediction": classification["prediction"],
            "classifiers": classification["classifiers"],
            "contributions": classification["contributions"],
            "explanation": classification["explanation"],
            "kpis": kpis,
            "baseline": classification["baseline"],
            "images": image_urls,
            "legend": LEGEND,
        }
        (output_dir / "result.json").write_text(
            json.dumps(_json_safe(result), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        self._notify(progress, "done", 1.0)
        return _json_safe(result)
