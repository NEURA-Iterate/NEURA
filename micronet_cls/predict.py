import json
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from micronet_cls.model import Classifier, Encoder, InputPrep
from micronet_cls.raw_preprocess import normalize_single_image, tile_whole_image


def load_classifier_checkpoint(model_ckpt, device=None):
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    checkpoint = torch.load(model_ckpt, map_location="cpu", weights_only=False)
    encoder = Encoder()
    model = Classifier(encoder)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    prep = InputPrep(checkpoint.get("input_prep_mode", "none"))
    model.to(device).eval()
    prep.to(device).eval()
    return model, prep, checkpoint, device


def _probability_dict(probabilities):
    return {
        "Batch 1": float(probabilities[0]),
        "Batch 2": float(probabilities[1]),
        "Batch 3": float(probabilities[2]),
    }


def _validate_csv_array(array, path):
    if array.shape != (512, 512):
        raise ValueError(f"CSV crop must be 512x512, got {array.shape}: {path}")
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError(f"CSV crop must be finite and in [0, 1]: {path}")


def predict_csv_with_model(model, prep, array, device, batch_size=32):
    array = np.asarray(array, dtype=np.float32)
    _validate_csv_array(array, "array")
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(prep(tensor.to(device))).float(), dim=1)[0]
    return probabilities.cpu().numpy()


def _read_raw_image(path):
    with Image.open(path) as image:
        array = np.asarray(image)
    warning = None
    if array.ndim == 3:
        if array.shape[2] < 1:
            raise ValueError(f"Raw image has no channels: {path}")
        differences = np.abs(array.astype(np.int16) - array[..., :1].astype(np.int16))
        if differences.max(initial=0) > 1:
            warning = "RGB channels differ beyond the near-identical threshold; using channel 0."
            warnings.warn(warning, RuntimeWarning)
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Expected a grayscale or RGB microscopy image, got {array.shape}: {path}")
    return array, warning


def _agnostic_scale_for_checkpoint(model_ckpt, agnostic_scale):
    if agnostic_scale is not None:
        return float(agnostic_scale)
    run_dir = Path(model_ckpt).resolve().parent.parent
    path = run_dir / "profile_agnostic.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Raw prediction requires a DA-v1 scale; expected {path} or an explicit agnostic_scale"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["scale"])


def _predict_raw_with_model(model, prep, path, scale, device, batch_size=16):
    raw, channel_warning = _read_raw_image(path)
    if raw.shape[0] < 512 or raw.shape[1] < 512:
        raise ValueError("Image yields zero crops under the edge policy")
    normalized, normalization_info = normalize_single_image(
        raw, scale, return_info=True
    )
    crops, positions = tile_whole_image(normalized, 512)
    if not crops:
        raise ValueError("Image yields zero crops under the edge policy")
    predictions = []
    with torch.no_grad():
        for start in range(0, len(crops), batch_size):
            batch = torch.from_numpy(np.stack(crops[start : start + batch_size])).unsqueeze(1)
            probabilities = torch.softmax(
                model(prep(batch.to(device))).float(), dim=1
            ).cpu().numpy()
            predictions.extend(probabilities)
    mean_probability = np.stack(predictions).mean(axis=0)
    warning_text = (
        "Raw mode uses the detector-agnostic DA-v1 variant; measured parity is recorded in raw_parity.json."
    )
    if channel_warning:
        warning_text += " " + channel_warning
    if normalization_info["fallback_used"]:
        warning_text += " Flat-surface fallback was used."
    return {
        "predicted_batch": f"Batch {int(mean_probability.argmax()) + 1}",
        "probabilities": _probability_dict(mean_probability),
        "n_crops": len(crops),
        "tile_positions": [[int(y), int(x)] for y, x in positions],
        "per_crop_probabilities": [_probability_dict(values) for values in predictions],
        "mode": "raw-DA-v1",
        "warning": warning_text,
        "fallback_used": normalization_info["fallback_used"],
    }


def predict_with_model(
    model,
    prep,
    path,
    mode="csv",
    device=None,
    agnostic_scale=None,
    batch_size=16,
):
    device = torch.device(device or next(model.parameters()).device)
    path = Path(path)
    if mode == "csv":
        array = np.loadtxt(path, delimiter=",", dtype=np.float32)
        probabilities = predict_csv_with_model(model, prep, array, device)
        return {
            "predicted_batch": f"Batch {int(probabilities.argmax()) + 1}",
            "probabilities": _probability_dict(probabilities),
            "n_crops": 1,
            "mode": "csv",
            "warning": None,
        }
    if mode != "raw":
        raise ValueError("mode must be 'csv' or 'raw'")
    if agnostic_scale is None:
        raise ValueError("predict_with_model raw mode requires an explicit agnostic_scale")
    return _predict_raw_with_model(
        model, prep, path, float(agnostic_scale), device, batch_size
    )


def predict(model_ckpt, path, mode="csv", agnostic_scale=None):
    model, prep, _, device = load_classifier_checkpoint(model_ckpt)
    if mode == "csv":
        array = np.loadtxt(path, delimiter=",", dtype=np.float32)
        probabilities = predict_csv_with_model(model, prep, array, device)
        return {
            "predicted_batch": f"Batch {int(probabilities.argmax()) + 1}",
            "probabilities": _probability_dict(probabilities),
            "n_crops": 1,
            "mode": "csv",
            "warning": None,
        }
    if mode != "raw":
        raise ValueError("mode must be 'csv' or 'raw'")
    scale = _agnostic_scale_for_checkpoint(model_ckpt, agnostic_scale)
    return _predict_raw_with_model(model, prep, path, scale, device, batch_size=16)


def predict_bytes(model_ckpt, file_bytes, filename, mode="csv", agnostic_scale=None):
    suffix = Path(filename).suffix or (".csv" if mode == "csv" else ".tif")
    with tempfile.TemporaryDirectory(prefix="micronet-query-") as directory:
        path = Path(directory) / f"query{suffix}"
        path.write_bytes(file_bytes)
        return predict(model_ckpt, path, mode=mode, agnostic_scale=agnostic_scale)
