import hashlib
from pathlib import Path

import torch
from torch import nn
from torchvision import models


MICRONET_REPO = "jstuckner/microscopy-resnet50-micronet"
MICRONET_FILENAME = "resnet50_micronet_weights.pth"
MICRONET_REVISION = "c810b294acda2461775542841e6d0ba824ec46a9"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_micronet(cache_dir="/hf") -> dict:
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=MICRONET_REPO,
            filename=MICRONET_FILENAME,
            revision=MICRONET_REVISION,
            cache_dir=str(cache_dir),
        )
    )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "revision": MICRONET_REVISION,
    }


def _extract_state_dict(checkpoint):
    current = checkpoint
    while isinstance(current, dict):
        if current and all(torch.is_tensor(value) for value in current.values()):
            return current
        nested = next(
            (current[key] for key in ("state_dict", "model", "encoder", "backbone") if key in current),
            None,
        )
        if nested is None:
            break
        current = nested
    raise ValueError(f"Could not find a tensor state_dict in checkpoint keys: {list(checkpoint)[:20]}")


def _strip_prefixes(state_dict):
    result = dict(state_dict)
    for prefix in ("module.", "model.", "encoder.", "backbone.", "resnet."):
        if result and all(key.startswith(prefix) for key in result):
            result = {key[len(prefix) :]: value for key, value in result.items()}
    return result


def load_micronet_encoder(path):
    from micronet_cls.model import Encoder

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    state_dict = _strip_prefixes(_extract_state_dict(checkpoint))
    discarded_fc = sorted(key for key in state_dict if key.startswith("fc."))
    non_fc_state = {key: value for key, value in state_dict.items() if not key.startswith("fc.")}

    base = models.resnet50(weights=None)
    base.fc = nn.Identity()
    incompatible = base.load_state_dict(non_fc_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected non-fc state-dict mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    encoder = Encoder.from_resnet(base)

    verified = {}
    for key in ("conv1.weight", "layer1.0.conv1.weight", "layer4.2.conv3.weight"):
        if key in non_fc_state:
            verified[key] = bool(torch.equal(encoder.state_dict()[key], non_fc_state[key]))
            if not verified[key]:
                raise RuntimeError(f"Loaded encoder tensor differs from checkpoint: {key}")
    report = {
        "checkpoint_keys_sample": list(state_dict)[:20],
        "missing_keys": [],
        "unexpected_keys": [],
        "discarded_fc_keys": discarded_fc,
        "tensors_loaded": len(non_fc_state),
        "verified_tensors": verified,
        "verified_tensor_count": len(verified),
    }
    return encoder, report
