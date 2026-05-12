import json
from pathlib import Path

import torch

from src.model import TransformerNextEvent


def load_json_dict(path: str) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_token_to_id(path: str) -> dict:
    return load_json_dict(path)


def load_template_to_id(path: str) -> dict:
    return load_json_dict(path)


def load_model_checkpoint(model_path: str, device: str = "cpu"):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    if "model_config" not in checkpoint:
        raise ValueError("Checkpoint must contain model_config")

    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain model_state_dict")

    config = checkpoint["model_config"]

    model = TransformerNextEvent(
        vocab_size=int(config["vocab_size"]),
        d_model=int(config.get("d_model", 128)),
        nhead=int(config.get("nhead", 4)),
        num_layers=int(config.get("num_layers", 2)),
        dim_feedforward=int(config.get("dim_feedforward", 256)),
        dropout=float(config.get("dropout", 0.1)),
        max_len=int(config.get("max_len", 64)),
        pad_idx=int(config.get("pad_idx", 0)),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config, checkpoint
