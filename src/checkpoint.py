import io
import json
import pickle
import torch

from src.model import build_transformer_from_config


def load_token_to_id_from_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()

    if name.endswith(".json"):
        return json.loads(raw.decode("utf-8"))

    if name.endswith(".pkl") or name.endswith(".pickle"):
        return pickle.loads(raw)

    raise ValueError("Unsupported vocab format. Use .json or .pkl")


def load_model_from_uploaded_checkpoint(
    uploaded_file,
    vocab_size,
    device="cpu",
    default_config=None,
):
    if default_config is None:
        default_config = {
            "pad_idx": 0,
            "d_model": 128,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "max_len": 64,
        }

    raw = uploaded_file.read()
    checkpoint = torch.load(
        io.BytesIO(raw),
        map_location=device,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_config = checkpoint.get("model_config", default_config)
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model_config = checkpoint.get("model_config", default_config)
        state_dict = checkpoint["state_dict"]
    else:
        model_config = default_config
        state_dict = checkpoint

    model = build_transformer_from_config(
        vocab_size=vocab_size,
        config=model_config,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, model_config
