import json
import pickle
from pathlib import Path
import pandas as pd


def build_vocab_from_events(event_tokens):
    unique_tokens = sorted(set(event_tokens))

    token_to_id = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for token in unique_tokens:
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)

    id_to_token = {idx: token for token, idx in token_to_id.items()}

    return token_to_id, id_to_token


def encode_sequence(features, token_to_id):
    unk_id = token_to_id.get("<UNK>", 1)
    return [token_to_id.get(token, unk_id) for token in features]


def add_encoded_features(df: pd.DataFrame, token_to_id: dict) -> pd.DataFrame:
    df = df.copy()
    unk_id = token_to_id.get("<UNK>", 1)

    df["EncodedFeatures"] = df["Features"].apply(
        lambda seq: [token_to_id.get(token, unk_id) for token in seq]
    )

    df["NumUnknown"] = df["EncodedFeatures"].apply(lambda seq: sum(x == unk_id for x in seq))
    df["UnkRatio"] = df.apply(
        lambda row: row["NumUnknown"] / row["SeqLen"] if row["SeqLen"] else 0.0,
        axis=1,
    )
    df["VocabularyCoverage"] = 1.0 - df["UnkRatio"]

    return df


def save_vocab(token_to_id: dict, path: str):
    path = Path(path)

    if path.suffix == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(token_to_id, f, ensure_ascii=False, indent=2)
    else:
        with open(path, "wb") as f:
            pickle.dump(token_to_id, f)


def load_vocab(path: str):
    path = Path(path)

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    with open(path, "rb") as f:
        return pickle.load(f)
