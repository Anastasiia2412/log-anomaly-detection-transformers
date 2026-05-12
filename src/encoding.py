import hashlib

from config import UNK_OFFSET, NUM_UNK_BUCKETS


def stable_hash_bucket(token, num_buckets=NUM_UNK_BUCKETS):
    h = hashlib.md5(str(token).encode("utf-8")).hexdigest()
    return int(h, 16) % num_buckets


def encode_token(token, token_to_id):
    if token in token_to_id:
        return int(token_to_id[token])

    bucket = stable_hash_bucket(token)
    return UNK_OFFSET + bucket


def encode_sequence(seq, token_to_id):
    return [encode_token(token, token_to_id) for token in seq]


def is_unk_id(x):
    return UNK_OFFSET <= int(x) < UNK_OFFSET + NUM_UNK_BUCKETS


def add_encoded_features(df, token_to_id):
    df = df.copy()

    df["EncodedFeatures"] = df["Features"].apply(
        lambda seq: encode_sequence(seq, token_to_id)
    )

    df["UnkCount"] = df["EncodedFeatures"].apply(
        lambda seq: sum(is_unk_id(x) for x in seq)
    )

    df["UnkRatio"] = df.apply(
        lambda row: row["UnkCount"] / row["SeqLen"] if row["SeqLen"] else 0.0,
        axis=1,
    )

    df["VocabularyCoverage"] = 1.0 - df["UnkRatio"]

    return df
