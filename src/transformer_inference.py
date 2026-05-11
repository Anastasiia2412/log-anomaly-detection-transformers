import numpy as np
import torch


def is_unk_id(x, unk_ids=None):
    if unk_ids is None:
        unk_ids = {1}
    return int(x) in set(unk_ids)


@torch.no_grad()
def sequence_local_nll_known_only(
    model,
    encoded_seq,
    pad_idx=0,
    max_len=64,
    device="cpu",
    batch_size=512,
    unk_ids=None,
):
    model.eval()

    if len(encoded_seq) < 2:
        return []

    prefixes = []
    masks = []
    targets = []

    for i in range(1, len(encoded_seq)):
        prefix = encoded_seq[:i]
        target = encoded_seq[i]

        if is_unk_id(target, unk_ids=unk_ids):
            continue

        if len(prefix) > max_len:
            prefix = prefix[-max_len:]

        length = len(prefix)

        padded = prefix + [pad_idx] * (max_len - length)
        mask = [1] * length + [0] * (max_len - length)

        prefixes.append(padded)
        masks.append(mask)
        targets.append(target)

    if len(targets) == 0:
        return []

    all_nll = []

    for start in range(0, len(prefixes), batch_size):
        end = start + batch_size

        input_ids = torch.tensor(
            prefixes[start:end],
            dtype=torch.long,
            device=device,
        )

        attention_mask = torch.tensor(
            masks[start:end],
            dtype=torch.long,
            device=device,
        )

        batch_targets = torch.tensor(
            targets[start:end],
            dtype=torch.long,
            device=device,
        )

        logits = model(input_ids, attention_mask)
        log_probs = torch.log_softmax(logits, dim=1)

        nll = -log_probs[
            torch.arange(len(batch_targets), device=device),
            batch_targets,
        ]

        all_nll.extend(nll.detach().cpu().numpy().tolist())

    return all_nll


def aggregate_scores(local_scores, method="max", topk=3):
    if len(local_scores) == 0:
        return 0.0

    arr = np.array(local_scores, dtype=float)

    if method == "mean":
        return float(arr.mean())

    if method == "max":
        return float(arr.max())

    if method == "p95":
        return float(np.percentile(arr, 95))

    if method == "topk_mean":
        k = min(topk, len(arr))
        return float(np.sort(arr)[-k:].mean())

    raise ValueError(f"Unknown aggregation method: {method}")


def compute_transformer_scores_for_df(
    df,
    model,
    pad_idx=0,
    max_len=64,
    device="cpu",
    method="max",
    topk=3,
    batch_size=512,
    unk_ids=None,
):
    scores = []
    local_scores_all = []

    for seq in df["EncodedFeatures"]:
        local_scores = sequence_local_nll_known_only(
            model=model,
            encoded_seq=seq,
            pad_idx=pad_idx,
            max_len=max_len,
            device=device,
            batch_size=batch_size,
            unk_ids=unk_ids,
        )

        score = aggregate_scores(
            local_scores,
            method=method,
            topk=topk,
        )

        scores.append(score)
        local_scores_all.append(local_scores)

    return np.array(scores), local_scores_all


def add_transformer_predictions(
    df,
    model,
    threshold,
    pad_idx=0,
    max_len=64,
    device="cpu",
    method="max",
    topk=3,
    batch_size=512,
    unk_ids=None,
):
    df = df.copy()

    scores, local_scores = compute_transformer_scores_for_df(
        df=df,
        model=model,
        pad_idx=pad_idx,
        max_len=max_len,
        device=device,
        method=method,
        topk=topk,
        batch_size=batch_size,
        unk_ids=unk_ids,
    )

    df["anomaly_score"] = scores
    df["prediction"] = (df["anomaly_score"] >= threshold).astype(int)
    df["local_nll_scores"] = local_scores

    def risk(score):
        if score >= threshold * 1.5:
            return "high"
        if score >= threshold:
            return "medium"
        return "low"

    df["risk_level"] = df["anomaly_score"].apply(risk)

    return df
