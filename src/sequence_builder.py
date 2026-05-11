import pandas as pd


def build_sequences_by_time_window(
    df: pd.DataFrame,
    window: str = "30min",
    min_seq_len: int = 5,
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["window_start"] = df["timestamp"].dt.floor(window)

    sequences = []

    for window_start, group in df.groupby("window_start"):
        group = group.sort_values("timestamp")

        features = group["EventToken"].tolist()
        timestamps = group["timestamp"].tolist()

        if len(features) < min_seq_len:
            continue

        time_intervals = [0.0]
        for i in range(1, len(timestamps)):
            delta = (timestamps[i] - timestamps[i - 1]).total_seconds()
            time_intervals.append(float(delta))

        latency = float((timestamps[-1] - timestamps[0]).total_seconds())

        sequences.append({
            "WindowStart": window_start,
            "WindowEnd": window_start + pd.Timedelta(window),
            "Features": features,
            "TimeInterval": time_intervals,
            "Latency": latency,
            "SeqLen": len(features),
            "LogNames": sorted(group["LogName"].unique().tolist()),
            "Sources": sorted(group["Source"].unique().tolist()),
            "Levels": sorted(group["Level"].unique().tolist()),
            "EventIDs": sorted(group["Event ID"].astype(str).unique().tolist()),
            "RawMessages": group["Message"].tolist(),
            "ReviewTemplates": group["EventTemplateForReview"].tolist(),
        })

    return pd.DataFrame(sequences)


def split_long_sequence_row(row, max_len: int = 64, stride: int = 32):
    features = row["Features"]
    intervals = row["TimeInterval"]
    raw_messages = row["RawMessages"]
    review_templates = row["ReviewTemplates"]

    n = len(features)

    if n <= max_len:
        new_row = row.copy()
        new_row["ChunkID"] = 0
        new_row["OriginalSeqLen"] = n
        new_row["ChunkStartPos"] = 0
        new_row["ChunkEndPos"] = n
        new_row["WasChunked"] = False
        return [new_row]

    chunks = []
    starts = list(range(0, n - max_len + 1, stride))

    last_start = n - max_len
    if starts[-1] != last_start:
        starts.append(last_start)

    for chunk_id, start in enumerate(starts):
        end = start + max_len

        new_row = row.copy()
        new_row["Features"] = features[start:end]
        new_row["TimeInterval"] = intervals[start:end]
        new_row["RawMessages"] = raw_messages[start:end]
        new_row["ReviewTemplates"] = review_templates[start:end]

        new_row["SeqLen"] = len(new_row["Features"])
        new_row["Latency"] = float(sum(new_row["TimeInterval"]))

        new_row["ChunkID"] = chunk_id
        new_row["OriginalSeqLen"] = n
        new_row["ChunkStartPos"] = start
        new_row["ChunkEndPos"] = end
        new_row["WasChunked"] = True

        chunks.append(new_row)

    return chunks


def split_long_sequences(df: pd.DataFrame, max_len: int = 64, stride: int = 32) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        rows.extend(split_long_sequence_row(row, max_len=max_len, stride=stride))

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).reset_index(drop=True)
