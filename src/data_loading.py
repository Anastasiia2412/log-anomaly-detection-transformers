import csv
from pathlib import Path
import pandas as pd

WINDOWS_COLUMNS = [
    "Level",
    "Date and Time",
    "Source",
    "Event ID",
    "Task Category",
    "Message",
]


def load_windows_event_csv(file_obj, log_name: str, source_name: str = "") -> pd.DataFrame:
    rows = []

    if hasattr(file_obj, "read"):
        content = file_obj.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8-sig", errors="replace")
        lines = content.splitlines()
        reader = csv.reader(lines)
    else:
        with open(file_obj, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = list(csv.reader(f))

    header = next(reader, None)

    for row in reader:
        if not row or all(not str(x).strip() for x in row):
            continue

        if len(row) == 6:
            rows.append(row)
        elif len(row) > 6:
            rows.append(row[:5] + [",".join(row[5:])])
        else:
            rows.append(row + [None] * (6 - len(row)))

    df = pd.DataFrame(rows, columns=WINDOWS_COLUMNS)
    df["LogName"] = log_name
    df["SourceFile"] = source_name or getattr(file_obj, "name", "")

    return df


def clean_windows_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["Level", "Date and Time", "Source", "Event ID", "Task Category"]:
        df[col] = df[col].astype(str).str.strip()

    df["Message"] = df["Message"].fillna("").astype(str).str.strip()

    df["timestamp"] = pd.to_datetime(
        df["Date and Time"],
        errors="coerce",
        dayfirst=True,
    )

    df = df.dropna(subset=["timestamp"]).copy()
    return df


def combine_logs(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()

    all_logs = pd.concat(parts, ignore_index=True)

    dedup_cols = [
        "LogName",
        "timestamp",
        "Level",
        "Source",
        "Event ID",
        "Task Category",
        "Message",
    ]

    all_logs = all_logs.drop_duplicates(subset=dedup_cols, keep="first").copy()
    all_logs = all_logs.sort_values("timestamp").reset_index(drop=True)

    return all_logs
