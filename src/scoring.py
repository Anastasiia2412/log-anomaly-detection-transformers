import numpy as np
import pandas as pd


HARD_CRITICAL_EVENT_IDS = {
    "41", "6008", "7031", "7034",
    "17832", "17883", "1102", "4625",
    "4740", "9002", "4014", "18456"
}

SOFT_SUSPICIOUS_EVENT_IDS = {
    "1000", "1001", "10010"
}


def extract_key_events(row) -> list[str]:
    key_events = []

    levels = [str(x).lower() for x in row.get("Levels", [])]
    event_ids = [str(x) for x in row.get("EventIDs", [])]
    sources = [str(x) for x in row.get("Sources", [])]
    templates = row.get("ReviewTemplates", [])

    if any("critical" in x for x in levels):
        key_events.append("Critical level event")

    if any("error" in x for x in levels):
        key_events.append("Error level event")

    for event_id in event_ids:
        if event_id in HARD_CRITICAL_EVENT_IDS:
            key_events.append(f"Hard critical Event ID {event_id}")
        elif event_id in SOFT_SUSPICIOUS_EVENT_IDS:
            key_events.append(f"Suspicious Event ID {event_id}")

    for source in sources:
        source_low = source.lower()
        if "mssqlserver" in source_low:
            key_events.append("MSSQLSERVER events")
        if "windows error reporting" in source_low:
            key_events.append("Windows Error Reporting")
        if "sqlserveragent" in source_low:
            key_events.append("SQLSERVERAGENT events")

    for template in templates[:100]:
        t = str(template).lower()
        if "sqlservr.exe" in t or "sqlexception" in t:
            key_events.append("SQL Server exception / sqlservr.exe")
        if "unexpected shutdown" in t or "unplanned" in t:
            key_events.append("Unexpected shutdown/restart")
        if "terminated unexpectedly" in t:
            key_events.append("Service terminated unexpectedly")

    seen = set()
    unique = []
    for item in key_events:
        if item not in seen:
            unique.append(item)
            seen.add(item)

    return unique[:10]


def rule_based_anomaly_score(row) -> float:
    score = 0.0

    levels = [str(x).lower() for x in row.get("Levels", [])]
    event_ids = set(str(x) for x in row.get("EventIDs", []))
    sources = [str(x).lower() for x in row.get("Sources", [])]
    templates = [str(x).lower() for x in row.get("ReviewTemplates", [])]

    if any("critical" in x for x in levels):
        score += 0.45

    if any("error" in x for x in levels):
        score += 0.25

    hard_matches = event_ids.intersection(HARD_CRITICAL_EVENT_IDS)
    score += 0.25 * min(len(hard_matches), 3)

    soft_matches = event_ids.intersection(SOFT_SUSPICIOUS_EVENT_IDS)
    score += 0.12 * min(len(soft_matches), 3)

    if any("mssqlserver" in s for s in sources):
        if any("error" in x for x in levels):
            score += 0.35
        else:
            score += 0.08

    if any("windows error reporting" in s for s in sources):
        score += 0.25

    sql_exception_count = sum(
        ("sqlservr.exe" in t or "sqlexception" in t or "sqldump" in t)
        for t in templates
    )
    if sql_exception_count >= 1:
        score += 0.25
    if sql_exception_count >= 2:
        score += 0.25
    if sql_exception_count >= 5:
        score += 0.15

    if row.get("SeqLen", 0) >= 64:
        score += 0.10
    if row.get("SeqLen", 0) >= 100:
        score += 0.15

    unk_ratio = float(row.get("UnkRatio", 0.0))
    if unk_ratio > 0.3:
        score += 0.05

    return float(min(score, 1.0))


def score_dataframe(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    df = df.copy()

    df["anomaly_score"] = df.apply(rule_based_anomaly_score, axis=1)
    df["prediction"] = (df["anomaly_score"] >= threshold).astype(int)
    df["key_events"] = df.apply(extract_key_events, axis=1)

    def risk(score):
        if score >= 0.8:
            return "high"
        if score >= threshold:
            return "medium"
        return "low"

    df["risk_level"] = df["anomaly_score"].apply(risk)

    return df
