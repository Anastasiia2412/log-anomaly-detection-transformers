def extract_key_events(row, max_items=8):
    items = []

    event_ids = row.get("EventIDs", [])
    sources = row.get("Sources", [])
    levels = row.get("Levels", [])

    for event_id in event_ids:
        items.append(f"EventID {event_id}")

    for source in sources:
        items.append(str(source))

    for level in levels:
        if str(level).lower() in ["error", "critical", "warning"]:
            items.append(str(level))

    unique = []

    for item in items:
        if item not in unique:
            unique.append(item)

    return unique[:max_items]


import numpy as np


def forecast_next_window_risk(scored_df, threshold, lookback=6):
    """
    Forecast anomaly risk for the next time window based on recent anomaly scores.
    This is a lightweight forecasting layer over Transformer anomaly scores.
    """
    if scored_df.empty:
        return {
            "forecast_risk": "unknown",
            "forecast_probability": 0.0,
            "forecast_score": 0.0,
            "reason": "Нет данных для прогноза.",
        }

    df = scored_df.sort_values("WindowStart").copy()
    recent = df.tail(lookback)

    scores = recent["anomaly_score"].astype(float).values

    last_score = float(scores[-1])
    mean_score = float(np.mean(scores))
    max_score = float(np.max(scores))

    anomaly_ratio = float((scores >= threshold).mean())

    if len(scores) >= 2:
        x = np.arange(len(scores))
        trend = float(np.polyfit(x, scores, 1)[0])
    else:
        trend = 0.0

    forecast_score = (
        0.45 * last_score
        + 0.35 * mean_score
        + 0.20 * max_score
        + max(trend, 0.0)
    )

    raw_probability = forecast_score / (threshold * 1.5)
    forecast_probability = float(np.clip(raw_probability, 0.0, 1.0))

    if forecast_score >= threshold * 1.25 or anomaly_ratio >= 0.5:
        forecast_risk = "high"
    elif forecast_score >= threshold * 0.85 or anomaly_ratio >= 0.25:
        forecast_risk = "medium"
    else:
        forecast_risk = "low"

    reason_parts = [
        f"Последний anomaly score: {last_score:.3f}.",
        f"Средний score за последние {len(recent)} окон: {mean_score:.3f}.",
        f"Максимальный score за последние {len(recent)} окон: {max_score:.3f}.",
        f"Доля окон выше threshold: {anomaly_ratio:.2f}.",
    ]

    if trend > 0:
        reason_parts.append(f"Наблюдается рост anomaly score: trend={trend:.3f}.")
    else:
        reason_parts.append(f"Рост anomaly score не выражен: trend={trend:.3f}.")

    return {
        "forecast_risk": forecast_risk,
        "forecast_probability": forecast_probability,
        "forecast_score": float(forecast_score),
        "last_score": last_score,
        "mean_recent_score": mean_score,
        "max_recent_score": max_score,
        "recent_anomaly_ratio": anomaly_ratio,
        "trend": trend,
        "reason": " ".join(reason_parts),
    }
