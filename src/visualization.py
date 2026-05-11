import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def make_score_timeline(df: pd.DataFrame, threshold: float | None = None):
    fig = px.line(
        df,
        x="WindowStart",
        y="anomaly_score",
        color="risk_level" if "risk_level" in df.columns else None,
        markers=True,
        title="Anomaly score over time",
    )

    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            annotation_text=f"threshold={threshold:.3f}",
        )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Anomaly score",
        height=450,
    )

    return fig


def make_level_distribution(df: pd.DataFrame):
    rows = []

    for levels in df["Levels"]:
        for level in levels:
            rows.append({"Level": level})

    level_df = pd.DataFrame(rows)

    if level_df.empty:
        return None

    fig = px.bar(
        level_df["Level"].value_counts().reset_index(),
        x="Level",
        y="count",
        title="Event level distribution in windows",
    )

    return fig
