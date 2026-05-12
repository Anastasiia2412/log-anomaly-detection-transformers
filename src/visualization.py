import plotly.express as px


def make_score_timeline(df, threshold):
    plot_df = df.copy()
    plot_df["WindowStart"] = plot_df["WindowStart"].astype(str)

    fig = px.line(
        plot_df,
        x="WindowStart",
        y="anomaly_score",
        color="prediction",
        title="Transformer anomaly score timeline",
        markers=True,
    )

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        annotation_text=f"threshold={threshold:.3f}",
    )

    fig.update_layout(
        xaxis_title="Window start",
        yaxis_title="Anomaly score",
        legend_title="Prediction",
    )

    return fig
