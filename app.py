import streamlit as st
import pandas as pd

from config import WINDOW_SIZE, MIN_SEQ_LEN, MAX_LEN, STRIDE
from src.data_loading import load_windows_event_csv, clean_windows_events, combine_logs
from src.preprocessing import add_event_tokens
from src.sequence_builder import build_sequences_by_time_window, split_long_sequences
from src.encoding import build_vocab_from_events, add_encoded_features
from src.scoring import score_dataframe
from src.visualization import make_score_timeline, make_level_distribution


st.set_page_config(
    page_title="Windows Event Log Anomaly Detector",
    page_icon="🪟",
    layout="wide",
)


st.title("Windows Event Log Anomaly Detector")
st.write(
    "Прототип сервиса для анализа Windows Event Logs и поиска подозрительных временных окон."
)

st.info(
    "Демо-версия использует rule-based scoring для интерфейса. "
    "Transformer-модель можно подключить как следующий модуль inference."
)

with st.sidebar:
    st.header("Settings")
    window_size = st.selectbox("Window size", ["15min", "30min", "60min"], index=1)
    min_seq_len = st.number_input("Minimum sequence length", min_value=2, max_value=50, value=MIN_SEQ_LEN)
    max_len = st.number_input("Max chunk length", min_value=16, max_value=256, value=MAX_LEN)
    stride = st.number_input("Chunk stride", min_value=8, max_value=128, value=STRIDE)
    threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)


st.header("1. Upload logs")

col1, col2 = st.columns(2)

with col1:
    system_files = st.file_uploader(
        "Upload System CSV files",
        type=["csv"],
        accept_multiple_files=True,
    )

with col2:
    application_files = st.file_uploader(
        "Upload Application CSV files",
        type=["csv"],
        accept_multiple_files=True,
    )


if st.button("Run analysis", type="primary"):
    parts = []

    if system_files:
        for file in system_files:
            raw = load_windows_event_csv(file, log_name="System", source_name=file.name)
            clean = clean_windows_events(raw)
            parts.append(clean)

    if application_files:
        for file in application_files:
            raw = load_windows_event_csv(file, log_name="Application", source_name=file.name)
            clean = clean_windows_events(raw)
            parts.append(clean)

    if not parts:
        st.error("Загрузите хотя бы один CSV-файл.")
        st.stop()

    windows_clean = combine_logs(parts)

    if windows_clean.empty:
        st.error("После загрузки не осталось валидных строк.")
        st.stop()

    windows_clean = add_event_tokens(windows_clean)

    st.session_state["windows_clean"] = windows_clean

    sequences = build_sequences_by_time_window(
        windows_clean,
        window=window_size,
        min_seq_len=int(min_seq_len),
    )

    if sequences.empty:
        st.error("Не удалось сформировать последовательности. Попробуйте уменьшить min_seq_len.")
        st.stop()

    sequences = split_long_sequences(
        sequences,
        max_len=int(max_len),
        stride=int(stride),
    )

    token_to_id, _ = build_vocab_from_events(windows_clean["EventToken"].tolist())
    sequences = add_encoded_features(sequences, token_to_id)

    scored = score_dataframe(sequences, threshold=threshold)

    st.session_state["scored"] = scored
    st.session_state["token_to_id"] = token_to_id


if "scored" in st.session_state:
    windows_clean = st.session_state["windows_clean"]
    scored = st.session_state["scored"]
    token_to_id = st.session_state["token_to_id"]

    st.header("2. Dataset summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Events", len(windows_clean))
    c2.metric("Windows / chunks", len(scored))
    c3.metric("Vocabulary size", len(token_to_id))
    c4.metric("Mean UNK ratio", f"{scored['UnkRatio'].mean():.3f}")

    st.write("Date range:")
    st.code(f"{windows_clean['timestamp'].min()} — {windows_clean['timestamp'].max()}")

    st.subheader("Level distribution")
    level_counts = windows_clean["Level"].value_counts().reset_index()
    level_counts.columns = ["Level", "count"]
    st.dataframe(level_counts, use_container_width=True)

    st.subheader("Top sources")
    source_counts = windows_clean["Source"].value_counts().head(20).reset_index()
    source_counts.columns = ["Source", "count"]
    st.dataframe(source_counts, use_container_width=True)

    st.header("3. Vocabulary diagnostics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean coverage", f"{scored['VocabularyCoverage'].mean():.3f}")
    c2.metric("Max UNK ratio", f"{scored['UnkRatio'].max():.3f}")
    c3.metric("Windows with UNK > 0.3", int((scored["UnkRatio"] > 0.3).sum()))

    st.header("4. Anomaly scores")

    fig = make_score_timeline(scored, threshold=threshold)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Suspicious windows")

    result_cols = [
        "WindowStart",
        "WindowEnd",
        "SeqLen",
        "anomaly_score",
        "prediction",
        "risk_level",
        "UnkRatio",
        "EventIDs",
        "Sources",
        "key_events",
    ]

    sorted_scored = scored.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    st.dataframe(
        sorted_scored[result_cols],
        use_container_width=True,
        height=400,
    )

    st.header("5. Window details")

    selected_idx = st.number_input(
        "Select row index from sorted table",
        min_value=0,
        max_value=max(len(sorted_scored) - 1, 0),
        value=0,
    )

    selected = sorted_scored.iloc[int(selected_idx)]

    st.write("Selected window:")
    st.json({
        "WindowStart": str(selected["WindowStart"]),
        "WindowEnd": str(selected["WindowEnd"]),
        "SeqLen": int(selected["SeqLen"]),
        "anomaly_score": float(selected["anomaly_score"]),
        "prediction": int(selected["prediction"]),
        "risk_level": selected["risk_level"],
        "key_events": selected["key_events"],
    })

    details = pd.DataFrame({
        "TimeInterval": selected["TimeInterval"],
        "EventToken": selected["Features"],
        "Template": selected["ReviewTemplates"],
        "Message": selected["RawMessages"],
    })

    st.dataframe(details, use_container_width=True, height=450)

    st.header("6. Export report")

    export_df = sorted_scored.copy()
    csv = export_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="Download CSV report",
        data=csv,
        file_name="windows_anomaly_report.csv",
        mime="text/csv",
    )

else:
    st.warning("Загрузите CSV-файлы и нажмите Run analysis.")
