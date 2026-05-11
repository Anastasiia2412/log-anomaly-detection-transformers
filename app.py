import streamlit as st
import pandas as pd
import torch

from config import WINDOW_SIZE, MIN_SEQ_LEN, MAX_LEN, STRIDE
from src.data_loading import load_windows_event_csv, clean_windows_events, combine_logs
from src.preprocessing import add_event_tokens
from src.sequence_builder import build_sequences_by_time_window, split_long_sequences
from src.encoding import build_vocab_from_events, add_encoded_features
from src.scoring import score_dataframe, extract_key_events
from src.transformer_inference import add_transformer_predictions
from src.checkpoint import (
    load_token_to_id_from_uploaded_file,
    load_model_from_uploaded_checkpoint,
)
from src.visualization import make_score_timeline


st.set_page_config(
    page_title="Windows Event Log Anomaly Detector",
    page_icon="🪟",
    layout="wide",
)


st.title("Windows Event Log Anomaly Detector")
st.write(
    "Прототип сервиса для анализа Windows Event Logs и выявления аномальных временных окон "
    "с использованием Transformer-модели next-event prediction."
)

with st.sidebar:
    st.header("Settings")

    scoring_mode = st.radio(
        "Scoring mode",
        ["Transformer", "Rule-based fallback"],
        index=0,
    )

    window_size = st.selectbox("Window size", ["15min", "30min", "60min"], index=1)
    min_seq_len = st.number_input("Minimum sequence length", min_value=2, max_value=50, value=MIN_SEQ_LEN)
    max_len = st.number_input("Max chunk length", min_value=16, max_value=256, value=MAX_LEN)
    stride = st.number_input("Chunk stride", min_value=8, max_value=128, value=STRIDE)

    threshold = st.number_input(
        "Threshold",
        min_value=0.0,
        value=13.034093,
        step=0.1,
    )

    score_method = st.selectbox(
        "Aggregation method",
        ["max", "topk_mean", "p95", "mean"],
        index=0,
    )


st.header("1. Upload model artifacts")

model_file = None
vocab_file = None

if scoring_mode == "Transformer":
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        model_file = st.file_uploader(
            "Upload Transformer checkpoint (.pt/.pth)",
            type=["pt", "pth"],
        )

    with col_m2:
        vocab_file = st.file_uploader(
            "Upload token_to_id vocabulary (.json/.pkl)",
            type=["json", "pkl", "pickle"],
        )

    st.caption(
        "Модель и словарь не хранятся в GitHub. Для демонстрации загрузите локальные артефакты."
    )


st.header("2. Upload logs")

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
    if scoring_mode == "Transformer" and (model_file is None or vocab_file is None):
        st.error("Для Transformer-режима нужно загрузить checkpoint модели и vocabulary.")
        st.stop()

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

    if scoring_mode == "Transformer":
        device = "cuda" if torch.cuda.is_available() else "cpu"

        token_to_id = load_token_to_id_from_uploaded_file(vocab_file)

        sequences = add_encoded_features(sequences, token_to_id)

        model, model_config = load_model_from_uploaded_checkpoint(
            uploaded_file=model_file,
            vocab_size=len(token_to_id),
            device=device,
            default_config={
                "pad_idx": token_to_id.get("<PAD>", 0),
                "d_model": 128,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 256,
                "dropout": 0.1,
                "max_len": int(max_len),
            },
        )

        scored = add_transformer_predictions(
            df=sequences,
            model=model,
            threshold=float(threshold),
            pad_idx=token_to_id.get("<PAD>", 0),
            max_len=int(max_len),
            device=device,
            method=score_method,
            topk=3,
            batch_size=512,
            unk_ids={token_to_id.get("<UNK>", 1)},
        )

        scored["key_events"] = scored.apply(extract_key_events, axis=1)

        st.session_state["model_config"] = model_config
        st.session_state["device"] = device

    else:
        token_to_id, _ = build_vocab_from_events(windows_clean["EventToken"].tolist())
        sequences = add_encoded_features(sequences, token_to_id)
        scored = score_dataframe(sequences, threshold=0.5)

    st.session_state["windows_clean"] = windows_clean
    st.session_state["scored"] = scored
    st.session_state["token_to_id"] = token_to_id
    st.session_state["scoring_mode"] = scoring_mode
    st.session_state["threshold"] = float(threshold)


if "scored" in st.session_state:
    windows_clean = st.session_state["windows_clean"]
    scored = st.session_state["scored"]
    token_to_id = st.session_state["token_to_id"]
    threshold = st.session_state["threshold"]
    scoring_mode = st.session_state["scoring_mode"]

    st.header("3. Dataset summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Events", len(windows_clean))
    c2.metric("Windows / chunks", len(scored))
    c3.metric("Vocabulary size", len(token_to_id))
    c4.metric("Mean UNK ratio", f"{scored['UnkRatio'].mean():.3f}")

    st.write("Date range:")
    st.code(f"{windows_clean['timestamp'].min()} — {windows_clean['timestamp'].max()}")

    st.write("Scoring mode:")
    st.code(scoring_mode)

    if scoring_mode == "Transformer":
        st.write("Device:")
        st.code(st.session_state.get("device", "cpu"))

        st.write("Model config:")
        st.json(st.session_state.get("model_config", {}))

    st.subheader("Event level distribution")
    level_counts = windows_clean["Level"].value_counts().reset_index()
    level_counts.columns = ["Level", "count"]
    st.dataframe(level_counts, use_container_width=True)

    st.subheader("Top sources")
    source_counts = windows_clean["Source"].value_counts().head(20).reset_index()
    source_counts.columns = ["Source", "count"]
    st.dataframe(source_counts, use_container_width=True)

    st.header("4. Vocabulary diagnostics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean coverage", f"{scored['VocabularyCoverage'].mean():.3f}")
    c2.metric("Max UNK ratio", f"{scored['UnkRatio'].max():.3f}")
    c3.metric("Windows with UNK > 0.3", int((scored["UnkRatio"] > 0.3).sum()))

    if scored["UnkRatio"].mean() > 0.2:
        st.warning(
            "Доля неизвестных токенов высокая. Для нового сервера рекомендуется калибровка threshold."
        )

    st.header("5. Anomaly detection results")

    fig = make_score_timeline(scored, threshold=threshold)
    st.plotly_chart(fig, use_container_width=True)

    sorted_scored = scored.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

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

    st.subheader("Suspicious windows")
    st.dataframe(
        sorted_scored[result_cols],
        use_container_width=True,
        height=420,
    )

    st.header("6. Window details")

    selected_idx = st.number_input(
        "Select row index from sorted table",
        min_value=0,
        max_value=max(len(sorted_scored) - 1, 0),
        value=0,
    )

    selected = sorted_scored.iloc[int(selected_idx)]

    st.json({
        "WindowStart": str(selected["WindowStart"]),
        "WindowEnd": str(selected["WindowEnd"]),
        "SeqLen": int(selected["SeqLen"]),
        "anomaly_score": float(selected["anomaly_score"]),
        "prediction": int(selected["prediction"]),
        "risk_level": selected["risk_level"],
        "UnkRatio": float(selected["UnkRatio"]),
        "key_events": selected["key_events"],
    })

    details = pd.DataFrame({
        "TimeInterval": selected["TimeInterval"],
        "EventToken": selected["Features"],
        "Template": selected["ReviewTemplates"],
        "Message": selected["RawMessages"],
    })

    if "local_nll_scores" in selected and isinstance(selected["local_nll_scores"], list):
        local_scores = selected["local_nll_scores"]
        details["LocalNLL"] = [None] + local_scores[: max(0, len(details) - 1)]

    st.dataframe(details, use_container_width=True, height=450)

    st.header("7. Export report")

    csv = sorted_scored.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="Download CSV report",
        data=csv,
        file_name="windows_anomaly_report.csv",
        mime="text/csv",
    )

else:
    st.warning("Загрузите модель, словарь, CSV-файлы и нажмите Run analysis.")
