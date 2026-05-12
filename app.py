from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from config import (
    MODEL_PATH,
    VOCAB_PATH,
    TEMPLATE_TO_ID_PATH,
    WINDOW_SIZE,
    MIN_SEQ_LEN,
    MAX_LEN,
    STRIDE,
    THRESHOLD,
    SCORE_METHOD,
    PAD_IDX,
)

from src.data_loading import (
    load_windows_event_csv,
    clean_windows_events,
    combine_logs,
)
from src.preprocessing import add_event_tokens
from src.sequence_builder import (
    build_sequences_by_time_window,
    split_long_sequences,
)
from src.encoding import add_encoded_features
from src.scoring import extract_key_events
from src.transformer_inference import add_transformer_predictions
from src.checkpoint import (
    load_token_to_id,
    load_template_to_id,
    load_model_checkpoint,
)
from src.visualization import make_score_timeline


st.set_page_config(
    page_title="Windows Event Log Anomaly Detector",
    page_icon="🪟",
    layout="wide",
)


st.title("Windows Event Log Anomaly Detector")

st.write(
    "Прототип сервиса для анализа Windows Event Logs. "
    "Система использует обученную Transformer-модель next-event prediction "
    "и вычисляет anomaly score временных окон на основе ошибки прогноза следующего события."
)


@st.cache_resource
def load_artifacts():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    token_to_id = load_token_to_id(VOCAB_PATH)
    template_to_id = load_template_to_id(TEMPLATE_TO_ID_PATH)

    model, model_config, checkpoint = load_model_checkpoint(
        MODEL_PATH,
        device=device,
    )

    return model, token_to_id, template_to_id, model_config, checkpoint, device


st.header("1. Model status")

if (
    not Path(MODEL_PATH).exists()
    or not Path(VOCAB_PATH).exists()
    or not Path(TEMPLATE_TO_ID_PATH).exists()
):
    st.error("Не найдены локальные артефакты модели, словаря или template mapping.")
    st.write("Положите файлы в папку artifacts/:")
    st.code(
        f"{MODEL_PATH}\n{VOCAB_PATH}\n{TEMPLATE_TO_ID_PATH}",
        language="text",
    )
    st.stop()

try:
    model, token_to_id, template_to_id, model_config, checkpoint, device = load_artifacts()
except Exception as e:
    st.error("Ошибка загрузки модели, словаря или template mapping.")
    st.exception(e)
    st.stop()


c1, c2, c3, c4 = st.columns(4)
c1.metric("Model", "loaded")
c2.metric("Vocabulary size", len(token_to_id))
c3.metric("Device", device)
c4.metric("Threshold", f"{THRESHOLD:.4f}")

with st.expander("Model configuration"):
    st.json({
        "MODEL_PATH": MODEL_PATH,
        "VOCAB_PATH": VOCAB_PATH,
        "TEMPLATE_TO_ID_PATH": TEMPLATE_TO_ID_PATH,
        "WINDOW_SIZE": WINDOW_SIZE,
        "MIN_SEQ_LEN": MIN_SEQ_LEN,
        "MAX_LEN": MAX_LEN,
        "STRIDE": STRIDE,
        "SCORE_METHOD": SCORE_METHOD,
        "THRESHOLD": THRESHOLD,
        "model_config": model_config,
    })


st.header("2. Upload Windows Event Logs")

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


run_button = st.button("Run analysis", type="primary")


if run_button:
    parts = []

    if system_files:
        for file in system_files:
            raw = load_windows_event_csv(
                file,
                log_name="System",
                source_name=file.name,
            )
            clean = clean_windows_events(raw)
            parts.append(clean)

    if application_files:
        for file in application_files:
            raw = load_windows_event_csv(
                file,
                log_name="Application",
                source_name=file.name,
            )
            clean = clean_windows_events(raw)
            parts.append(clean)

    if not parts:
        st.error("Загрузите хотя бы один CSV-файл.")
        st.stop()

    with st.spinner("Loading and preprocessing logs..."):
        windows_clean = combine_logs(parts)
        windows_clean = add_event_tokens(windows_clean)

        # ВАЖНО:
        # Модель обучалась на EventTemplateID вида W..., которые соответствуют
        # полному шаблону события EventTemplateForReview, а не короткому EventToken.
        windows_clean["EventTemplateID"] = windows_clean["EventTemplateForReview"].map(template_to_id)

        # fallback: если вдруг template_to_id был построен по короткому EventToken
        missing_mask = windows_clean["EventTemplateID"].isna()
        if missing_mask.any():
            windows_clean.loc[missing_mask, "EventTemplateID"] = (
                windows_clean.loc[missing_mask, "EventToken"].map(template_to_id)
            )

        windows_clean["EventTemplateID"] = windows_clean["EventTemplateID"].fillna("<UNKNOWN_TEMPLATE>")

        matched_templates = int((windows_clean["EventTemplateID"] != "<UNKNOWN_TEMPLATE>").sum())
        total_templates = int(len(windows_clean))

    with st.spinner("Building event sequences..."):
        sequences = build_sequences_by_time_window(
            windows_clean,
            window=WINDOW_SIZE,
            min_seq_len=MIN_SEQ_LEN,
        )

        if sequences.empty:
            st.error("Не удалось сформировать последовательности.")
            st.stop()

        sequences = split_long_sequences(
            sequences,
            max_len=MAX_LEN,
            stride=STRIDE,
        )

    with st.spinner("Encoding events and running Transformer inference..."):
        sequences = add_encoded_features(
            sequences,
            token_to_id,
        )

        scored = add_transformer_predictions(
            df=sequences,
            model=model,
            threshold=THRESHOLD,
            pad_idx=PAD_IDX,
            max_len=MAX_LEN,
            device=device,
            method=SCORE_METHOD,
            topk=3,
            batch_size=512,
        )

        scored["key_events"] = scored.apply(
            extract_key_events,
            axis=1,
        )

    st.session_state["windows_clean"] = windows_clean
    st.session_state["scored"] = scored


if "scored" not in st.session_state:
    st.warning("Загрузите логи и нажмите Run analysis.")
    st.stop()


windows_clean = st.session_state["windows_clean"]
scored = st.session_state["scored"]

st.header("3. Dataset summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Events", len(windows_clean))
c2.metric("Windows/chunks", len(scored))
c3.metric("Predicted anomalies", int(scored["prediction"].sum()))
c4.metric("Mean anomaly score", f"{scored['anomaly_score'].mean():.3f}")

if "EventTemplateID" in windows_clean.columns:
    matched_events = int((windows_clean["EventTemplateID"] != "<UNKNOWN_TEMPLATE>").sum())
    total_events = int(len(windows_clean))
    st.write(
        f"Template mapping coverage: {matched_events}/{total_events} "
        f"({matched_events / total_events:.3f})"
    )

st.write("Date range:")
st.code(
    f"{windows_clean['timestamp'].min()} — {windows_clean['timestamp'].max()}",
    language="text",
)

st.subheader("Event level distribution")
level_counts = windows_clean["Level"].value_counts().reset_index()
level_counts.columns = ["Level", "count"]
st.dataframe(level_counts, use_container_width=True)

st.subheader("Top event sources")
source_counts = windows_clean["Source"].value_counts().head(20).reset_index()
source_counts.columns = ["Source", "count"]
st.dataframe(source_counts, use_container_width=True)


st.header("4. Vocabulary diagnostics")

c1, c2, c3 = st.columns(3)
c1.metric("Mean vocabulary coverage", f"{scored['VocabularyCoverage'].mean():.3f}")
c2.metric("Mean UNK ratio", f"{scored['UnkRatio'].mean():.3f}")
c3.metric("Windows with UNK > 0.3", int((scored["UnkRatio"] > 0.3).sum()))

if scored["UnkRatio"].mean() > 0.2:
    st.warning(
        "Доля неизвестных событий высокая. Для нового сервера может потребоваться "
        "калибровка порога или расширение словаря."
    )


st.header("5. Transformer anomaly forecast")

fig = make_score_timeline(
    scored,
    threshold=THRESHOLD,
)

st.plotly_chart(fig, use_container_width=True)

sorted_scored = scored.sort_values(
    "anomaly_score",
    ascending=False,
).reset_index(drop=True)

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


st.header("6. Window details with LocalNLL")

selected_idx = st.number_input(
    "Select row index from suspicious windows table",
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
    "risk_level": str(selected["risk_level"]),
    "UnkRatio": float(selected["UnkRatio"]),
    "key_events": selected["key_events"],
})

details = pd.DataFrame({
    "TimeInterval": selected["TimeInterval"],
    "Feature": selected["Features"],
    "OriginalEventToken": selected.get("EventTokens", [""] * len(selected["Features"])),
    "EncodedToken": selected["EncodedFeatures"],
    "Template": selected["ReviewTemplates"],
    "Message": selected["RawMessages"],
})

local_scores = selected.get("local_nll_scores", [])
local_nll_column = [None] * len(details)

if isinstance(local_scores, list):
    for i, score in enumerate(local_scores):
        pos = i + 1
        if pos < len(local_nll_column):
            local_nll_column[pos] = score

details["LocalNLL"] = local_nll_column

st.dataframe(
    details,
    use_container_width=True,
    height=450,
)


st.header("7. Export report")

csv = sorted_scored.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="Download CSV report",
    data=csv,
    file_name="windows_anomaly_report.csv",
    mime="text/csv",
)
