# Windows Event Log Anomaly Detector

This project contains a prototype service for Windows Event Log anomaly analysis.  
The system uses a Transformer model trained with a next-event prediction objective. It processes Windows `System` and `Application` logs, builds event sequences, calculates anomaly scores for time windows, and estimates short-term anomaly risk for the next window.

## Project structure

```text
log-anomaly-detection-transformers/
├── app.py                  # Streamlit service
├── config.py               # Service configuration
├── src/                    # Source code of the processing and inference pipeline
│   ├── artifacts/          # Local model artifacts, not tracked in Git
│   ├── checkpoint.py       # Model and vocabulary loading
│   ├── data_loading.py     # Windows CSV loading and cleaning
│   ├── encoding.py         # Event token encoding
│   ├── model.py            # Transformer architecture
│   ├── preprocessing.py    # Event message normalization
│   ├── scoring.py          # Key event extraction and risk forecast
│   ├── sequence_builder.py # Time-window sequence construction
│   ├── transformer_inference.py # Transformer scoring
│   └── visualization.py    # Timeline visualization
├── notebooks/              # Experimental notebooks
└── README.md
Service purpose

The service is designed for analysing exported Windows Event Logs and helping a DBA or system engineer identify suspicious time intervals faster.

Instead of reviewing thousands of raw log entries manually, the application:

parses System and Application CSV files;
normalizes event messages;
maps events to learned templates;
builds 30-minute event sequences;
calculates Transformer-based anomaly scores;
marks suspicious windows;
shows local event-level evidence;
estimates anomaly risk for the next time window;
exports the result as a CSV report.
Required local artifacts

The service expects trained model artifacts in:

src/artifacts/

Required files:

transformer_windows_30min_v3.pt
token_to_id_windows_30min_v3.json
template_to_id_windows_30min_v3.json

These files are not stored in GitHub. They are kept locally because they contain trained model weights and environment-specific event-template mappings.

Input data format

The service expects CSV files exported from Windows Event Viewer.

Required columns:

Level
Date and Time
Source
Event ID
Task Category
Message

The user can upload one or more files for:

System logs;
Application logs.
Installation

Clone the repository:

git clone https://github.com/Anastasiia2412/log-anomaly-detection-transformers.git
cd log-anomaly-detection-transformers

Create and activate a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r src/requirements.txt

If Streamlit is not installed by the requirements file, install it manually:

pip install streamlit
Running the service

Start the application:

python3 -m streamlit run app.py

The browser will open the service page. If it does not open automatically, copy the local URL from the terminal, usually:

http://localhost:8501
How to use
Open the Streamlit application.
Check that the model status is loaded.
Upload one or more System CSV files.
Upload one or more Application CSV files.
Click Run analysis.
Review the dataset summary and vocabulary diagnostics.
Open the anomaly score timeline.
Inspect suspicious windows and LocalNLL details.
Check the next-window anomaly risk forecast.
Download the CSV report if needed.
Main output fields

The service displays the following results:

anomaly_score — Transformer-based score of unexpected event transitions;
prediction — binary prediction, where 0 means normal and 1 means anomaly;
risk_level — low, medium, or high risk level;
VocabularyCoverage — share of known event tokens in the sequence;
UnkRatio — share of unknown tokens;
key_events — important Event IDs and sources in the selected window;
LocalNLL — local negative log-likelihood for event-level inspection;
Forecasted risk — estimated anomaly risk for the next 30-minute window.
Notes

The service does not replace a DBA or monitoring system. It helps prioritize log inspection by highlighting time windows with unusual event sequences.

High anomaly score means that the event pattern was unexpected for the trained Transformer model. It does not automatically prove the root cause of an incident. The result should be interpreted together with SQL Server logs, Windows system state, service history, and administrator knowledge.

Notebooks

The notebooks/ directory contains the experimental part of the project:

Windows_dataset.ipynb — Windows Event Log parsing and sequence construction;
Windows_logs_training.ipynb — labelled sequence preparation and training data export;
Transformer_tuning.ipynb — Transformer tuning and aggregation comparison;
Models.ipynb — baseline models and comparison experiments;
ServerB.ipynb — transferability testing and threshold calibration;
ServerC.ipynb — additional server testing.
