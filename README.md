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
└── README.md '''

#Service purpose

The service is designed for analysing exported Windows Event Logs and helping a DBA or system engineer identify suspicious time intervals faster.


The user can upload one or more files for:
System logs;
Application logs.

Installation
1. Clone the repository:
git clone https://github.com/Anastasiia2412/log-anomaly-detection-transformers.git
cd log-anomaly-detection-transformers

2. Create and activate a virtual environment:
python3 -m venv .venv
source .venv/bin/activate

3.Install dependencies:
pip install -r src/requirements.txt

If Streamlit is not installed by the requirements file, install it manually:
pip install streamlit

4.Running the service
Start the application:
python3 -m streamlit run app.py

The browser will open the service page. If it does not open automatically, copy the local URL from the terminal, usually:
http://localhost:8501

How to use:
-Open the Streamlit application.
-Check that the model status is loaded.
-Upload one or more System CSV files.
-Upload one or more Application CSV files.
-Click Run analysis.
-Review the dataset summary and vocabulary diagnostics.
-Open the anomaly score timeline.
-Inspect suspicious windows and LocalNLL details.
-Check the next-window anomaly risk forecast.
-Download the CSV report if needed.


Notebooks

Windows_dataset.ipynb — Windows Event Log parsing and sequence construction;
Windows_logs_training.ipynb — labelled sequence preparation and training data export;
Transformer_tuning.ipynb — Transformer tuning and aggregation comparison;
Models.ipynb — baseline models and comparison experiments;
ServerB.ipynb — transferability testing and threshold calibration;
ServerC.ipynb — additional server testing.
