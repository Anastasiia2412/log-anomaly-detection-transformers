# Project notebooks

This folder contains experimental notebooks used during development of the Windows Event Log anomaly detection system.

- Windows_dataset.ipynb — Windows Event Log parsing and sequence construction (Server A);
- Windows_logs_training.ipynb — Transformer training, threshold calibration, test evaluation, and service artifact export;
- Transformer_tuning.ipynb — Transformer tuning and aggregation comparison;
- Models.ipynb — baseline models, LSTM, Transformer and comparison experiments;
- ServerB.ipynb — Windows Event Log parsing and sequence construction (Server B);
- ServerC.ipynb — Windows Event Log parsing and sequence construction (Server C).

The production service code is located in `app.py` and the `src/` directory.
