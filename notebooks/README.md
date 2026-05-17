# Project notebooks

This folder contains experimental notebooks used during development of the Windows Event Log anomaly detection system.

- `Windows_dataset.ipynb` — parsing, preprocessing, template extraction, and sequence construction.
- `Windows_logs_training.ipynb` — labelled Windows log sequences and training data preparation.
- `Transformer_tuning.ipynb` — Transformer hyperparameter tuning and aggregation comparison.
- `Models.ipynb` — baseline models and experimental comparisons.
- `ServerB.ipynb` — transferability testing on Server B and threshold calibration.
- `ServerC.ipynb` — additional transferability/testing notebook.

The production service code is located in `app.py` and the `src/` directory.
