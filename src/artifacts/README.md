# Local model artifacts

This directory is used by the Streamlit service to load the trained Transformer model and event dictionaries.

Expected local files:

- `transformer_windows_30min_v3.pt`
- `token_to_id_windows_30min_v3.json`
- `template_to_id_windows_30min_v3.json`

The actual artifact files are not tracked in Git because they contain trained model weights and environment-specific event mappings.
