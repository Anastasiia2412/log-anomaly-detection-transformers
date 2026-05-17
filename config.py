MODEL_PATH = "src/artifacts/transformer_windows_30min_v3.pt"
VOCAB_PATH = "src/artifacts/token_to_id_windows_30min_v3.json"
TEMPLATE_TO_ID_PATH = "src/artifacts/template_to_id_windows_30min_v3.json"

WINDOW_SIZE = "30min"
MIN_SEQ_LEN = 5

MAX_LEN = 64
STRIDE = 32

THRESHOLD = 13.034093
SCORE_METHOD = "max"

PAD_IDX = 0
UNK_OFFSET = 1
NUM_UNK_BUCKETS = 64
