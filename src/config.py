from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATASET_DIR   = PROJECT_ROOT / "dataset"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Features -------------------------------------------------------------
FEATURE_COLS = ['ear_mean', 'mar']

LABEL_MAP = {
    "alert": 0,
    "neutral_low_vigilance": 1,
    "drowsy": 2,
}
IDX_TO_NAME = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = len(LABEL_MAP)

# --- Temporal spec --------------------------------------------------------
# Source .mov runs at ~29.97 fps; features were sampled every 6th frame.
SOURCE_FPS   = 29.9689
SAMPLE_STEP  = 6
SAMPLE_HZ    = SOURCE_FPS / SAMPLE_STEP   # ~4.995 Hz

WINDOW_SECONDS = 10.0
WINDOW_SIZE    = 50       # samples in a 10 s causal window
TRAIN_STRIDE   = 5        # 1 s = 5 samples
INFER_STRIDE   = 5        # 1 s = 5 samples → 1 output / second

# --- Training defaults ----------------------------------------------------
DEFAULT_EPOCHS     = 40
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR         = 1e-3
DEFAULT_VAL_FRAC   = 0.15
DEFAULT_SEED       = 42