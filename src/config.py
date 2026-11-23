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

# --- Alarm ----------------------------------------------------------------
# Three user-supplied WAV files. Any length; they will loop.
# Place them in artifacts/ or override via CLI.
ALARM_MILD_WAV      = ARTIFACTS_DIR / "alarm_mild.wav"
ALARM_INTENSE_WAV   = ARTIFACTS_DIR / "alarm_intense.wav"
ALARM_FACE_LOST_WAV = ARTIFACTS_DIR / "alarm_face_lost.wav"

# Timing semantics (all wall-clock, seconds)
ALARM_DROWSY_ENTER_SEC   = 3.0     # continuous drowsy before MILD latches
ALARM_INTENSE_ONSET_SEC  = 12.0    # continuous drowsy before INTENSE latches
ALARM_RELEASE_SEC        = 3.0     # continuous non-drowsy clears latches
FACE_LOST_DELAY_SEC      = 2.0     # face-loss duration before pitch is consulted
FACE_LOST_RELEASE_SEC    = 5.0     # continuous non-drowsy after face return

# Head-pitch fallback (rule-based; not a model input)
HEAD_PITCH_SEVERE_DEG      = 35.0  # threshold on max(|baseline - pitch|)
PITCH_WINDOW_SAMPLES       = 10    # ~2 s at 5 Hz
PITCH_BASELINE_MIN_SAMPLES = 10    # samples required to lock the baseline