# Drowsiness Detection — Windowed CNN

Real-time drowsiness detection from EAR / MAR features.

## Spec

| Item | Value |
|---|---|
| Window | 10 s causal (50 samples @ ~5 Hz) |
| Training stride | 1 s |
| Inference stride | 1 s |
| Output | 1 label per second |
| Features | `ear_mean`, `mar` |
| Classes | alert / neutral_low_vigilance / drowsy |
| Model | 1-D CNN, receptive field ≈ 6 s |

## Install

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
.\.venv\Scripts\Activate.ps1
```

## Train

Point the loader at the whole dataset directory. Any folders you want
excluded (e.g. Fold5_part1, Fold5_part2) simply must not be present under
`dataset/` — the loader discovers everything recursively.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train.ps1
# or tune
powershell -ExecutionPolicy Bypass -File scripts\train.ps1 -Epochs 60 -BatchSize 64
```

Outputs:
- `artifacts/best_model.pt`
- `artifacts/norm_stats.npz`

## Inference

Batch mode — one `*_predictions.csv` per input CSV, one row per second:

```powershell
# single video
powershell -ExecutionPolicy Bypass -File scripts\infer.ps1 -Input dataset\Fold1_part1\Fold1_part1\01\0_features.csv

# whole folder, write all predictions to a directory
powershell -ExecutionPolicy Bypass -File scripts\infer.ps1 -Input dataset -OutDir predictions
```

Each output row:

| sample_index | timestamp_s | label | p_alert | p_neutral_low_vigilance | p_drowsy |
|---|---|---|---|---|---|

## Real-time use

The production-ready interface is `src.stream.DrowsinessDetector`:

```python
from src.stream import DrowsinessDetector

det = DrowsinessDetector("artifacts/best_model.pt",
                         "artifacts/norm_stats.npz",
                         device="cpu")

# feed one feature sample per call
out = det.push(ear_mean=0.28, mar=0.03)
if out is not None:               # a fresh label became available
    print(out["label_name"], out["probs"])
```

`push()` returns `None` until the 10 s buffer is full, and then returns a
dict once per second (every `INFER_STRIDE` samples). This matches exactly
the labels produced by batch inference.

## Notes

- **Grouping**: windows never cross videos; validation is grouped at the
  `(candidate_id, video_id)` level.
- **Normalization**: mean/std computed on training windows only; stored
  alongside the model, applied automatically at inference.
- **Missing frames**: rows with `face_detected=0` are forward/backward
  filled per video so the timeline stays uniform.
- **Leakage caution**: because candidate IDs are duplicated across folds
  in the extracted dataset, a stricter split (grouping by person) may be
  worth doing if you plan to report validation numbers as estimates of
  cross-subject performance.