from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.config import (
    ARTIFACTS_DIR, FEATURE_COLS, IDX_TO_NAME, NUM_CLASSES,
    WINDOW_SIZE, INFER_STRIDE,
)
from src.model import DrowsyCNN


# -------------------------------------------------------- offline (fast)
def infer_one_csv_vectorized(csv_path: Path,
                             model: DrowsyCNN,
                             mean: np.ndarray, std: np.ndarray,
                             device: torch.device,
                             window_size: int = WINDOW_SIZE,
                             stride: int = INFER_STRIDE) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # per-video: forward/back-fill missing face rows, keep timeline intact
    df = df.sort_values("sample_index").reset_index(drop=True)
    for c in FEATURE_COLS:
        df[c] = df[c].ffill().bfill()
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    n = len(df)
    if n < window_size:
        return pd.DataFrame()

    feats = df[FEATURE_COLS].to_numpy(np.float32)
    feats = (feats - mean) / std

    starts = np.arange(0, n - window_size + 1, stride)
    idx = starts[:, None] + np.arange(window_size)[None, :]      # (K, T)
    windows = feats[idx]                                          # (K, T, F)

    with torch.no_grad():
        t = torch.from_numpy(windows).to(device)
        probs = torch.softmax(model(t), dim=1).cpu().numpy()      # (K, K_cls)

    # Emit at the END of the causal window
    end_idx = starts + window_size - 1

    out = pd.DataFrame({
        "sample_index": df["sample_index"].iloc[end_idx].values,
        "timestamp_s":  df["timestamp_s"].iloc[end_idx].values,
        "label":        [IDX_TO_NAME[i] for i in probs.argmax(1)],
    })
    for i in range(NUM_CLASSES):
        out[f"p_{IDX_TO_NAME[i]}"] = probs[:, i].astype(np.float32)

    return out


# -------------------------------------------------------- load helpers
def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = DrowsyCNN(in_features=len(FEATURE_COLS),
                      num_classes=ckpt.get("num_classes", NUM_CLASSES))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


# -------------------------------------------------------- entry point
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True,
                   help="A single *_features.csv OR a directory (recurses)")
    p.add_argument("--model", type=Path, default=ARTIFACTS_DIR / "best_model.pt")
    p.add_argument("--norm",  type=Path, default=ARTIFACTS_DIR / "norm_stats.npz")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Where to save *_predictions.csv. Defaults next to source CSVs.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    stats = np.load(args.norm, allow_pickle=True)
    mean, std = stats["mean"].astype(np.float32), stats["std"].astype(np.float32)
    model = load_model(args.model, device)

    if args.input.is_dir():
        csvs = sorted(args.input.glob("**/*_features.csv"))
    else:
        csvs = [args.input]

    if not csvs:
        raise SystemExit("No CSVs found.")

    print(f"[infer] {len(csvs)} CSVs  |  device={device}  |  model={args.model}")

    for cp in csvs:
        result = infer_one_csv_vectorized(cp, model, mean, std, device)
        if result.empty:
            print(f"[infer] {cp.name}: too short, skipped")
            continue

        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.out_dir / (cp.stem + "_predictions.csv")
        else:
            out_path = cp.with_name(cp.stem + "_predictions.csv")

        result.to_csv(out_path, index=False)
        first_t = result["timestamp_s"].iloc[0]
        last_t  = result["timestamp_s"].iloc[-1]
        print(f"[infer] {cp.name:24s} → {len(result):4d} labels "
              f"[{first_t:.1f}s … {last_t:.1f}s] → {out_path.name}")


if __name__ == "__main__":
    main()