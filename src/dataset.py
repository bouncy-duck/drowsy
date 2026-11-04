from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import (
    DATASET_DIR, FEATURE_COLS, LABEL_MAP,
    WINDOW_SIZE, TRAIN_STRIDE,
)


# ---------------------------------------------------------------- discovery
def discover_csvs(root: Path) -> list[Path]:
    """Recursively find every *_features.csv under `root`."""
    return sorted(Path(root).glob("**/*_features.csv"))


# ------------------------------------------------------------------- loader
def load_all(root: Path = DATASET_DIR) -> pd.DataFrame:
    files = discover_csvs(root)
    if not files:
        raise RuntimeError(f"No *_features.csv found under {root}")
    print(f"[data] found {len(files)} feature CSVs under {root}")

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[data] skip {f}  ({e})")
            continue
        if df.empty:
            continue
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    # sort so forward/back fill preserves timeline
    df = df.sort_values(["candidate_id", "video_id", "sample_index"]).reset_index(drop=True)

    # Fill missing face frames by carry-forward / carry-back per video
    g = df.groupby(["candidate_id", "video_id"], sort=False)
    df[FEATURE_COLS] = g[FEATURE_COLS].transform(lambda s: s.ffill().bfill())

    # drop rows still missing (e.g. video starts with no face) and unknown labels
    df = df.dropna(subset=FEATURE_COLS + ["recording_label_name"]).reset_index(drop=True)
    df = df[df["recording_label_name"].isin(LABEL_MAP)].reset_index(drop=True)
    df["label"] = df["recording_label_name"].map(LABEL_MAP).astype(np.int64)

    print(f"[data] kept {len(df)} rows across "
          f"{df.groupby(['candidate_id','video_id']).ngroups} videos")
    return df


# ---------------------------------------------------------------- windows
def make_windows(df: pd.DataFrame, stride: int = TRAIN_STRIDE):
    """Return X (N, WINDOW_SIZE, F), y (N,), groups (N,) of video keys."""
    X_list, y_list, g_list = [], [], []

    for (cid, vid), g in df.groupby(["candidate_id", "video_id"], sort=False):
        g = g.sort_values("sample_index").reset_index(drop=True)
        feats = g[FEATURE_COLS].to_numpy(np.float32)
        label = int(g["label"].iloc[0])          # homogeneous per video
        key   = f"{cid}::{vid}"
        n = len(feats)
        if n < WINDOW_SIZE:
            continue
        for start in range(0, n - WINDOW_SIZE + 1, stride):
            X_list.append(feats[start:start + WINDOW_SIZE])
            y_list.append(label)
            g_list.append(key)

    if not X_list:
        raise RuntimeError("No windows produced — check WINDOW_SIZE / data")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    groups = np.array(g_list)
    return X, y, groups


# ------------------------------------------------------------------- split
def grouped_split(X, y, groups, val_frac=0.15, seed=42):
    """Val split at video level. Returns (train_idx, val_idx)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    return next(gss.split(X, y, groups))