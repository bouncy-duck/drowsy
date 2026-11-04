from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from src.config import (
    DATASET_DIR, ARTIFACTS_DIR, FEATURE_COLS, LABEL_MAP, IDX_TO_NAME,
    NUM_CLASSES, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LR,
    DEFAULT_VAL_FRAC, DEFAULT_SEED, TRAIN_STRIDE,
)
from src.dataset import load_all, make_windows, grouped_split
from src.model import DrowsyCNN


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=DATASET_DIR)
    p.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int,   default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr",         type=float, default=DEFAULT_LR)
    p.add_argument("--val-frac",   type=float, default=DEFAULT_VAL_FRAC)
    p.add_argument("--seed",       type=int,   default=DEFAULT_SEED)
    p.add_argument("--stride",     type=int,   default=TRAIN_STRIDE,
                   help="window stride in samples during training")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device = {device}")

    # ---------------- data ----------------
    df = load_all(args.data_root)
    print("[train] label distribution (per video):")
    print(df.groupby("recording_label_name").size())

    X, y, groups = make_windows(df, stride=args.stride)
    print(f"[train] windows = {X.shape}, "
          f"label counts = {np.bincount(y, minlength=NUM_CLASSES).tolist()}, "
          f"unique videos = {len(np.unique(groups))}")

    tr_idx, va_idx = grouped_split(X, y, groups,
                                   val_frac=args.val_frac, seed=args.seed)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]
    print(f"[train] train windows = {len(X_tr)}, val windows = {len(X_va)}")

    # ---------------- normalization (train-only) ----------------
    mean = X_tr.reshape(-1, X_tr.shape[-1]).mean(axis=0)
    std  = X_tr.reshape(-1, X_tr.shape[-1]).std(axis=0) + 1e-6
    np.savez(ARTIFACTS_DIR / "norm_stats.npz", mean=mean, std=std,
             feature_cols=np.array(FEATURE_COLS, dtype=object))
    print(f"[train] mean={mean}, std={std}")

    X_tr = (X_tr - mean) / std
    X_va = (X_va - mean) / std

    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
        batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ---------------- model ----------------
    model = DrowsyCNN(in_features=len(FEATURE_COLS),
                      num_classes=NUM_CLASSES).to(device)

    counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float32)
    weights = torch.tensor((counts.sum() / (counts + 1e-6)) / NUM_CLASSES,
                           dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc, best_path = 0.0, ARTIFACTS_DIR / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train(); tl = tn = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); opt.step()
            tl += loss.item() * xb.size(0); tn += xb.size(0)
        sched.step()

        # ---- val ----
        model.eval(); correct = total = 0; vl = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                vl += criterion(out, yb).item() * xb.size(0)
                correct += (out.argmax(1) == yb).sum().item()
                total   += xb.size(0)
        val_acc = correct / total

        print(f"[{epoch:02d}/{args.epochs}] "
              f"train_loss={tl/tn:.4f}  val_loss={vl/total:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "features":    FEATURE_COLS,
                "window_size": X.shape[1],
                "num_classes": NUM_CLASSES,
                "val_acc":     val_acc,
            }, best_path)

    print(f"\n[train] best val acc = {best_acc:.4f}  →  {best_path}")

    # ---- final validation report ----
    model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
            gts.append(yb.numpy())
    preds = np.concatenate(preds); gts = np.concatenate(gts)
    print("\n[val] classification report")
    print(classification_report(gts, preds,
                                 labels=list(range(NUM_CLASSES)),
                                 target_names=[IDX_TO_NAME[i] for i in range(NUM_CLASSES)],
                                 zero_division=0))
    print("[val] confusion matrix")
    print(confusion_matrix(gts, preds, labels=list(range(NUM_CLASSES))))


if __name__ == "__main__":
    main()