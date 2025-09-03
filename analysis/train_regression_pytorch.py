#!/usr/bin/env python3
"""
Train and evaluate a regression MLP on HDF5 data using PyTorch.

Supports multiple input files. Filters rows with non finite targets.

Typical run:
  python train_regression_pytorch.py \
    --input-h5 /path/a.h5 /path/b.h5 \
    --features-dset jet_features \
    --targets-dset targets \
    --weights-dset weights \
    --output-dir out_mass_reg \
    --standardize \
    --loss huber
"""
import os
import argparse
import h5py
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import OrderedDict

# ------------- data -------------
def _read_one(h5_path, features_key, targets_key, weights_key):
    with h5py.File(h5_path, "r") as f:
        fx = features_key if features_key in f else ("jet_features" if "jet_features" in f else None)
        ty = targets_key if targets_key in f else ("targets" if "targets" in f else ("labels" if "labels" in f else None))
        wx = None
        if weights_key is not None and weights_key in f:
            wx = weights_key
        elif "weights" in f:
            wx = "weights"
        if fx is None or ty is None:
            raise KeyError(f"{h5_path} missing datasets for features or targets")
        X = f[fx][:]
        y = f[ty][:]
        w = f[wx][:] if wx is not None else None
    y = np.squeeze(y)
    if y.ndim != 1:
        raise ValueError(f"{h5_path} targets must be 1D")
    if w is None:
        w = np.ones_like(y, dtype=np.float32)
    return X.astype(np.float32), y.astype(np.float32), w.astype(np.float32)

def load_h5_multi(paths, features_key, targets_key, weights_key=None):
    Xs, ys, ws = [], [], []
    feat_dim = None
    total = 0
    for p in paths:
        X, y, w = _read_one(p, features_key, targets_key, weights_key)
        if feat_dim is None:
            feat_dim = X.shape[1]
        elif X.shape[1] != feat_dim:
            raise ValueError(f"Feature width mismatch across files. Saw {feat_dim} and {X.shape[1]} in {p}")
        Xs.append(X)
        ys.append(y)
        ws.append(w)
        total += len(y)
        print(f"Loaded {len(y)} rows from {p}, features {X.shape[1]}")
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    w = np.concatenate(ws, axis=0)
    print(f"Total rows {total}")
    return X, y, w

def split_data(X, y, w, test_size=0.2, val_size=0.2, random_state=42):
    X_temp, X_test, y_temp, y_test, w_temp, w_test = train_test_split(
        X, y, w, test_size=test_size, random_state=random_state, shuffle=True
    )
    val_size_adj = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_temp, y_temp, w_temp, test_size=val_size_adj, random_state=random_state, shuffle=True
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, w_train, w_val, w_test

def standardize_fit(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0.0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)

def standardize_apply(X, mean, std):
    return (X - mean) / std

# ------------- model -------------
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout_rate=0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for i, size in enumerate(hidden_sizes, 1):
            layers.append((f"linear{i}", nn.Linear(prev, size)))
            layers.append((f"bn{i}", nn.BatchNorm1d(size)))
            layers.append((f"relu{i}", nn.ReLU()))
            if dropout_rate > 0:
                layers.append((f"dropout{i}", nn.Dropout(dropout_rate)))
            prev = size
        layers.append(("output", nn.Linear(prev, 1)))
        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.net(x).squeeze(1)

# ------------- weighted losses -------------
class WeightedMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.MSELoss(reduction="none")
    def forward(self, preds, targets, weights):
        loss = self.base(preds, targets)
        return torch.sum(weights * loss) / torch.sum(weights)

class WeightedHuber(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.base = nn.HuberLoss(delta=delta, reduction="none")
    def forward(self, preds, targets, weights):
        loss = self.base(preds, targets)
        return torch.sum(weights * loss) / torch.sum(weights)

class WeightedMSLE(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, preds, targets, weights):
        preds_c = torch.clamp(preds, min=0.0)
        targets_c = torch.clamp(targets, min=0.0)
        loss = (torch.log1p(preds_c) - torch.log1p(targets_c)) ** 2
        return torch.sum(weights * loss) / torch.sum(weights)

def make_loss(name, huber_delta=1.0):
    name = name.lower()
    if name == "mse":
        return WeightedMSE()
    if name == "huber":
        return WeightedHuber(delta=huber_delta)
    if name == "msle":
        return WeightedMSLE()
    raise ValueError(f"Unknown loss {name}")

# ------------- train and eval -------------
def make_loader(X, y, w, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float(), torch.from_numpy(w).float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    count = 0
    for Xb, yb, wb in loader:
        Xb, yb, wb = Xb.to(device), yb.to(device), wb.to(device)
        optimizer.zero_grad()
        preds = model(Xb)
        loss = criterion(preds, yb, wb)
        loss.backward()
        optimizer.step()
        total += loss.item() * Xb.size(0)
        count += Xb.size(0)
    return total / max(count, 1)

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total = 0.0
    count = 0
    preds_all, trues_all = [], []
    for Xb, yb, wb in loader:
        Xb, yb, wb = Xb.to(device), yb.to(device), wb.to(device)
        preds = model(Xb)
        loss = criterion(preds, yb, wb)
        total += loss.item() * Xb.size(0)
        count += Xb.size(0)
        preds_all.append(preds.cpu().numpy())
        trues_all.append(yb.cpu().numpy())
    preds_all = np.concatenate(preds_all)
    trues_all = np.concatenate(trues_all)
    return total / max(count, 1), trues_all, preds_all

# ------------- plots -------------
def plot_loss(train_losses, val_losses, path):
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_weights(model, path):
    w = np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])
    w = np.abs(w)
    plt.figure()
    plt.hist(w, bins=100, log=True)
    plt.xlabel("Absolute weight value")
    plt.ylabel("Count, log")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_pred_vs_true(y_true, y_pred, path):
    plt.figure()
    plt.scatter(y_true, y_pred, s=6, alpha=0.5)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_residuals(y_true, y_pred, path):
    res = y_pred - y_true
    plt.figure()
    plt.hist(res, bins=80, alpha=0.9)
    plt.xlabel("Residual, pred minus true")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ------------- metrics -------------
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_pred - y_true)))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

# ------------- main -------------
def main():
    ap = argparse.ArgumentParser(description="Train regression MLP with PyTorch")
    ap.add_argument("--input-h5", nargs="+", required=True)
    ap.add_argument("--features-dset", default="features",
                    help="Will try 'jet_features' if missing")
    ap.add_argument("--targets-dset", default="targets",
                    help="Will try 'targets' or 'labels' if missing")
    ap.add_argument("--weights-dset", default="weights",
                    help="Optional weights dataset")

    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 64, 32])
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--test-split", type=float, default=0.2)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--loss", choices=["mse", "huber", "msle"], default="mse")
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument("--standardize", action="store_true")

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y, w = load_h5_multi(args.input_h5, args.features-dset if False else args.features_dset,
                            args.targets-dset if False else args.targets_dset,
                            args.weights-dset if False else args.weights_dset)

    # filter non finite targets
    mask = np.isfinite(y)
    if not np.all(mask):
        kept = int(mask.sum())
        dropped = int((~mask).sum())
        print(f"Filtering non finite targets, kept {kept}, dropped {dropped}")
        X, y, w = X[mask], y[mask], w[mask]

    X_train, X_val, X_test, y_train, y_val, y_test, w_train, w_val, w_test = split_data(
        X, y, w, test_size=args.test_split, val_size=args.val_split, random_state=args.random_state
    )

    if args.standardize:
        mean, std = standardize_fit(X_train)
        X_train = standardize_apply(X_train, mean, std)
        X_val = standardize_apply(X_val, mean, std)
        X_test = standardize_apply(X_test, mean, std)
        with open(os.path.join(args.output_dir, "feature_scaler.json"), "w") as f:
            json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)

    train_loader = make_loader(X_train, y_train, w_train, args.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, w_val, args.batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test, w_test, args.batch_size, shuffle=False)

    model = MLPRegressor(X_train.shape[1], args.hidden_sizes, args.dropout).to(device)
    criterion = make_loss(args.loss, args.huber_delta)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val = float("inf")
    bad_epochs = 0
    train_losses, val_losses = [], []
    best_model_path = os.path.join(args.output_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, criterion, optimizer, device)
        va, _, _ = eval_epoch(model, val_loader, criterion, device)
        train_losses.append(tr)
        val_losses.append(va)
        print(f"Epoch {epoch:3d}  train_loss={tr:.6f}  val_loss={va:.6f}")
        if va < best_val:
            best_val = va
            bad_epochs = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping")
                break

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    plot_loss(train_losses, val_losses, os.path.join(args.output_dir, "loss_curve.png"))
    plot_weights(model, os.path.join(args.output_dir, "weight_dist.png"))

    val_loss, yv, pv = eval_epoch(model, val_loader, criterion, device)
    test_loss, yt, pt = eval_epoch(model, test_loader, criterion, device)

    val_mae, val_rmse, val_r2 = mae(yv, pv), rmse(yv, pv), r2(yv, pv)
    test_mae, test_rmse, test_r2 = mae(yt, pt), rmse(yt, pt), r2(yt, pt)

    plot_pred_vs_true(yv, pv, os.path.join(args.output_dir, "pred_vs_true_val.png"))
    plot_residuals(yv, pv, os.path.join(args.output_dir, "residuals_val.png"))
    plot_pred_vs_true(yt, pt, os.path.join(args.output_dir, "pred_vs_true_test.png"))
    plot_residuals(yt, pt, os.path.join(args.output_dir, "residuals_test.png"))

    report = {
        "val": {"loss": float(val_loss), "mae": float(val_mae), "rmse": float(val_rmse), "r2": float(val_r2)},
        "test": {"loss": float(test_loss), "mae": float(test_mae), "rmse": float(test_rmse), "r2": float(test_r2)},
        "best_model_path": best_model_path,
        "loss_type": args.loss,
        "huber_delta": args.huber_delta if args.loss == "huber" else None,
        "standardized": bool(args.standardize),
        "hidden_sizes": args.hidden_sizes,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "epochs_trained": len(train_losses),
        "num_input_files": len(args.input_h5),
    }
    with open(os.path.join(args.output_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("Validation  loss {:.6f}  MAE {:.6f}  RMSE {:.6f}  R2 {:.4f}".format(val_loss, val_mae, val_rmse, val_r2))
    print("Test        loss {:.6f}  MAE {:.6f}  RMSE {:.6f}  R2 {:.4f}".format(test_loss, test_mae, test_rmse, test_r2))
    print(f"Best model saved to {best_model_path}")

if __name__ == "__main__":
    main()
