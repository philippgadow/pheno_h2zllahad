#!/usr/bin/env python3
"""Train a classification MLP on jet_features using PyTorch."""

import argparse
import json
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


SENTINEL = -999999.0


def load_data(h5_path, features_key="jet_features", label_key="is_signal", test_size=0.2, random_state=42):
    with h5py.File(h5_path, "r") as f:
        X = f[features_key][:]
        if label_key in f:
            y = f[label_key][:]
        else:
            y = f["labels"][:]
            if y.ndim > 1:
                y = np.argmax(y, axis=1)
            y = (y == 36).astype(np.float32)

    if label_key == "is_signal":
        y = (y > 0.5).astype(np.float32)
    else:
        y = y.astype(np.float32)

    mask = np.all(np.isfinite(X), axis=1)
    mask &= np.all(X != SENTINEL, axis=1)
    mask &= np.isfinite(y)
    X = X[mask]
    y = y[mask]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_size = test_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size,
        random_state=random_state,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


class MLP(nn.Module):
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
        layers.append(("sigmoid", nn.Sigmoid()))
        self.net = nn.Sequential(*[layer for _, layer in layers])

    def forward(self, x):
        return self.net(x).squeeze(1)


def make_loader(X, y, batch_size, shuffle=True):
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def plot_loss(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_predictions(y_true, y_pred, save_path):
    plt.figure()
    plt.hist(y_pred[y_true == 0], bins=50, alpha=0.5, label="background", density=True)
    plt.hist(y_pred[y_true == 1], bins=50, alpha=0.5, label="signal", density=True)
    plt.xlabel("Predicted score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_weights(model, save_path):
    weights = np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])
    weights = np.abs(weights)
    plt.figure()
    plt.hist(weights, bins=100, log=True)
    plt.xlabel("|weight|")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def compute_significance(y_true, y_pred, save_path):
    thresholds = np.linspace(0, 1, 101)
    sigs = []
    for t in thresholds:
        S = np.sum((y_pred >= t) & (y_true == 1))
        B = np.sum((y_pred >= t) & (y_true == 0))
        sigs.append(S / np.sqrt(B) if B > 0 else 0)
    sigs = np.array(sigs)
    idx = np.argmax(sigs)
    best_t, best_sig = thresholds[idx], sigs[idx]
    plt.figure()
    plt.plot(thresholds, sigs)
    plt.scatter([best_t], [best_sig], color="red", label=f"best={best_t:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("S/sqrt(B)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return best_t, best_sig


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    return running_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds, trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item() * X_batch.size(0)
            preds.append(outputs.cpu().numpy())
            trues.append(y_batch.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return running_loss / len(loader.dataset), trues, preds


def main():
    parser = argparse.ArgumentParser(description="Train NN for jet tagging (PyTorch)")
    parser.add_argument("--input-h5", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 32, 16])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(
        args.input_h5, test_size=args.test_split, random_state=args.random_state
    )

    train_loader = make_loader(X_train, y_train, args.batch_size)
    val_loader = make_loader(X_val, y_val, args.batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test, args.batch_size, shuffle=False)

    model = MLP(X_train.shape[1], args.hidden_sizes, args.dropout).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_loss, epochs_no_improve = float("inf"), 0
    train_losses, val_losses = [], []
    best_model_path = os.path.join(args.output_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = eval_epoch(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        if val_loss < best_loss:
            best_loss, epochs_no_improve = val_loss, 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print("Early stopping")
                break

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    plot_loss(train_losses, val_losses, os.path.join(args.output_dir, "loss_curve.png"))
    plot_weights(model, os.path.join(args.output_dir, "weight_dist.png"))

    _, y_val_true, y_val_pred = eval_epoch(model, val_loader, criterion, device)
    plot_predictions(y_val_true, y_val_pred, os.path.join(args.output_dir, "predictions_val.png"))
    best_thr, best_sig = compute_significance(
        y_val_true, y_val_pred, os.path.join(args.output_dir, "significance.png")
    )

    _, y_test_true, y_test_pred = eval_epoch(model, test_loader, criterion, device)
    plot_predictions(y_test_true, y_test_pred, os.path.join(args.output_dir, "predictions_test.png"))

    metrics = {
        "best_threshold": float(best_thr),
        "best_significance": float(best_sig),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Best threshold (val): {best_thr:.3f}, significance: {best_sig:.3f}")
    print(f"Best model saved to {best_model_path}")


if __name__ == "__main__":
    main()
