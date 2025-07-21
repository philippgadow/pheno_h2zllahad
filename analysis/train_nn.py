#!/usr/bin/env python3
"""
Train and evaluate a neural network on jet tagging data saved in HDF5 format using PyTorch.

Features:
- Load 'jet_features', 'labels' from HDF5
- Split into train, validation, and test sets
- Build configurable MLP with BatchNorm and Dropout
- Train with early stopping and checkpointing
- Plot training & validation loss
- Plot model weight distribution
- Evaluate on validation & test sets: prediction histograms
- Compute significance (S/sqrt(B)) vs threshold and find optimal

Use:
    python train_nn_pytorch.py \
        --input-h5 /path/to/jet_data.h5 \
        --output-dir /path/to/output \
        --batch-size 128 \
        --epochs 50 \
        --learning-rate 1e-3 \
        --hidden-sizes 64 32 16 \
        --dropout 0.1 \
        --test-split 0.2
"""
import os
import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import OrderedDict


def load_data(h5_path, test_size=0.2, random_state=42):
    with h5py.File(h5_path, 'r') as f:
        X = f['jet_features'][:]
        y = f['labels'][:]

    # if one‑hot encoded, flatten to integer labels
    if y.ndim > 1:
        y = np.argmax(y, axis=1)

    # binary target: flavor 36 (hadronically decaying BSM higgs) vs all others
    # note: you could also choose a different target, e.g. eta_c (PID: 441) or J/Psi (PID: 443)
    y = (y == 36).astype(np.float32)

    # split off test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # split train/val
    val_size = test_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size,
        random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def plot_feature_distributions(X, y, output_dir):
    """
    For each jet feature, plot histograms of background (y==0) vs signal (y==1)
    and save into output_dir as feature_<name>.png
    """
    # hard-coded, to be changed
    feature_names = ['PT', 'Eta', 'Phi', 'Mass', 'BTag', 'TauTag']
    # ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    for idx, name in enumerate(feature_names):
        plt.figure()
        plt.hist(X[y==0, idx], bins=50, alpha=0.5, label='background', density=True)
        plt.hist(X[y==1, idx], bins=50, alpha=0.5, label='signal', density=True)
        plt.xlabel(name)
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'feature_{name}.png'))
        plt.close()



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout_rate=0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for i, size in enumerate(hidden_sizes, 1):
            layers.append((f'linear{i}', nn.Linear(prev, size)))
            layers.append((f'bn{i}', nn.BatchNorm1d(size)))
            layers.append((f'relu{i}', nn.ReLU()))
            if dropout_rate > 0:
                layers.append((f'dropout{i}', nn.Dropout(dropout_rate)))
            prev = size
        layers.append(('output', nn.Linear(prev, 1)))
        layers.append(('sigmoid', nn.Sigmoid()))
        # pass an OrderedDict, not a plain dict
        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.net(x).squeeze(1)


def plot_loss(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_weights(model, save_path):
    weights = np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])
    weights = np.abs(weights)
    plt.figure()
    plt.hist(weights, bins=100, log=True)
    plt.xlabel('Absolute weight value')
    plt.ylabel('Count (log)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_predictions(y_true, y_pred, save_path):
    plt.figure()
    plt.hist(y_pred[y_true==0], bins=50, alpha=0.5, label='background')
    plt.hist(y_pred[y_true==1], bins=50, alpha=0.5, label='signal')
    plt.xlabel('Predicted score')
    plt.ylabel('Count')
    plt.legend()
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
    plt.scatter([best_t], [best_sig], color='red', label=f'best={best_t:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('S/√B')
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
    parser = argparse.ArgumentParser(description='Train NN for jet tagging (PyTorch)')
    parser.add_argument('--input-h5', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[64, 32, 16])
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--test-split', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(
        args.input_h5, test_size=args.test_split, random_state=args.random_state
    )

    # plot all jet-feature distributions on the test set
    plot_feature_distributions(X_test, y_test, args.output_dir)


    def make_loader(X, y):
        return DataLoader(
            TensorDataset(torch.from_numpy(X).float(),
                          torch.from_numpy(y).float()),
            batch_size=args.batch_size,
            shuffle=True
        )

    train_loader = make_loader(X_train, y_train)
    val_loader = make_loader(X_val, y_val)
    test_loader = make_loader(X_test, y_test)

    model = MLP(X_train.shape[1], args.hidden_sizes, args.dropout).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_loss, epochs_no_improve = float('inf'), 0
    train_losses, val_losses = [], []
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')

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

    model.load_state_dict(torch.load(best_model_path))

    plot_loss(train_losses, val_losses, os.path.join(args.output_dir, 'loss_curve.png'))
    plot_weights(model, os.path.join(args.output_dir, 'weight_dist.png'))

    _, y_val_true, y_val_pred = eval_epoch(model, val_loader, criterion, device)
    plot_predictions(y_val_true, y_val_pred, os.path.join(args.output_dir, 'predictions_val.png'))
    best_thr, best_sig = compute_significance(
        y_val_true, y_val_pred,
        os.path.join(args.output_dir, 'significance.png')
    )

    _, y_test_true, y_test_pred = eval_epoch(model, test_loader, criterion, device)
    plot_predictions(y_test_true, y_test_pred, os.path.join(args.output_dir, 'predictions_test.png'))

    print(f"Best threshold (val): {best_thr:.3f}, significance: {best_sig:.3f}")
    print(f"Best model saved to {best_model_path}")


if __name__ == '__main__':
    main()
