#!/usr/bin/env python3
"""
Train a regression MLP on ghost track variables using PyTorch.
Enhanced version with plots matching the original Keras script.

Usage:
    python analysis/nn_training/train_regression_pytorch.py \
      --input-h5 nn_training_input/jet_data.h5 \
      --output-dir nn_training_output/regression \
      --loss msle \
      --standardize robust
"""

import os
import argparse
import h5py
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

SENTINEL = -999999.0


def _decode_if_bytes(value):
    """Decode numpy byte strings into Python str."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="ignore").rstrip("\x00")
    return str(value)

# ============================================================================
# Data Loading
# ============================================================================


def load_h5_file(h5_path, features_key="ghost_track_vars", targets_key="targets", class_key="signal_class"):
    """Load features and targets from a single HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        if features_key not in f:
            raise KeyError(f"{h5_path} missing '{features_key}' dataset")
        if targets_key not in f:
            raise KeyError(f"{h5_path} missing '{targets_key}' dataset")

        X = f[features_key][:]
        y = f[targets_key][:]
        if class_key in f:
            signal_class = f[class_key][:]
        else:
            signal_class = np.full(len(y), -1, dtype=np.int64)

    y = np.squeeze(y)
    if y.ndim != 1:
        raise ValueError(f"{h5_path} targets must be 1D, got shape {y.shape}")

    return X.astype(np.float32), y.astype(np.float32), signal_class.astype(np.int64)


def load_h5_multi(paths, features_key="ghost_track_vars", targets_key="targets",
                  class_key="signal_class", filter_nonfinite=True):
    """Load and concatenate data from multiple HDF5 files."""
    Xs, ys, class_list = [], [], []
    feat_dim = None

    for p in paths:
        X, y, cls = load_h5_file(p, features_key, targets_key, class_key)

        if feat_dim is None:
            feat_dim = X.shape[1]
        elif X.shape[1] != feat_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {feat_dim}, got {X.shape[1]} in {p}"
            )

        Xs.append(X)
        ys.append(y)
        class_list.append(cls)
        print(f"Loaded {len(y)} rows from {os.path.basename(p)}")

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    classes = np.concatenate(class_list, axis=0)

    print(f"Total rows before filtering: {len(y)}")

    if filter_nonfinite:
        mask = np.isfinite(y)
        n_dropped = int((~mask).sum())
        if n_dropped > 0:
            print(f"Filtering {n_dropped} rows with non-finite targets")
            X = X[mask]
            y = y[mask]
            classes = classes[mask]
            print(f"Total rows after filtering: {len(y)}")

    valid_features = np.all(np.isfinite(X), axis=1)
    valid_features &= np.all(X != SENTINEL, axis=1)
    n_invalid = int((~valid_features).sum())
    if n_invalid > 0:
        print(f"Filtering {n_invalid} rows with invalid features")
        X = X[valid_features]
        y = y[valid_features]
        classes = classes[valid_features]
        print(f"Total rows after feature filtering: {len(y)}")

    if len(y) == 0:
        raise ValueError(
            "No valid regression examples left after filtering non-finite targets/features."
            " Check that the provided HDF5 file(s) contain signal jets with valid truth masses."
        )

    return X, y, classes


def split_data(X, y, class_ids=None, test_size=0.2, val_size=0.2, random_state=42):
    """Split data into train/val/test sets, keeping class IDs aligned if provided."""
    arrays = [X, y]
    if class_ids is not None:
        arrays.append(class_ids)

    split = train_test_split(
        *arrays, test_size=test_size, random_state=random_state, shuffle=True
    )

    if class_ids is not None:
        X_temp, X_test, y_temp, y_test, c_temp, c_test = split
    else:
        X_temp, X_test, y_temp, y_test = split
        c_temp = c_test = None

    val_fraction = val_size / (1.0 - test_size)
    arrays = [X_temp, y_temp]
    if c_temp is not None:
        arrays.append(c_temp)

    split = train_test_split(
        *arrays, test_size=val_fraction, random_state=random_state, shuffle=True
    )

    if c_temp is not None:
        X_train, X_val, y_train, y_val, c_train, c_val = split
    else:
        X_train, X_val, y_train, y_val = split
        c_train = c_val = None

    print(f"Split sizes - Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, c_train, c_val, c_test


def load_signal_class_lookup(h5_path):
    """Load optional metadata about each signal class."""
    lookup = {}
    try:
        with h5py.File(h5_path, "r") as f:
            if "signal_class_ids" not in f:
                return lookup

            ids = f["signal_class_ids"][:]
            names = f["signal_class_names"][:] if "signal_class_names" in f else None
            keys = f["signal_class_keys"][:] if "signal_class_keys" in f else None
            masses = f["signal_class_mass_GeV"][:] if "signal_class_mass_GeV" in f else None
            truth_pids = f["signal_class_truth_pid"][:] if "signal_class_truth_pid" in f else None

            for i, class_id in enumerate(ids):
                entry = {"class_id": int(class_id)}
                if names is not None and i < len(names):
                    entry["name"] = _decode_if_bytes(names[i])
                if keys is not None and i < len(keys):
                    entry["key"] = _decode_if_bytes(keys[i])
                if masses is not None and i < len(masses):
                    entry["mass_GeV"] = float(masses[i])
                if truth_pids is not None and i < len(truth_pids):
                    entry["truth_pid"] = int(truth_pids[i])
                lookup[int(class_id)] = entry
    except OSError as exc:
        print(f"Warning: failed to read signal class metadata from {h5_path}: {exc}")

    return lookup


# ============================================================================
# Data Standardization
# ============================================================================


def fit_scaler(X, scaler_type="robust"):
    """Fit a scaler to the training data."""
    if scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")

    scaler.fit(X)
    return scaler


def save_scaler(scaler, path):
    """Save scaler to pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)


def load_scaler(path):
    """Load scaler from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


# ============================================================================
# Model Definition
# ============================================================================


class MLPRegressor(nn.Module):
    """Multi-layer perceptron for regression."""

    def __init__(self, input_dim=7, hidden_sizes=[50, 50, 50, 50, 50],
                 dropout_rate=0.0, use_batch_norm=True):
        super().__init__()

        layers = []
        prev_size = input_dim

        for i, hidden_size in enumerate(hidden_sizes, 1):
            layers.append((f"linear{i}", nn.Linear(prev_size, hidden_size)))

            if use_batch_norm:
                layers.append((f"bn{i}", nn.BatchNorm1d(hidden_size)))

            layers.append((f"relu{i}", nn.ReLU()))

            if dropout_rate > 0:
                layers.append((f"dropout{i}", nn.Dropout(dropout_rate)))

            prev_size = hidden_size

        layers.append(("output", nn.Linear(prev_size, 1)))

        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.net(x).squeeze(1)


# ============================================================================
# Loss Functions
# ============================================================================


class WeightedMSE(nn.Module):
    """Weighted Mean Squared Error."""

    def __init__(self):
        super().__init__()
        self.base = nn.MSELoss(reduction='none')

    def forward(self, preds, targets, weights=None):
        loss = self.base(preds, targets)
        if weights is not None:
            return torch.sum(weights * loss) / torch.sum(weights)
        return torch.mean(loss)


class WeightedMSLE(nn.Module):
    """Weighted Mean Squared Logarithmic Error."""

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, weights=None):
        preds_c = torch.clamp(preds, min=0.0)
        targets_c = torch.clamp(targets, min=0.0)

        loss = (torch.log1p(preds_c) - torch.log1p(targets_c)) ** 2

        if weights is not None:
            return torch.sum(weights * loss) / torch.sum(weights)
        return torch.mean(loss)


class WeightedHuber(nn.Module):
    """Weighted Huber Loss."""

    def __init__(self, delta=1.0):
        super().__init__()
        self.base = nn.HuberLoss(delta=delta, reduction='none')

    def forward(self, preds, targets, weights=None):
        loss = self.base(preds, targets)
        if weights is not None:
            return torch.sum(weights * loss) / torch.sum(weights)
        return torch.mean(loss)


def make_loss(loss_name, huber_delta=1.0):
    """Create a loss function by name."""
    loss_name = loss_name.lower()
    if loss_name == "mse":
        return WeightedMSE()
    elif loss_name == "msle":
        return WeightedMSLE()
    elif loss_name == "huber":
        return WeightedHuber(delta=huber_delta)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


# ============================================================================
# Training and Evaluation
# ============================================================================


def make_dataloader(X, y, batch_size, shuffle=True):
    """Create a PyTorch DataLoader."""
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float()
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        total_samples += X_batch.size(0)

    return total_loss / total_samples


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        preds = model(X_batch)
        loss = criterion(preds, y_batch)

        total_loss += loss.item() * X_batch.size(0)
        total_samples += X_batch.size(0)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    return total_loss / total_samples, all_targets, all_preds


# ============================================================================
# Metrics
# ============================================================================


def compute_metrics(y_true, y_pred):
    """Compute regression metrics."""
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

    return {"mae": mae, "rmse": rmse, "r2": r2}


def summarize_signal_performance(y_true, y_pred, class_ids, tolerance=1.0, class_lookup=None):
    """Summarize per-class accuracy for signal jets only."""
    if class_ids is None:
        return {
            "n_total": 0,
            "n_correct": 0,
            "n_wrong": 0,
            "fraction_correct": None,
            "mae": None,
            "per_class": []
        }

    mask = class_ids >= 0
    total = int(mask.sum())
    if total == 0:
        return {
            "n_total": 0,
            "n_correct": 0,
            "n_wrong": 0,
            "fraction_correct": None,
            "mae": None,
            "per_class": []
        }

    diffs = np.abs(y_pred[mask] - y_true[mask])
    correct = diffs <= tolerance
    per_class = []

    unique_classes = np.unique(class_ids[mask])
    for cls in unique_classes:
        cls_mask = mask & (class_ids == cls)
        cls_diffs = np.abs(y_pred[cls_mask] - y_true[cls_mask])
        cls_correct = cls_diffs <= tolerance
        n_cls = int(cls_mask.sum())
        entry = {
            "class_id": int(cls),
            "n": n_cls,
            "n_correct": int(cls_correct.sum()),
            "n_wrong": int(n_cls - cls_correct.sum()),
            "fraction_correct": float(cls_correct.mean()) if n_cls > 0 else None,
            "mae": float(np.mean(cls_diffs)) if n_cls > 0 else None
        }
        if class_lookup and int(cls) in class_lookup:
            entry.update(class_lookup[int(cls)])
        per_class.append(entry)

    return {
        "n_total": total,
        "n_correct": int(correct.sum()),
        "n_wrong": int(total - correct.sum()),
        "fraction_correct": float(correct.mean()) if total > 0 else None,
        "mae": float(np.mean(diffs)) if total > 0 else None,
        "per_class": per_class
    }


# ============================================================================
# Plotting Functions (matching original script)
# ============================================================================


def plot_loss_curve(train_losses, val_losses, save_path):
    """Plot training and validation loss curves - matches original 'Training_curve_*.png'"""
    fig = plt.figure(figsize=(8, 8))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Test')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    fig.savefig(save_path)
    plt.close()


def plot_weight_distribution(model, save_path):
    """Plot distribution of model weights - matches original 'Weights_*.png'"""
    weights = []
    for param in model.parameters():
        weights.append(param.detach().cpu().numpy().flatten())
    weights = np.concatenate(weights)
    weightsFlat = np.absolute(weights)
    weightsFlat = np.sort(weightsFlat)

    fig = plt.figure(figsize=(8, 8))
    plt.hist(weightsFlat, bins=10**(np.arange(-24, 7)/2.),
             range=(1.e-12, 1.e3), density=False,
             histtype="step", linewidth=1, linestyle="solid", edgecolor="black")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1.e-12, 1.e3)
    plt.xlabel("Value")
    plt.ylabel("nWeights")
    fig.savefig(save_path)
    plt.close()


def plot_predictions_histogram(y_true, y_pred, save_path, title="Predictions", bins=100, range_tuple=None):
    """
    Plot histogram of predicted values - matches original 'Reg_F*.png' and 'Reg_SR*.png'
    This creates a histogram similar to the original script's format.
    """
    fig = plt.figure(figsize=(10, 10))

    # Main plot area (top 70%)
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)

    # Determine plotting range if not provided
    if range_tuple is None:
        combined = np.concatenate([y_true, y_pred])
        finite = combined[np.isfinite(combined)]
        if finite.size == 0:
            range_tuple = (0.0, 1.0)
        else:
            ymin, ymax = np.percentile(finite, [0.5, 99.5])
            span = ymax - ymin
            if span <= 0:
                span = max(abs(ymin), 1.0)
            padding = 0.05 * span
            range_tuple = (float(ymin - padding), float(ymax + padding))

    # Create histogram
    counts_pred, bins_edges = np.histogram(y_pred, bins=bins, range=range_tuple)
    counts_true, _ = np.histogram(y_true, bins=bins, range=range_tuple)
    bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2

    # Plot predicted as black line
    ax1.hist(y_pred, bins=bins, range=range_tuple, histtype='step',
             color='black', linewidth=2, label='Predicted')

    # Plot true as points with error bars
    ax1.errorbar(bin_centers, counts_true, yerr=np.sqrt(counts_true),
                 fmt='o', color='black', markersize=4, label='True')

    ax1.set_ylabel('Events')
    ax1.set_title(title)
    ax1.legend()
    ax1.set_xlim(range_tuple)
    ax1.set_ylim(0, max(counts_pred.max(), counts_true.max()) * 1.3)

    # Ratio plot (bottom 30%)
    ax2 = plt.subplot2grid((4, 1), (3, 0))

    # Calculate ratio
    ratio = np.ones_like(counts_true, dtype=float)
    ratio_err = np.zeros_like(counts_true, dtype=float)
    mask = counts_pred > 0
    ratio[mask] = counts_true[mask] / counts_pred[mask]
    ratio_err[mask] = np.sqrt(counts_true[mask]) / counts_pred[mask]

    # Plot ratio
    ax2.errorbar(bin_centers, ratio, yerr=ratio_err,
                 fmt='o', color='black', markersize=4)
    ax2.axhline(y=1.0, color='black', linestyle='-', linewidth=1)
    ax2.fill_between(range_tuple, 0.95, 1.05, alpha=0.3, color='gray')

    ax2.set_xlabel('Predicted Mass')
    ax2.set_ylabel('True / Pred')
    ax2.set_xlim(range_tuple)
    ax2.set_ylim(0.90, 1.10)
    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()


def plot_predictions(y_true, y_pred, save_path, title="Predictions"):
    """Plot predicted vs true values as scatter plot."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, s=5, alpha=0.5)

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')

    plt.xlabel('True Mass')
    plt.ylabel('Predicted Mass')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_residuals(y_true, y_pred, save_path, title="Residuals"):
    """Plot residual distribution."""
    residuals = y_pred - y_true

    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=80, edgecolor='black', linewidth=0.5, alpha=0.7)
    plt.xlabel('Residual (Predicted - True)')
    plt.ylabel('Count')
    plt.title(title)
    plt.axvline(0, color='red', linestyle='--', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================================
# Main Training Script
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train regression MLP on ghost track variables"
    )

    # Data arguments
    parser.add_argument("--input-h5", nargs="+", required=True,
                        help="Input HDF5 files")
    parser.add_argument("--features-key", default="ghost_track_vars",
                        help="HDF5 dataset key for features")
    parser.add_argument("--targets-key", default="targets",
                        help="HDF5 dataset key for targets (truth mass)")
    parser.add_argument("--class-key", default="signal_class",
                        help="HDF5 dataset key for signal class IDs")

    # Output arguments
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for results")

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Maximum number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[50, 50, 50, 50, 50],
                        help="Hidden layer sizes")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout rate")
    parser.add_argument("--no-batch-norm", action="store_true",
                        help="Disable batch normalization")
    parser.add_argument("--signal-accuracy-tolerance", type=float, default=1.0,
                        help="Tolerance in GeV to consider a signal prediction correct")

    # Loss function
    parser.add_argument("--loss", choices=["mse", "msle", "huber"], default="msle",
                        help="Loss function (default: msle to match original)")
    parser.add_argument("--huber-delta", type=float, default=1.0,
                        help="Delta parameter for Huber loss")

    # Data preprocessing
    parser.add_argument("--standardize", choices=["none", "standard", "robust"], default="robust",
                        help="Standardization method (default: robust to match original)")
    parser.add_argument("--test-split", type=float, default=0.2,
                        help="Test set fraction")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation set fraction")

    # Early stopping
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")

    # Other
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save arguments
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ========================================================================
    # Load Data
    # ========================================================================
    print("\n" + "="*80)
    print("Loading data...")
    print("="*80)

    class_lookup = load_signal_class_lookup(args.input_h5[0])
    X, y, class_ids = load_h5_multi(
        args.input_h5, args.features_key, args.targets_key, args.class_key
    )

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        _class_train,
        class_val,
        class_test,
    ) = split_data(
        X,
        y,
        class_ids=class_ids,
        test_size=args.test_split,
        val_size=args.val_split,
        random_state=args.random_seed,
    )

    # ========================================================================
    # Standardize
    # ========================================================================
    if args.standardize != "none":
        scaler = fit_scaler(X_train, args.standardize)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        scaler_path = os.path.join(args.output_dir, "scaler.pkl")
        save_scaler(scaler, scaler_path)
        print(f"Saved scaler to {scaler_path}")

    # ========================================================================
    # Create DataLoaders
    # ========================================================================
    train_loader = make_dataloader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, args.batch_size, shuffle=False)
    test_loader = make_dataloader(X_test, y_test, args.batch_size, shuffle=False)

    # ========================================================================
    # Create Model
    # ========================================================================
    print("\n" + "="*80)
    print("Creating model...")
    print("="*80)

    model = MLPRegressor(
        input_dim=X_train.shape[1],
        hidden_sizes=args.hidden_sizes,
        dropout_rate=args.dropout,
        use_batch_norm=not args.no_batch_norm
    ).to(device)

    print(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {n_params:,}")

    # ========================================================================
    # Training Setup
    # ========================================================================
    criterion = make_loss(args.loss, args.huber_delta)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    best_model_path = os.path.join(args.output_dir, "best_model.pth")

    # ========================================================================
    # Training Loop
    # ========================================================================
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        val_loss, _, _ = eval_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"Train Loss: {train_loss:.6f}  "
              f"Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  → New best model saved (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

    # ========================================================================
    # Load Best Model and Evaluate
    # ========================================================================
    print("\n" + "="*80)
    print("Loading best model and evaluating...")
    print("="*80)

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    val_loss, y_val_true, y_val_pred = eval_epoch(model, val_loader, criterion, device)
    test_loss, y_test_true, y_test_pred = eval_epoch(model, test_loader, criterion, device)

    val_metrics = compute_metrics(y_val_true, y_val_pred)
    test_metrics = compute_metrics(y_test_true, y_test_pred)

    print("\nValidation Results:")
    print(f"  Loss: {val_loss:.6f}")
    print(f"  MAE:  {val_metrics['mae']:.6f}")
    print(f"  RMSE: {val_metrics['rmse']:.6f}")
    print(f"  R²:   {val_metrics['r2']:.4f}")

    print("\nTest Results:")
    print(f"  Loss: {test_loss:.6f}")
    print(f"  MAE:  {test_metrics['mae']:.6f}")
    print(f"  RMSE: {test_metrics['rmse']:.6f}")
    print(f"  R²:   {test_metrics['r2']:.4f}")

    signal_stats_val = summarize_signal_performance(
        y_val_true, y_val_pred, class_val, args.signal_accuracy_tolerance, class_lookup
    )
    signal_stats_test = summarize_signal_performance(
        y_test_true, y_test_pred, class_test, args.signal_accuracy_tolerance, class_lookup
    )

    print(f"\nSignal-only accuracy (±{args.signal_accuracy_tolerance:.2f} GeV tolerance):")

    def _print_signal_summary(split_name, stats):
        if stats["n_total"] == 0 or stats["fraction_correct"] is None:
            print(f"  {split_name}: no matched signal jets.")
            return
        frac = stats["fraction_correct"] * 100.0
        print(
            f"  {split_name}: {stats['n_correct']}/{stats['n_total']} "
            f"({frac:.1f}%) correct predictions"
        )
        for entry in stats["per_class"]:
            label = entry.get("name") or entry.get("key") or f"class {entry['class_id']}"
            mass = entry.get("mass_GeV")
            if isinstance(mass, (float, int)) and np.isfinite(mass):
                label = f"{label} (m={mass:.2f} GeV)"
            print(
                f"    - {label}: {entry['n_correct']}/{entry['n']} correct "
                f"({entry['fraction_correct']*100:.1f}%), MAE={entry['mae']:.3f}"
            )

    _print_signal_summary("Validation", signal_stats_val)
    _print_signal_summary("Test", signal_stats_test)

    # ========================================================================
    # Create Plots
    # ========================================================================
    print("\n" + "="*80)
    print("Creating plots...")
    print("="*80)

    plot_loss_curve(train_losses, val_losses,
                    os.path.join(args.output_dir, "Training_curve.png"))

    plot_weight_distribution(model,
                             os.path.join(args.output_dir, "Weights.png"))

    plot_predictions_histogram(y_val_true, y_val_pred,
                               os.path.join(args.output_dir, "Reg_Val.png"),
                               title="Validation Set - Full")

    plot_predictions_histogram(y_test_true, y_test_pred,
                               os.path.join(args.output_dir, "Reg_Test.png"),
                               title="Test Set - Full")

    plot_predictions(y_val_true, y_val_pred,
                     os.path.join(args.output_dir, "predictions_val_scatter.png"),
                     title="Validation Set Predictions")

    plot_predictions(y_test_true, y_test_pred,
                     os.path.join(args.output_dir, "predictions_test_scatter.png"),
                     title="Test Set Predictions")

    plot_residuals(y_val_true, y_val_pred,
                   os.path.join(args.output_dir, "residuals_val.png"),
                   title="Validation Set Residuals")

    plot_residuals(y_test_true, y_test_pred,
                   os.path.join(args.output_dir, "residuals_test.png"),
                   title="Test Set Residuals")

    # ========================================================================
    # Save Report
    # ========================================================================
    report = {
        "validation": {
            "loss": float(val_loss),
            "mae": float(val_metrics['mae']),
            "rmse": float(val_metrics['rmse']),
            "r2": float(val_metrics['r2'])
        },
        "test": {
            "loss": float(test_loss),
            "mae": float(test_metrics['mae']),
            "rmse": float(test_metrics['rmse']),
            "r2": float(test_metrics['r2'])
        },
        "training": {
            "epochs_trained": len(train_losses),
            "best_epoch": int(np.argmin(val_losses)) + 1,
            "best_val_loss": float(best_val_loss)
        },
        "model": {
            "architecture": args.hidden_sizes,
            "n_parameters": n_params,
            "loss_function": args.loss,
            "dropout": args.dropout,
            "batch_norm": not args.no_batch_norm
        },
        "data": {
            "n_train": len(y_train),
            "n_val": len(y_val),
            "n_test": len(y_test),
            "n_features": X_train.shape[1],
            "input_files": [os.path.basename(f) for f in args.input_h5]
        },
        "signal_only": {
            "tolerance_GeV": args.signal_accuracy_tolerance,
            "validation": signal_stats_val,
            "test": signal_stats_test
        },
    }

    report_path = os.path.join(args.output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to {report_path}")
    print(f"Best model saved to {best_model_path}")
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)


if __name__ == "__main__":
    main()
