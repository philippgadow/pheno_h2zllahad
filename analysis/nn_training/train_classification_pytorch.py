#!/usr/bin/env python3
"""Train a classification MLP on jet_features using PyTorch."""

import argparse
import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
try:
    import optuna
except ImportError:
    optuna = None

SENTINEL = -999999.0


@dataclass
class ClassificationDataset:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def prepare_datasets(args) -> ClassificationDataset:
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(
        args.input_h5,
        features_key=args.features_key,
        test_size=args.test_split,
        random_state=args.random_state,
    )
    print(
        f"Split sizes - Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)} "
        f"(signal frac train={y_train.mean():.3f})"
    )
    return ClassificationDataset(X_train, X_val, X_test, y_train, y_val, y_test)


def create_loaders(data: ClassificationDataset, batch_size: int):
    train_loader = make_loader(data.X_train, data.y_train, batch_size)
    val_loader = make_loader(data.X_val, data.y_val, batch_size, shuffle=False)
    test_loader = make_loader(data.X_test, data.y_test, batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def hyperparams_from_args(args) -> Dict[str, float]:
    return {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "hidden_sizes": args.hidden_sizes,
        "dropout": args.dropout,
        "patience": args.patience,
    }


def merge_hyperparams(base: Dict[str, float], overrides: Optional[Dict[str, float]]) -> Dict[str, float]:
    merged = dict(base)
    if overrides:
        merged.update(overrides)
    return merged


def suggest_optuna_overrides(trial: "optuna.trial.Trial", args) -> Dict[str, float]:
    overrides: Dict[str, float] = {}
    overrides["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    overrides["learning_rate"] = trial.suggest_float("learning_rate", 3e-4, 5e-3, log=True)
    overrides["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
    n_layers = trial.suggest_int("n_layers", 2, 5)
    hidden_sizes: List[int] = []
    for i in range(n_layers):
        hidden_sizes.append(trial.suggest_int(f"hidden_size_layer_{i}", 32, 256, step=32))
    overrides["hidden_sizes"] = hidden_sizes
    overrides["epochs"] = args.epochs
    overrides["patience"] = args.patience
    return overrides


def overrides_from_trial_params(params: Dict[str, float], args) -> Dict[str, float]:
    overrides = {
        "batch_size": int(params["batch_size"]),
        "learning_rate": float(params["learning_rate"]),
        "dropout": float(params["dropout"]),
        "epochs": args.epochs,
        "patience": args.patience,
    }
    n_layers = int(params["n_layers"])
    overrides["hidden_sizes"] = [int(params[f"hidden_size_layer_{i}"]) for i in range(n_layers)]
    return overrides


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


def compute_significance(y_true, y_pred, save_path: Optional[str] = None):
    thresholds = np.linspace(0, 1, 101)
    sigs = []
    for t in thresholds:
        S = np.sum((y_pred >= t) & (y_true == 1))
        B = np.sum((y_pred >= t) & (y_true == 0))
        sigs.append(S / np.sqrt(B) if B > 0 else 0)
    sigs = np.array(sigs)
    idx = np.argmax(sigs)
    best_t, best_sig = thresholds[idx], sigs[idx]
    if save_path:
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


def plot_roc_curve(y_true, y_pred, save_path: Optional[str] = None):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    if save_path:
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    return float(roc_auc)


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


def run_training(
    args,
    data: ClassificationDataset,
    device: torch.device,
    hyperparams: Dict[str, float],
    output_dir: str,
    save_artifacts: bool = True,
    trial: Optional["optuna.trial.Trial"] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    batch_size = int(hyperparams["batch_size"])
    epochs = int(hyperparams["epochs"])
    patience = int(hyperparams["patience"])
    learning_rate = float(hyperparams["learning_rate"])
    hidden_sizes = [int(h) for h in hyperparams["hidden_sizes"]]
    dropout = float(hyperparams["dropout"])

    train_loader, val_loader, test_loader = create_loaders(data, batch_size)

    model = MLP(data.X_train.shape[1], hidden_sizes, dropout).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float("inf")
    patience_counter = 0
    best_state = None
    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = eval_epoch(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if save_artifacts:
            print(f"Epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if save_artifacts:
                    print("Early stopping triggered")
                break

        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    val_loss, y_val_true, y_val_pred = eval_epoch(model, val_loader, criterion, device)
    test_loss, y_test_true, y_test_pred = eval_epoch(model, test_loader, criterion, device)

    best_thr, best_sig = compute_significance(y_val_true, y_val_pred)
    val_auc = plot_roc_curve(y_val_true, y_val_pred)
    test_auc = plot_roc_curve(y_test_true, y_test_pred)

    results = {
        "val_loss": float(val_loss),
        "test_loss": float(test_loss),
        "val_auc": float(val_auc),
        "test_auc": float(test_auc),
        "best_threshold": float(best_thr),
        "best_significance": float(best_sig),
    }

    if save_artifacts:
        plot_loss(train_losses, val_losses, os.path.join(output_dir, "loss_curve.png"))
        plot_weights(model, os.path.join(output_dir, "weight_dist.png"))
        plot_predictions(y_val_true, y_val_pred, os.path.join(output_dir, "predictions_val.png"))
        plot_predictions(y_test_true, y_test_pred, os.path.join(output_dir, "predictions_test.png"))
        compute_significance(
            y_val_true, y_val_pred, os.path.join(output_dir, "significance.png")
        )
        plot_roc_curve(
            y_val_true, y_val_pred, os.path.join(output_dir, "roc_val.png")
        )
        plot_roc_curve(
            y_test_true, y_test_pred, os.path.join(output_dir, "roc_test.png")
        )

        best_model_path = os.path.join(output_dir, "best_model.pth")
        torch.save(best_state, best_model_path)

        metrics = {
            "best_threshold": float(best_thr),
            "best_significance": float(best_sig),
            "val_auc": float(val_auc),
            "test_auc": float(test_auc),
            "val_loss": float(val_loss),
            "test_loss": float(test_loss),
            "hyperparameters": hyperparams,
        }
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        print(
            f"Best threshold (val): {best_thr:.3f}, significance: {best_sig:.3f}\n"
            f"Validation AUC: {val_auc:.3f}, Test AUC: {test_auc:.3f}"
        )
        print(f"Best model saved to {best_model_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train NN for jet tagging (PyTorch)")
    parser.add_argument("--input-h5", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--features-key", default="ghost_track_vars_with_reg")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 32, 16])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--optuna-trials", type=int, default=0,
                        help="Number of Optuna trials (0 disables HPO)")
    parser.add_argument("--optuna-study", type=str, default=None,
                        help="Optuna study name (optional)")
    parser.add_argument("--optuna-storage", type=str, default=None,
                        help="Optuna storage URI (e.g. sqlite:///study.db)")
    parser.add_argument("--optuna-direction", choices=["minimize", "maximize"], default="maximize",
                        help="Optimization direction (default: maximize validation AUC)")
    parser.add_argument("--optuna-timeout", type=float, default=None,
                        help="Optional timeout (seconds) for Optuna optimization")
    parser.add_argument("--optuna-pruner", choices=["median", "none"], default="median",
                        help="Pruner to use for Optuna trials (default: median)")
    args = parser.parse_args()

    if args.optuna_trials > 0 and optuna is None:
        raise RuntimeError("Optuna is not installed. Please install optuna or set --optuna-trials 0.")

    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_bundle = prepare_datasets(args)
    base_hparams = hyperparams_from_args(args)

    if args.optuna_trials and args.optuna_trials > 0:
        pruner = (
            optuna.pruners.MedianPruner()
            if args.optuna_pruner == "median"
            else optuna.pruners.NopPruner()
        )
        study = optuna.create_study(
            study_name=args.optuna_study,
            direction=args.optuna_direction,
            storage=args.optuna_storage,
            load_if_exists=bool(args.optuna_storage and args.optuna_study),
            pruner=pruner,
        )

        def objective(trial: "optuna.trial.Trial"):
            overrides = suggest_optuna_overrides(trial, args)
            params = merge_hyperparams(base_hparams, overrides)
            result = run_training(
                args,
                data_bundle,
                device,
                params,
                args.output_dir,
                save_artifacts=False,
                trial=trial,
            )
            if args.optuna_direction == "minimize":
                value = result["val_loss"]
                if not np.isfinite(value):
                    value = np.inf
            else:
                value = result["val_auc"]
                if not np.isfinite(value):
                    value = -np.inf
            return value

        print("\n" + "=" * 80)
        print(f"Running Optuna study with {args.optuna_trials} trials...")
        print("=" * 80)
        study.optimize(objective, n_trials=args.optuna_trials, timeout=args.optuna_timeout)

        best_trial = study.best_trial
        print(f"\nBest trial #{best_trial.number}: value={best_trial.value:.6f}")
        best_overrides = overrides_from_trial_params(best_trial.params, args)
        final_hparams = merge_hyperparams(base_hparams, best_overrides)

        summary = {
            "direction": args.optuna_direction,
            "best_value": best_trial.value,
            "best_params": best_trial.params,
            "best_hyperparams": final_hparams,
            "n_trials": len(study.trials),
        }
        summary_path = os.path.join(args.output_dir, "optuna_best.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Optuna summary saved to {summary_path}")

        run_training(
            args,
            data_bundle,
            device,
            final_hparams,
            args.output_dir,
            save_artifacts=True,
        )
    else:
        run_training(
            args,
            data_bundle,
            device,
            base_hparams,
            args.output_dir,
            save_artifacts=True,
        )


if __name__ == "__main__":
    main()
