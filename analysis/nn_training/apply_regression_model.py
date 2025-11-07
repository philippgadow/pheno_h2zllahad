#!/usr/bin/env python3
"""
Apply a trained regression network to an HDF5 jet dataset and store the
predictions for later use (e.g. as an extra classification feature).
"""

import argparse
import json
import os
import pickle
import shutil

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
try:
    from train_regression_pytorch import MLPRegressor, SENTINEL
except ImportError:  # pragma: no cover
    from .train_regression_pytorch import MLPRegressor, SENTINEL


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run regression inference on jet HDF5 and store predictions."
    )
    parser.add_argument("--input-h5", required=True, help="Source HDF5 file.")
    parser.add_argument(
        "--output-h5",
        help="Destination HDF5 file (default: overwrite input in-place).",
    )
    parser.add_argument(
        "--regression-run-dir",
        default="nn_training_output/regression",
        help="Directory containing best_model.pth, scaler.pkl, args.json.",
    )
    parser.add_argument(
        "--args-path",
        help="Explicit path to args.json (default: <run_dir>/args.json).",
    )
    parser.add_argument(
        "--model-path",
        help="Explicit path to best_model.pth (default: <run_dir>/best_model.pth).",
    )
    parser.add_argument(
        "--scaler-path",
        help="Explicit path to scaler.pkl (default: <run_dir>/scaler.pkl if present).",
    )
    parser.add_argument(
        "--features-key",
        default="ghost_track_vars",
        help="Dataset key containing regression input features.",
    )
    parser.add_argument(
        "--prediction-dataset",
        default="regression_prediction",
        help="Name of dataset that will store raw regression outputs.",
    )
    parser.add_argument(
        "--augmented-features-key",
        default="ghost_track_vars_with_reg",
        help="Dataset key for feature matrix augmented with regression output.",
    )
    parser.add_argument(
        "--no-augment-features",
        action="store_true",
        help="Skip writing an augmented feature matrix.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run inference on (cpu/cuda). Default chooses automatically.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing datasets with the same name.",
    )
    return parser.parse_args()


def resolve_paths(args):
    run_dir = args.regression_run_dir
    args_path = args.args_path or os.path.join(run_dir, "args.json")
    model_path = args.model_path or os.path.join(run_dir, "best_model.pth")
    scaler_path = args.scaler_path or os.path.join(run_dir, "scaler.pkl")
    return args_path, model_path, scaler_path


def load_training_config(args_path):
    with open(args_path, "r") as f:
        return json.load(f)


def load_scaler_if_available(scaler_path):
    if scaler_path and os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            return pickle.load(f)
    return None


def build_model(training_cfg, input_dim, device, model_path):
    hidden_sizes = training_cfg.get("hidden_sizes", [50, 50, 50, 50, 50])
    dropout = training_cfg.get("dropout", 0.0)
    use_batch_norm = not training_cfg.get("no_batch_norm", False)

    model = MLPRegressor(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout,
        use_batch_norm=use_batch_norm,
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_inference(model, features, scaler, batch_size, device):
    mask = np.all(np.isfinite(features), axis=1)
    mask &= np.all(features != SENTINEL, axis=1)

    total = features.shape[0]
    valid = int(mask.sum())
    print(f"Valid feature rows for inference: {valid}/{total}")

    predictions = np.full(total, SENTINEL, dtype=np.float32)
    if valid == 0:
        return predictions, mask

    inputs = features[mask].astype(np.float32)
    if scaler is not None:
        inputs = scaler.transform(inputs).astype(np.float32)

    dataset = TensorDataset(torch.from_numpy(inputs))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            out = model(batch)
            preds.append(out.cpu().numpy())

    preds = np.concatenate(preds).astype(np.float32)
    predictions[mask] = preds
    return predictions, mask


def write_dataset(h5_file, name, data, overwrite):
    if name in h5_file:
        if not overwrite:
            raise RuntimeError(
                f"Dataset '{name}' already exists. Use --overwrite to replace it."
            )
        del h5_file[name]
    h5_file.create_dataset(name, data=data, compression="gzip")


def main():
    args = parse_args()
    output_h5 = args.output_h5 or args.input_h5

    if args.output_h5 and args.output_h5 != args.input_h5:
        shutil.copy2(args.input_h5, args.output_h5)
        print(f"Copied '{args.input_h5}' → '{args.output_h5}'")

    args_path, model_path, scaler_path = resolve_paths(args)
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"Could not find training args at '{args_path}'")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model weights at '{model_path}'")
    if args.scaler_path and not os.path.exists(args.scaler_path):
        raise FileNotFoundError(f"Scaler path '{args.scaler_path}' does not exist")
    training_cfg = load_training_config(args_path)
    scaler = load_scaler_if_available(scaler_path)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Running inference on device: {device}")

    with h5py.File(output_h5, "r") as f:
        if args.features_key not in f:
            raise KeyError(f"Dataset '{args.features_key}' not found in {output_h5}")
        features = f[args.features_key][:].astype(np.float32)

    model = build_model(training_cfg, features.shape[1], device, model_path)

    predictions, mask = run_inference(
        model, features, scaler, args.batch_size, device
    )

    n_valid = int(mask.sum())
    if n_valid > 0:
        valid_preds = predictions[mask]
        print(
            f"Stored predictions for {n_valid} rows "
            f"(min={valid_preds.min():.3f}, max={valid_preds.max():.3f})"
        )
    else:
        print("Warning: no rows received valid predictions.")

    with h5py.File(output_h5, "r+") as f:
        write_dataset(f, args.prediction_dataset, predictions, args.overwrite)
        print(
            f"Wrote predictions to '{args.prediction_dataset}' "
            f"in {output_h5} shape={predictions.shape}"
        )

        if not args.no_augment_features:
            augmented = np.concatenate(
                [features, predictions[:, None]], axis=1
            ).astype(np.float32)
            write_dataset(
                f, args.augmented_features_key, augmented, args.overwrite
            )
            print(
                f"Wrote augmented features to '{args.augmented_features_key}' "
                f"shape={augmented.shape}"
            )


if __name__ == "__main__":
    main()
