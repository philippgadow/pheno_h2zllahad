#!/usr/bin/env python3
"""
Train an XGBoost regressor on ghost track variables for mass prediction.

XGBoost often works better than neural networks for tabular data with a small
number of features. This script includes hyperparameter tuning and produces
the same diagnostic plots as the neural network version.

Usage:
    python train_xgboost_regression.py \
      --input-h5 signal_a05.h5 signal_a10.h5 signal_a15.h5 \
      --output-dir outputs/xgboost_run1 \
      --tune-hyperparameters
"""

import os
import argparse
import h5py
import json
import pickle
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint


# ============================================================================
# Data Loading
# ============================================================================

def load_h5_file(h5_path, features_key="ghost_track_vars", targets_key="targets"):
    """Load features and targets from a single HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        if features_key not in f:
            raise KeyError(f"{h5_path} missing '{features_key}' dataset")
        if targets_key not in f:
            raise KeyError(f"{h5_path} missing '{targets_key}' dataset")
        
        X = f[features_key][:]
        y = f[targets_key][:]
    
    y = np.squeeze(y)
    if y.ndim != 1:
        raise ValueError(f"{h5_path} targets must be 1D, got shape {y.shape}")
    
    return X.astype(np.float32), y.astype(np.float32)


def load_h5_multi(paths, features_key="ghost_track_vars", targets_key="targets", 
                  filter_nonfinite=True):
    """
    Load and concatenate data from multiple HDF5 files.
    Filters out rows with non-finite targets by default.
    """
    Xs, ys = [], []
    feat_dim = None
    
    for p in paths:
        X, y = load_h5_file(p, features_key, targets_key)
        
        if feat_dim is None:
            feat_dim = X.shape[1]
        elif X.shape[1] != feat_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {feat_dim}, got {X.shape[1]} in {p}"
            )
        
        Xs.append(X)
        ys.append(y)
        print(f"Loaded {len(y)} rows from {os.path.basename(p)}")
    
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    
    print(f"Total rows before filtering: {len(y)}")
    
    if filter_nonfinite:
        mask = np.isfinite(y)
        n_dropped = int((~mask).sum())
        if n_dropped > 0:
            print(f"Filtering {n_dropped} rows with non-finite targets")
            X = X[mask]
            y = y[mask]
            print(f"Total rows after filtering: {len(y)}")
    
    valid_features = np.all(np.isfinite(X), axis=1)
    n_invalid = int((~valid_features).sum())
    if n_invalid > 0:
        print(f"Filtering {n_invalid} rows with invalid features")
        X = X[valid_features]
        y = y[valid_features]
        print(f"Total rows after feature filtering: {len(y)}")
    
    return X, y


def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """Split data into train/val/test sets."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    val_fraction = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction, random_state=random_state, shuffle=True
    )
    
    print(f"Split sizes - Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================================
# Data Standardization
# ============================================================================

def fit_scaler(X, scaler_type="robust"):
    """Fit a scaler to the training data."""
    if scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "none":
        return None
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    scaler.fit(X)
    return scaler


def save_scaler(scaler, path):
    """Save scaler to pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)


# ============================================================================
# XGBoost Model Training
# ============================================================================

def train_xgboost(X_train, y_train, X_val, y_val, params=None, early_stopping_rounds=50):
    """
    Train an XGBoost regressor with early stopping.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data for early stopping
        params: Dictionary of XGBoost parameters
        early_stopping_rounds: Number of rounds for early stopping
    
    Returns:
        Trained XGBoost model
    """
    if params is None:
        # Default parameters optimized for regression
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'tree_method': 'hist',
            'n_jobs': -1
        }
    
    print("\nXGBoost parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    # Create DMatrix for faster training
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Training with early stopping
    evals = [(dtrain, 'train'), (dval, 'val')]
    evals_result = {}
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params.get('n_estimators', 1000),
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=10
    )
    
    return model, evals_result


def tune_hyperparameters(X_train, y_train, n_iter=50, cv=3, random_state=42):
    """
    Perform randomized hyperparameter search for XGBoost.
    
    Args:
        X_train, y_train: Training data
        n_iter: Number of parameter settings sampled
        cv: Number of cross-validation folds
        random_state: Random seed
    
    Returns:
        Best parameters found
    """
    print("\n" + "="*80)
    print("Starting hyperparameter tuning...")
    print("="*80)
    
    # Parameter distributions for random search
    param_dist = {
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'n_estimators': randint(100, 1000),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_weight': randint(1, 10),
        'gamma': uniform(0, 0.5),
        'reg_alpha': uniform(0, 1.0),
        'reg_lambda': uniform(0, 2.0)
    }
    
    # Base model
    base_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        random_state=random_state,
        n_jobs=-1
    )
    
    # Random search
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_squared_error',
        random_state=random_state,
        verbose=2,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    
    print("\nBest parameters found:")
    for k, v in random_search.best_params_.items():
        print(f"  {k}: {v}")
    print(f"\nBest CV score (neg MSE): {random_search.best_score_:.6f}")
    
    return random_search.best_params_


# ============================================================================
# Metrics and Evaluation
# ============================================================================

def compute_metrics(y_true, y_pred):
    """Compute regression metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    
    # MSLE (if all values are non-negative)
    if np.all(y_true >= 0) and np.all(y_pred >= 0):
        msle = float(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))
    else:
        msle = float('nan')
    
    return {"mae": mae, "rmse": rmse, "r2": r2, "msle": msle}


def predict(model, X):
    """Make predictions using XGBoost model."""
    dtest = xgb.DMatrix(X)
    return model.predict(dtest)


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_training_curve(evals_result, save_path):
    """Plot training and validation loss curves."""
    fig = plt.figure(figsize=(8, 8))
    
    train_metric = evals_result['train']['rmse']
    val_metric = evals_result['val']['rmse']
    
    plt.plot(train_metric, label='Train')
    plt.plot(val_metric, label='Test')
    plt.title('model loss')
    plt.ylabel('RMSE')
    plt.xlabel('iteration')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    fig.savefig(save_path)
    plt.close()


def plot_feature_importance(model, feature_names, save_path, top_n=None):
    """Plot feature importance."""
    importance = model.get_score(importance_type='gain')
    
    # Convert to sorted lists
    features = []
    scores = []
    for i in range(len(feature_names)):
        feat_key = f'f{i}'
        if feat_key in importance:
            features.append(feature_names[i])
            scores.append(importance[feat_key])
    
    # Sort by importance
    indices = np.argsort(scores)[::-1]
    if top_n is not None:
        indices = indices[:top_n]
    
    features = [features[i] for i in indices]
    scores = [scores[i] for i in indices]
    
    # Plot
    fig = plt.figure(figsize=(10, 6))
    plt.barh(range(len(features)), scores, align='center')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Feature Importance (Gain)')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close()


def plot_predictions_histogram(y_true, y_pred, save_path, title="Predictions", bins=100, range_tuple=(0, 5)):
    """Plot histogram of predicted values with ratio plot."""
    fig = plt.figure(figsize=(10, 10))
    
    # Main plot area (top 70%)
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    
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


def plot_predictions_scatter(y_true, y_pred, save_path, title="Predictions"):
    """Plot predicted vs true values as scatter plot."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, s=5, alpha=0.3)
    
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
    
    # Add statistics
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    plt.axvline(mean_res, color='blue', linestyle='--', linewidth=1, 
                label=f'Mean: {mean_res:.4f}')
    plt.text(0.02, 0.98, f'Std: {std_res:.4f}', 
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost regressor on ghost track variables"
    )
    
    # Data arguments
    parser.add_argument("--input-h5", nargs="+", required=True,
                        help="Input HDF5 files")
    parser.add_argument("--features-key", default="ghost_track_vars",
                        help="HDF5 dataset key for features")
    parser.add_argument("--targets-key", default="targets",
                        help="HDF5 dataset key for targets (truth mass)")
    
    # Output arguments
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for results")
    
    # Model arguments
    parser.add_argument("--max-depth", type=int, default=6,
                        help="Maximum tree depth")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                        help="Learning rate (eta)")
    parser.add_argument("--n-estimators", type=int, default=1000,
                        help="Number of boosting rounds")
    parser.add_argument("--subsample", type=float, default=0.8,
                        help="Subsample ratio")
    parser.add_argument("--colsample-bytree", type=float, default=0.8,
                        help="Column subsample ratio")
    parser.add_argument("--min-child-weight", type=int, default=3,
                        help="Minimum child weight")
    parser.add_argument("--gamma", type=float, default=0.0,
                        help="Minimum loss reduction for split")
    parser.add_argument("--reg-alpha", type=float, default=0.1,
                        help="L1 regularization")
    parser.add_argument("--reg-lambda", type=float, default=1.0,
                        help="L2 regularization")
    
    # Hyperparameter tuning
    parser.add_argument("--tune-hyperparameters", action="store_true",
                        help="Perform hyperparameter tuning")
    parser.add_argument("--tune-n-iter", type=int, default=50,
                        help="Number of iterations for hyperparameter tuning")
    parser.add_argument("--tune-cv", type=int, default=3,
                        help="Number of CV folds for tuning")
    
    # Data preprocessing
    parser.add_argument("--standardize", choices=["none", "standard", "robust"], default="none",
                        help="Standardization method (XGBoost typically doesn't need scaling)")
    parser.add_argument("--test-split", type=float, default=0.2,
                        help="Test set fraction")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation set fraction")
    
    # Training
    parser.add_argument("--early-stopping-rounds", type=int, default=50,
                        help="Early stopping patience")
    
    # Other
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.random_seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # ========================================================================
    # Load Data
    # ========================================================================
    print("\n" + "="*80)
    print("Loading data...")
    print("="*80)
    
    X, y = load_h5_multi(args.input_h5, args.features_key, args.targets_key)
    
    print(f"\nFeature shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target range: [{np.min(y):.3f}, {np.max(y):.3f}]")
    print(f"Target mean: {np.mean(y):.3f} ± {np.std(y):.3f}")
    
    # Feature names
    feature_names = [
        "nTracks", "deltaRLeadTrack", "leadTrackPtRatio",
        "angularity_2", "U1_0p7", "M2_0p3", "tau2"
    ]
    if X.shape[1] != len(feature_names):
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # ========================================================================
    # Split Data
    # ========================================================================
    print("\n" + "="*80)
    print("Splitting data...")
    print("="*80)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, 
        test_size=args.test_split,
        val_size=args.val_split,
        random_state=args.random_seed
    )
    
    # ========================================================================
    # Standardization (optional for XGBoost)
    # ========================================================================
    scaler = None
    if args.standardize != "none":
        print("\n" + "="*80)
        print(f"Standardizing features using {args.standardize} scaler...")
        print("="*80)
        print("Note: XGBoost typically doesn't require feature scaling")
        
        scaler = fit_scaler(X_train, args.standardize)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        scaler_path = os.path.join(args.output_dir, "scaler.pkl")
        save_scaler(scaler, scaler_path)
        print(f"Saved scaler to {scaler_path}")
    
    # ========================================================================
    # Hyperparameter Tuning (optional)
    # ========================================================================
    if args.tune_hyperparameters:
        best_params = tune_hyperparameters(
            X_train, y_train,
            n_iter=args.tune_n_iter,
            cv=args.tune_cv,
            random_state=args.random_seed
        )
        
        # Save tuned parameters
        with open(os.path.join(args.output_dir, "tuned_params.json"), "w") as f:
            json.dump(best_params, f, indent=2)
    else:
        # Use provided parameters
        best_params = {
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
            'n_estimators': args.n_estimators,
            'subsample': args.subsample,
            'colsample_bytree': args.colsample_bytree,
            'min_child_weight': args.min_child_weight,
            'gamma': args.gamma,
            'reg_alpha': args.reg_alpha,
            'reg_lambda': args.reg_lambda
        }
    
    # Add fixed parameters
    best_params.update({
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': args.random_seed,
        'tree_method': 'hist',
        'n_jobs': -1
    })
    
    # ========================================================================
    # Train Model
    # ========================================================================
    print("\n" + "="*80)
    print("Training XGBoost model...")
    print("="*80)
    
    model, evals_result = train_xgboost(
        X_train, y_train,
        X_val, y_val,
        params=best_params,
        early_stopping_rounds=args.early_stopping_rounds
    )
    
    # Save model
    model_path = os.path.join(args.output_dir, "xgboost_model.json")
    model.save_model(model_path)
    print(f"\nModel saved to {model_path}")
    
    # ========================================================================
    # Evaluate Model
    # ========================================================================
    print("\n" + "="*80)
    print("Evaluating model...")
    print("="*80)
    
    # Make predictions
    y_train_pred = predict(model, X_train)
    y_val_pred = predict(model, X_val)
    y_test_pred = predict(model, X_test)
    
    # Compute metrics
    train_metrics = compute_metrics(y_train, y_train_pred)
    val_metrics = compute_metrics(y_val, y_val_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)
    
    print("\nTraining Results:")
    print(f"  MAE:  {train_metrics['mae']:.6f}")
    print(f"  RMSE: {train_metrics['rmse']:.6f}")
    print(f"  R²:   {train_metrics['r2']:.4f}")
    if not np.isnan(train_metrics['msle']):
        print(f"  MSLE: {train_metrics['msle']:.6f}")
    
    print("\nValidation Results:")
    print(f"  MAE:  {val_metrics['mae']:.6f}")
    print(f"  RMSE: {val_metrics['rmse']:.6f}")
    print(f"  R²:   {val_metrics['r2']:.4f}")
    if not np.isnan(val_metrics['msle']):
        print(f"  MSLE: {val_metrics['msle']:.6f}")
    
    print("\nTest Results:")
    print(f"  MAE:  {test_metrics['mae']:.6f}")
    print(f"  RMSE: {test_metrics['rmse']:.6f}")
    print(f"  R²:   {test_metrics['r2']:.4f}")
    if not np.isnan(test_metrics['msle']):
        print(f"  MSLE: {test_metrics['msle']:.6f}")
    
    # ========================================================================
    # Create Plots
    # ========================================================================
    print("\n" + "="*80)
    print("Creating plots...")
    print("="*80)
    
    # Training curve
    plot_training_curve(evals_result, 
                       os.path.join(args.output_dir, "Training_curve.png"))
    
    # Feature importance
    plot_feature_importance(model, feature_names,
                           os.path.join(args.output_dir, "feature_importance.png"))
    
    # Histogram-style prediction plots
    plot_predictions_histogram(y_val, y_val_pred,
                               os.path.join(args.output_dir, "Reg_Val.png"),
                               title="Validation Set - Full")
    
    plot_predictions_histogram(y_test, y_test_pred,
                               os.path.join(args.output_dir, "Reg_Test.png"),
                               title="Test Set - Full")
    
    # Scatter plots
    plot_predictions_scatter(y_val, y_val_pred,
                            os.path.join(args.output_dir, "predictions_val_scatter.png"),
                            title="Validation Set Predictions")
    
    plot_predictions_scatter(y_test, y_test_pred,
                            os.path.join(args.output_dir, "predictions_test_scatter.png"),
                            title="Test Set Predictions")
    
    # Residual plots
    plot_residuals(y_val, y_val_pred,
                  os.path.join(args.output_dir, "residuals_val.png"),
                  title="Validation Set Residuals")
    
    plot_residuals(y_test, y_test_pred,
                  os.path.join(args.output_dir, "residuals_test.png"),
                  title="Test Set Residuals")
    
    # ========================================================================
    # Save Report
    # ========================================================================
    report = {
        "train": {
            "mae": float(train_metrics['mae']),
            "rmse": float(train_metrics['rmse']),
            "r2": float(train_metrics['r2']),
            "msle": float(train_metrics['msle'])
        },
        "validation": {
            "mae": float(val_metrics['mae']),
            "rmse": float(val_metrics['rmse']),
            "r2": float(val_metrics['r2']),
            "msle": float(val_metrics['msle'])
        },
        "test": {
            "mae": float(test_metrics['mae']),
            "rmse": float(test_metrics['rmse']),
            "r2": float(test_metrics['r2']),
            "msle": float(test_metrics['msle'])
        },
        "model": {
            "type": "XGBoost",
            "best_iteration": int(model.best_iteration),
            "n_features": X_train.shape[1],
            "parameters": best_params
        },
        "data": {
            "n_train": len(y_train),
            "n_val": len(y_val),
            "n_test": len(y_test),
            "n_features": X_train.shape[1],
            "feature_names": feature_names,
            "input_files": [os.path.basename(f) for f in args.input_h5]
        }
    }
    
    report_path = os.path.join(args.output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to {report_path}")
    print(f"Model saved to {model_path}")
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Best iteration: {model.best_iteration}")
    print(f"Test RMSE: {test_metrics['rmse']:.6f}")
    print(f"Test MAE:  {test_metrics['mae']:.6f}")
    print(f"Test R²:   {test_metrics['r2']:.4f}")
    
    # Check if model is performing well
    if test_metrics['r2'] > 0.5:
        print("\n✓ Model shows good predictive performance (R² > 0.5)")
    else:
        print("\n⚠ Warning: Model shows poor predictive performance (R² < 0.5)")
        print("  Consider:")
        print("  - Using --tune-hyperparameters for automatic optimization")
        print("  - Checking if the features are informative for the target")
        print("  - Verifying data quality and preprocessing")


if __name__ == "__main__":
    main()