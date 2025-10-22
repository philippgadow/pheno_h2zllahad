#!/usr/bin/env python3
"""
Plot ghost track variables split by truth mass categories.

This script creates HEP style visualizations using mplhep with proper histograms.
Example:
    python plot_features.py \
      --input-h5 signal_a05.h5 signal_a10.h5 signal_a15.h5 \
      --output-dir plots/features \
      --create-pdf
"""

import os
import re
import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm

# Use ROOT style
plt.style.use(hep.style.CMS)

# =============================================================================
# Utilities
# =============================================================================

def safe_name(s: str) -> str:
    """Return a filesystem safe slug for titles or feature names."""
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^\w\-\.\+]", "", s)
    return s[:120]  # keep filenames reasonable

# =============================================================================
# Data Loading
# =============================================================================

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


def load_h5_multi(paths, features_key="ghost_track_vars", targets_key="targets"):
    """Load and concatenate data from multiple HDF5 files."""
    Xs, ys = [], []
    for p in paths:
        X, y = load_h5_file(p, features_key, targets_key)
        Xs.append(X)
        ys.append(y)
        print(f"Loaded {len(y)} rows from {os.path.basename(p)}")

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)

    print(f"Total rows: {len(y)}")

    # Filter non finite rows
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    n_dropped = int((~mask).sum())
    if n_dropped > 0:
        print(f"Filtering {n_dropped} rows with non finite values")
        X = X[mask]
        y = y[mask]
        print(f"Total rows after filtering: {len(y)}")

    return X, y

# =============================================================================
# Mass Categories
# =============================================================================

def categorize_by_mass(masses):
    """
    Categorize jets by truth mass.

    0: < 1 GeV
    1: 1 to 2 GeV
    2: 2 to 3 GeV
    3: 3 to 4 GeV
    4: 4 to 100 GeV
    5: >= 100 GeV
    """
    categories = np.zeros(len(masses), dtype=int)
    categories[masses < 1.0] = 0
    categories[(masses >= 1.0) & (masses < 2.0)] = 1
    categories[(masses >= 2.0) & (masses < 3.0)] = 2
    categories[(masses >= 3.0) & (masses < 4.0)] = 3
    categories[(masses >= 4.0) & (masses < 100.0)] = 4
    categories[masses >= 100.0] = 5
    return categories


def get_category_names(tex=True):
    """Return names for each mass category."""
    if tex:
        return [
            r"$m<1$ GeV",
            r"$1\leq m<2$ GeV",
            r"$2\leq m<3$ GeV",
            r"$3\leq m<4$ GeV",
            r"$4\leq m<100$ GeV",
            r"$m\geq 100$ GeV",
        ]
    return ["< 1 GeV", "1 to 2 GeV", "2 to 3 GeV", "3 to 4 GeV", "4 to 100 GeV", ">= 100 GeV"]


def get_category_colors():
    """Return colors for each mass category."""
    # A readable, colorblind friendly palette
    return ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

# =============================================================================
# Plotting
# =============================================================================

# def _range_and_bins(arr, qlow=0.5, qhigh=99.5, nbins=50):
#     valid = np.isfinite(arr)
#     if not np.any(valid):
#         return None, None, valid
#     p1, p99 = np.percentile(arr[valid], [qlow, qhigh])
#     w = p99 - p1
#     lo, hi = p1 - 0.05 * w, p99 + 0.05 * w
#     bins = np.linspace(lo, hi, nbins + 1)
#     return (lo, hi), bins, valid

def _range_and_bins(arr, qlow=0.5, qhigh=99.5, nbins=50):
    # keep only finite and strictly positive entries
    valid = np.isfinite(arr) & (arr > 0)
    if not np.any(valid):
        return None, None, valid
    p1, p99 = np.percentile(arr[valid], [qlow, qhigh])
    w = p99 - p1
    lo, hi = p1 - 0.05 * w, p99 + 0.05 * w
    # clamp the lower edge to be strictly positive
    lo = max(lo, np.nextafter(0.0, 1.0))
    bins = np.linspace(lo, hi, nbins + 1)
    return (lo, hi), bins, valid



# def plot_feature_distributions(X, masses, feature_names, save_dir):
#     """One figure per feature with category split step histograms."""
#     os.makedirs(save_dir, exist_ok=True)
#     categories = categorize_by_mass(masses)
#     cat_names = get_category_names()
#     colors = get_category_colors()

#     for i, feat_name in enumerate(feature_names):
#         fig, ax = plt.subplots(figsize=(10, 8))
#         feat = X[:, i]
#         rng, bins, valid = _range_and_bins(feat, 0.5, 99.5, 50)

#         if rng is None:
#             plt.close(fig)
#             continue

#         for cat_idx in range(6):
#             mask = (categories == cat_idx) & valid
#             n = int(mask.sum())
#             if n == 0:
#                 continue
#             hist, edges = np.histogram(feat[mask], bins=bins)
#             hep.histplot((hist, edges),
#                          label=f"{cat_names[cat_idx]} (n={n})",
#                          color=colors[cat_idx],
#                          histtype='step',
#                          linewidth=2,
#                          ax=ax)

#         ax.set_xlabel(feat_name, fontsize=14)
#         ax.set_ylabel('Jets', fontsize=14)
#         ax.set_xlim(rng)
#         ax.legend(loc='best', fontsize=11, frameon=True)
#         plt.tight_layout()
#         out = os.path.join(save_dir, f"feature_{i:02d}_{safe_name(feat_name)}.png")
#         fig.savefig(out, dpi=150)
#         plt.close(fig)
#         print(f"  Saved: {os.path.basename(out)}")

def plot_feature_distributions(X, masses, feature_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    categories = categorize_by_mass(masses)
    cat_names = get_category_names()
    colors = get_category_colors()

    for i, feat_name in enumerate(feature_names):
        fig, ax = plt.subplots(figsize=(10, 8))
        feat = X[:, i]
        rng, bins, valid = _range_and_bins(feat, 0.5, 99.5, 50)
        if rng is None:
            plt.close(fig)
            continue

        for cat_idx in range(6):
            mask = (categories == cat_idx) & valid
            n = int(mask.sum())
            if n == 0:
                continue
            hist, edges = np.histogram(feat[mask], bins=bins)
            hep.histplot((hist, edges),
                         label=f"{cat_names[cat_idx]} (n={n})",
                         color=colors[cat_idx],
                         histtype='step',
                         linewidth=2,
                         ax=ax)

        ax.set_xlabel(feat_name, fontsize=14)
        ax.set_ylabel('Jets', fontsize=14)
        ax.set_xlim(rng)
        ax.legend(loc='best', fontsize=11, frameon=True)
        plt.tight_layout()
        out = os.path.join(save_dir, f"feature_{i:02d}_{safe_name(feat_name)}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {os.path.basename(out)}")


# def plot_feature_distributions_grid(X, masses, feature_names, save_path):
#     """All features in a single grid."""
#     categories = categorize_by_mass(masses)
#     cat_names = get_category_names()
#     colors = get_category_colors()

#     n_features = X.shape[1]
#     n_cols = 3
#     n_rows = int(np.ceil(n_features / n_cols))

#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
#     axes = axes.flatten() if n_features > 1 else [axes]

#     for i, feat_name in enumerate(feature_names):
#         ax = axes[i]
#         feat = X[:, i]
#         rng, bins, valid = _range_and_bins(feat, 0.5, 99.5, 40)
#         if rng is None:
#             ax.axis('off')
#             continue

#         for cat_idx in range(6):
#             mask = (categories == cat_idx) & valid
#             if not np.any(mask):
#                 continue
#             hist, edges = np.histogram(feat[mask], bins=bins)
#             hep.histplot((hist, edges),
#                          label=f"{cat_names[cat_idx]}",
#                          color=colors[cat_idx],
#                          histtype='step',
#                          linewidth=2,
#                          ax=ax)

#         ax.set_xlabel(feat_name, fontsize=10)
#         ax.set_ylabel('Jets', fontsize=10)
#         ax.set_xlim(rng)
#         if i == 0:
#             ax.legend(loc='best', fontsize=8, frameon=True)

#     for j in range(i + 1, len(axes)):
#         axes[j].axis('off')

#     plt.suptitle('Ghost track variables by mass category', fontsize=16, y=0.995)
#     plt.tight_layout()
#     fig.savefig(save_path, dpi=150, bbox_inches='tight')
#     plt.close(fig)
#     print(f"  Saved: {os.path.basename(save_path)}")

def plot_feature_distributions_grid(X, masses, feature_names, save_path):
    categories = categorize_by_mass(masses)
    cat_names = get_category_names()
    colors = get_category_colors()

    n_features = X.shape[1]
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for i, feat_name in enumerate(feature_names):
        ax = axes[i]
        feat = X[:, i]
        rng, bins, valid = _range_and_bins(feat, 0.5, 99.5, 40)
        if rng is None:
            ax.axis('off')
            continue

        for cat_idx in range(6):
            mask = (categories == cat_idx) & valid
            if not np.any(mask):
                continue
            hist, edges = np.histogram(feat[mask], bins=bins)
            hep.histplot((hist, edges),
                         label=f"{cat_names[cat_idx]}",
                         color=colors[cat_idx],
                         histtype='step',
                         linewidth=2,
                         ax=ax)

        ax.set_xlabel(feat_name, fontsize=10)
        ax.set_ylabel('Jets', fontsize=10)
        ax.set_xlim(rng)
        if i == 0:
            ax.legend(loc='best', fontsize=8, frameon=True)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Ghost track variables by mass category', fontsize=16, y=0.995)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.basename(save_path)}")


# def plot_feature_vs_mass(X, masses, feature_names, save_dir):
#     """2D hist per feature vs truth mass."""
#     os.makedirs(save_dir, exist_ok=True)
#     for i, feat_name in enumerate(feature_names):
#         fig, ax = plt.subplots(figsize=(10, 8))
#         feat = X[:, i]
#         valid = np.isfinite(feat) & np.isfinite(masses)
#         if not np.any(valid):
#             plt.close(fig)
#             continue

#         feat_valid = feat[valid]
#         mass_valid = masses[valid]

#         f1, f99 = np.percentile(feat_valid, [0.5, 99.5])
#         m1, m99 = np.percentile(mass_valid, [0.5, 99.5])

#         h = ax.hist2d(mass_valid, feat_valid,
#                       bins=[80, 80],
#                       range=[[m1, m99], [f1, f99]],
#                       cmap='viridis',
#                       norm=LogNorm(),
#                       cmin=1)

#         cbar = plt.colorbar(h[3], ax=ax)
#         cbar.set_label('Jets', fontsize=12)

#         ax.set_xlabel('Truth mass [GeV]', fontsize=14)
#         ax.set_ylabel(feat_name, fontsize=14)

#         for boundary in [1, 2, 3, 4, 100]:
#             if m1 < boundary < m99:
#                 ax.axvline(boundary, color='red', linestyle='--', linewidth=1.2, alpha=0.7)

#         plt.tight_layout()
#         out = os.path.join(save_dir, f"feature_vs_mass_{i:02d}_{safe_name(feat_name)}.png")
#         fig.savefig(out, dpi=150)
#         plt.close(fig)
#         print(f"  Saved: {os.path.basename(out)}")

from matplotlib.colors import LogNorm  # keep this import

def plot_feature_vs_mass(X, masses, feature_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i, feat_name in enumerate(feature_names):
        fig, ax = plt.subplots(figsize=(10, 8))
        feat = X[:, i]
        # require positive feature and positive mass
        valid = np.isfinite(feat) & np.isfinite(masses) & (feat > 0) & (masses > 0)
        if not np.any(valid):
            plt.close(fig)
            continue

        feat_valid = feat[valid]
        mass_valid = masses[valid]

        f1, f99 = np.percentile(feat_valid, [0.5, 99.5])
        m1, m99 = np.percentile(mass_valid, [0.5, 99.5])
        # clamp to positive
        f1 = max(f1, np.nextafter(0.0, 1.0))
        m1 = max(m1, np.nextafter(0.0, 1.0))

        h = ax.hist2d(mass_valid, feat_valid,
                      bins=[80, 80],
                      range=[[m1, m99], [f1, f99]],
                      cmap='viridis',
                      norm=LogNorm(),
                      cmin=1)

        cbar = plt.colorbar(h[3], ax=ax)
        cbar.set_label('Jets', fontsize=12)

        ax.set_xlabel('Truth mass [GeV]', fontsize=14)
        ax.set_ylabel(feat_name, fontsize=14)

        for boundary in [1, 2, 3, 4, 100]:
            if m1 < boundary < m99:
                ax.axvline(boundary, color='red', linestyle='--', linewidth=1.2, alpha=0.7)

        plt.tight_layout()
        out = os.path.join(save_dir, f"feature_vs_mass_{i:02d}_{safe_name(feat_name)}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {os.path.basename(out)}")


def plot_correlation_matrices(X, masses, feature_names, save_dir):
    """Correlation matrices per mass category."""
    categories = categorize_by_mass(masses)
    cat_names = get_category_names(tex=False)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    short_names = [fn.replace('GhostTrackVars_', '').replace('_', ' ') for fn in feature_names]

    for cat_idx in range(6):
        ax = axes[cat_idx]
        mask = categories == cat_idx
        n = int(mask.sum())

        if n > 10:
            X_cat = X[mask]
            corr = np.corrcoef(X_cat.T)
            im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Correlation', fontsize=10)

            ax.set_xticks(range(len(short_names)))
            ax.set_yticks(range(len(short_names)))
            ax.set_xticklabels(short_names, fontsize=8, rotation=45, ha='right')
            ax.set_yticklabels(short_names, fontsize=8)
            ax.set_title(f"{cat_names[cat_idx]} (n={n})", fontsize=12)

        else:
            ax.text(0.5, 0.5, f"Not enough data\n(n={n})",
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.axis('off')

    plt.suptitle('Feature correlations by mass category', fontsize=16)
    plt.tight_layout()
    out = os.path.join(save_dir, "correlation_matrices.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out)}")


# def plot_mass_distribution(masses, save_path):
#     """Distribution of truth masses, linear and log count axes."""
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
#     valid = np.isfinite(masses)
#     masses_valid = masses[valid]
#     if masses_valid.size == 0:
#         plt.close(fig)
#         return

#     p1, p99 = np.percentile(masses_valid, [0.1, 99.9])
#     bins = np.linspace(p1, p99, 81)

#     # Linear
#     hist, edges = np.histogram(masses_valid, bins=bins)
#     hep.histplot((hist, edges), histtype='fill', color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.0, ax=ax1)
#     ax1.set_xlabel('Truth mass [GeV]', fontsize=12)
#     ax1.set_ylabel('Jets', fontsize=12)
#     for b in [1, 2, 3, 4, 100]:
#         if p1 < b < p99:
#             ax1.axvline(b, color='red', linestyle='--', linewidth=1.2, alpha=0.8)

#     stats = f"Total: {len(masses_valid):,}\nMean: {np.mean(masses_valid):.2f} GeV\nMedian: {np.median(masses_valid):.2f} GeV\nStd: {np.std(masses_valid):.2f} GeV"
#     ax1.text(0.98, 0.97, stats, transform=ax1.transAxes, va='top', ha='right',
#              bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'),
#              fontsize=10, family='monospace')

#     # Log
#     hep.histplot((hist, edges), histtype='fill', color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.0, ax=ax2)
#     ax2.set_xlabel('Truth mass [GeV]', fontsize=12)
#     ax2.set_ylabel('Jets', fontsize=12)
#     ax2.set_yscale('log')
#     for b in [1, 2, 3, 4, 100]:
#         if p1 < b < p99:
#             ax2.axvline(b, color='red', linestyle='--', linewidth=1.2, alpha=0.8)

#     plt.tight_layout()
#     fig.savefig(save_path, dpi=150)
#     plt.close(fig)
#     print(f"  Saved: {os.path.basename(save_path)}")

def plot_mass_distribution(masses, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    valid = np.isfinite(masses) & (masses > 0)
    masses_valid = masses[valid]
    if masses_valid.size == 0:
        plt.close(fig)
        return

    p1, p99 = np.percentile(masses_valid, [0.1, 99.9])
    p1 = max(p1, np.nextafter(0.0, 1.0))
    bins = np.linspace(p1, p99, 81)

    hist, edges = np.histogram(masses_valid, bins=bins)
    hep.histplot((hist, edges), histtype='fill', color='steelblue', alpha=0.7,
                 edgecolor='black', linewidth=1.0, ax=ax1)
    ax1.set_xlabel('Truth mass [GeV]', fontsize=12)
    ax1.set_ylabel('Jets', fontsize=12)
    for b in [1, 2, 3, 4, 100]:
        if p1 < b < p99:
            ax1.axvline(b, color='red', linestyle='--', linewidth=1.2, alpha=0.8)

    stats = (
        f"Total: {len(masses_valid):,}\n"
        f"Mean: {np.mean(masses_valid):.2f} GeV\n"
        f"Median: {np.median(masses_valid):.2f} GeV\n"
        f"Std: {np.std(masses_valid):.2f} GeV"
    )
    ax1.text(0.98, 0.97, stats, transform=ax1.transAxes, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'),
             fontsize=10, family='monospace')

    hep.histplot((hist, edges), histtype='fill', color='steelblue', alpha=0.7,
                 edgecolor='black', linewidth=1.0, ax=ax2)
    ax2.set_xlabel('Truth mass [GeV]', fontsize=12)
    ax2.set_ylabel('Jets', fontsize=12)
    ax2.set_yscale('log')
    for b in [1, 2, 3, 4, 100]:
        if p1 < b < p99:
            ax2.axvline(b, color='red', linestyle='--', linewidth=1.2, alpha=0.8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {os.path.basename(save_path)}")


def plot_category_breakdown(masses, save_path):
    """Bar chart of counts per category."""
    categories = categorize_by_mass(masses)
    cat_names = get_category_names(tex=False)
    colors = get_category_colors()

    fig, ax = plt.subplots(figsize=(12, 7))
    counts = [int((categories == i).sum()) for i in range(6)]
    bars = ax.bar(range(6), counts, color=colors, alpha=0.85, edgecolor='black', linewidth=1.0)

    for i, bar in enumerate(bars):
        h = bar.get_height()
        pct = 100.0 * counts[i] / len(masses) if len(masses) else 0.0
        ax.text(bar.get_x() + bar.get_width() / 2.0, h, f'{counts[i]:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, weight='bold')

    ax.set_xticks(range(6))
    ax.set_xticklabels(cat_names, rotation=15, ha='right')
    ax.set_ylabel('Number of jets', fontsize=12)
    ax.set_title('Jet distribution by mass category', fontsize=14)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {os.path.basename(save_path)}")


def plot_feature_comparison_normalized(X, masses, feature_names, save_dir):
    """Normalized distributions to compare shapes across categories."""
    os.makedirs(save_dir, exist_ok=True)
    categories = categorize_by_mass(masses)
    cat_names = get_category_names()
    colors = get_category_colors()

    for i, feat_name in enumerate(feature_names):
        feat = X[:, i]
        rng, bins, valid = _range_and_bins(feat, 0.5, 99.5, 50)
        if rng is None:
            continue

        fig, ax = plt.subplots(figsize=(10, 8))
        for cat_idx in range(6):
            mask = (categories == cat_idx) & valid
            n = int(mask.sum())
            if n < 10:
                continue
            hist, edges = np.histogram(feat[mask], bins=bins)
            if hist.sum() > 0:
                hist = hist / hist.sum()
            hep.histplot((hist, edges),
                         label=f"{cat_names[cat_idx]} (n={n})",
                         color=colors[cat_idx],
                         histtype='step',
                         linewidth=2,
                         ax=ax)

        ax.set_xlabel(feat_name, fontsize=14)
        ax.set_ylabel('Normalized jets', fontsize=14)
        ax.set_xlim(rng)
        ax.legend(loc='best', fontsize=11, frameon=True)
        plt.tight_layout()
        out = os.path.join(save_dir, f"feature_normalized_{i:02d}_{safe_name(feat_name)}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {os.path.basename(out)}")

# =============================================================================
# Summary PDF
# =============================================================================

def create_summary_pdf(save_dir):
    """Create a single PDF with all PNGs in the directory."""
    pdf_path = os.path.join(save_dir, "feature_plots_summary.pdf")
    pngs = sorted([f for f in os.listdir(save_dir) if f.lower().endswith(".png")])
    if not pngs:
        print("  No PNG files found to create PDF")
        return

    with PdfPages(pdf_path) as pdf:
        for png in pngs:
            img = plt.imread(os.path.join(save_dir, png))
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            ax.imshow(img)
            ax.axis('off')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"  Saved: feature_plots_summary.pdf, contains {len(pngs)} pages")

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot ghost track variables split by mass categories"
    )
    # Data
    parser.add_argument("--input-h5", nargs="+", required=True, help="Input HDF5 files")
    parser.add_argument("--features-key", default="ghost_track_vars", help="HDF5 key for features")
    parser.add_argument("--targets-key", default="targets", help="HDF5 key for targets, truth mass")

    # Output
    parser.add_argument("--output-dir", required=True, help="Output directory for plots")

    # Options
    parser.add_argument("--no-individual-plots", action="store_true", help="Skip individual feature plots")
    parser.add_argument("--no-2d-plots", action="store_true", help="Skip 2D histograms")
    parser.add_argument("--no-normalized", action="store_true", help="Skip normalized distributions")
    parser.add_argument("--create-pdf", action="store_true", help="Create a single PDF with all PNGs")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Default feature names, replaced if length mismatches
    feature_names = [
        "nTracks",
        "deltaRLeadTrack",
        "leadTrackPtRatio",
        "angularity_2",
        "U1_0p7",
        "M2_0p3",
        "tau2",
    ]

    print("\n" + "=" * 80)
    print("Loading data...")
    print("=" * 80)
    X, y = load_h5_multi(args.input_h5, args.features_key, args.targets_key)
    print(f"\nFeature shape: {X.shape}")
    print(f"Target shape:  {y.shape}")
    print(f"Target range:  [{np.min(y):.3f}, {np.max(y):.3f}]")

    if X.shape[1] != len(feature_names):
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    categories = categorize_by_mass(y)
    cat_names = get_category_names(tex=False)
    print("\nJets per mass category:")
    for i, name in enumerate(cat_names):
        n = int((categories == i).sum())
        pct = 100.0 * n / len(y)
        print(f"  {name:12s}: {n:8d} jets ({pct:5.2f}%)")

    print("\n" + "=" * 80)
    print("Creating plots...")
    print("=" * 80)

    print("\n1. Mass distribution:")
    plot_mass_distribution(y, os.path.join(args.output_dir, "mass_distribution.png"))

    print("\n2. Category breakdown:")
    plot_category_breakdown(y, os.path.join(args.output_dir, "category_breakdown.png"))

    print("\n3. Feature distributions, grid:")
    plot_feature_distributions_grid(X, y, feature_names, os.path.join(args.output_dir, "features_grid.png"))

    if not args.no_individual_plots:
        print("\n4. Individual feature distributions:")
        plot_feature_distributions(X, y, feature_names, args.output_dir)
    else:
        print("\n4. Skipping individual feature distributions")

    if not args.no_normalized:
        print("\n5. Normalized feature distributions:")
        plot_feature_comparison_normalized(X, y, feature_names, args.output_dir)
    else:
        print("\n5. Skipping normalized feature distributions")

    if not args.no_2d_plots:
        print("\n6. Feature vs mass 2D histograms:")
        plot_feature_vs_mass(X, y, feature_names, args.output_dir)
    else:
        print("\n6. Skipping 2D histograms")

    print("\n7. Correlation matrices:")
    plot_correlation_matrices(X, y, feature_names, args.output_dir)

    if args.create_pdf:
        print("\n8. Creating summary PDF:")
        create_summary_pdf(args.output_dir)

    print("\n" + "=" * 80)
    print("Plotting complete!")
    print("=" * 80)
    print(f"\nAll plots saved to: {args.output_dir}\n")


if __name__ == "__main__":
    main()
