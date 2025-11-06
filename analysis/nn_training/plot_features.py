#!/usr/bin/env python3
"""Visualise input features with explicit signal/background separation."""

import os
import argparse
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.backends.backend_pdf import PdfPages


# HEP plot styling
plt.style.use(hep.style.CMS)

AUX_SENTINEL = -999999.0
COLOR_BACKGROUND = "#4c72b0"
COLOR_SIGNAL = "#dd8452"


def safe_name(s: str) -> str:
    """Return a filesystem friendly slug."""
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in s).strip("_")


def _decode_bytes_array(arr: Optional[np.ndarray]) -> Optional[List[str]]:
    if arr is None:
        return None
    out: List[str] = []
    for item in arr:
        if isinstance(item, (bytes, np.bytes_)):
            out.append(item.decode("utf-8", errors="ignore"))
        else:
            out.append(str(item))
    return out


def load_h5_file(
    h5_path: str,
    features_key: str = "ghost_track_vars",
    targets_key: str = "targets",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, Dict[str, object]]]:
    """Load features, regression targets, and classification labels from a single HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        if features_key not in f or targets_key not in f:
            raise KeyError(f"{h5_path} missing required datasets '{features_key}' or '{targets_key}'")

        X = f[features_key][:].astype(np.float32)
        y = np.squeeze(f[targets_key][:]).astype(np.float32)
        if y.ndim != 1:
            raise ValueError(f"{h5_path}: '{targets_key}' must be 1D, got shape {y.shape}")

        if "is_signal" not in f or "signal_class" not in f:
            raise KeyError(f"{h5_path} missing 'is_signal' or 'signal_class' datasets")

        is_signal = f["is_signal"][:].astype(np.int8)
        signal_class = f["signal_class"][:].astype(np.int64)
        if is_signal.shape != (len(y),) or signal_class.shape != (len(y),):
            raise ValueError(f"{h5_path}: label arrays must match targets length {len(y)}")

        class_entries: Dict[int, Dict[str, object]] = {}
        if "signal_class_ids" in f:
            ids = f["signal_class_ids"][:].astype(np.int64)
            names = _decode_bytes_array(f.get("signal_class_names"))
            keys = _decode_bytes_array(f.get("signal_class_keys"))
            pids = f.get("signal_class_truth_pid")
            masses = f.get("signal_class_mass_GeV")
            sources = _decode_bytes_array(f.get("signal_class_source_file"))

            for idx, cid in enumerate(ids):
                entry = {
                    "id": int(cid),
                    "name": names[idx] if names and idx < len(names) else f"signal_{int(cid)}",
                    "key": keys[idx] if keys and idx < len(keys) else "",
                    "pid": int(pids[idx]) if pids is not None else -1,
                    "mass": float(masses[idx]) if masses is not None else float("nan"),
                    "source": sources[idx] if sources and idx < len(sources) else "",
                }
                class_entries[int(cid)] = entry

    return X, y, is_signal, signal_class, class_entries


def load_h5_multi(
    paths: List[str],
    features_key: str = "ghost_track_vars",
    targets_key: str = "targets",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Load and concatenate multiple HDF5 files, ensuring consistent metadata."""
    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    sigs: List[np.ndarray] = []
    classes: List[np.ndarray] = []
    class_meta_union: Dict[int, Dict[str, object]] = {}

    for p in paths:
        X, y, is_signal, signal_class, info = load_h5_file(p, features_key, targets_key)
        print(f"Loaded {len(y):6d} rows from {os.path.basename(p)}")
        Xs.append(X)
        ys.append(y)
        sigs.append(is_signal)
        classes.append(signal_class)
        if info:
            for cid, entry in info.items():
                if cid not in class_meta_union:
                    class_meta_union[cid] = entry
                else:
                    existing = class_meta_union[cid]
                    # ensure consistency for duplicate ids
                    for key in ("name", "key", "pid", "mass", "source"):
                        if key in entry and key in existing:
                            if existing[key] != entry[key]:
                                print(f"Warning: class id {cid} has conflicting {key} between files; keeping first")

    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    sig_all = np.concatenate(sigs, axis=0)
    cls_all = np.concatenate(classes, axis=0)

    # Filter out rows with completely invalid features or targets
    finite_mask = np.isfinite(y_all) | (sig_all == 0)
    finite_mask &= np.all(np.isfinite(X_all), axis=1)
    n_filtered = int((~finite_mask).sum())
    if n_filtered > 0:
        print(f"Filtering {n_filtered} jets with non-finite values")
        X_all = X_all[finite_mask]
        y_all = y_all[finite_mask]
        sig_all = sig_all[finite_mask]
        cls_all = cls_all[finite_mask]

    return X_all, y_all, sig_all, cls_all, class_meta_union


def _compute_bins(
    arrays: List[np.ndarray],
    nbins: int = 60,
    feature_name: Optional[str] = None
) -> Optional[Tuple[Tuple[float, float], np.ndarray]]:
    vals = []
    for a in arrays:
        if a.size == 0:
            continue
        finite = np.isfinite(a)
        if np.any(finite):
            filtered = a[finite]
            filtered = filtered[filtered != AUX_SENTINEL]
            filtered = filtered[~np.isclose(filtered, -1.0)]
            if filtered.size > 0:
                vals.append(filtered)
    if not vals:
        return None
    combined = np.concatenate(vals)
    if combined.size == 0:
        return None

    if feature_name == "nTracks":
        lo, hi = 0.0, 20.0
        bins = np.linspace(lo, hi, 21)
        return (lo, hi), bins

    if feature_name == "angularity_2":
        lo, hi = 0.0, 50.0
        bins = np.linspace(lo, hi, 26)
        return (lo, hi), bins

    lo_q, hi_q = np.percentile(combined, [0.5, 99.5])
    if not np.isfinite(lo_q) or not np.isfinite(hi_q) or hi_q <= lo_q:
        return None
    span = hi_q - lo_q
    lo = lo_q - 0.05 * span
    hi = hi_q + 0.05 * span
    bins = np.linspace(lo, hi, nbins + 1)
    return (lo, hi), bins


def _build_class_labels(meta: Dict[int, Dict[str, object]]) -> Dict[int, Dict[str, object]]:
    if not meta:
        return {}
    labels: Dict[int, Dict[str, object]] = {}
    for cid, entry in meta.items():
        labels[cid] = {
            "id": cid,
            "name": entry.get("name", f"signal_{cid}"),
            "pid": entry.get("pid", -1),
            "mass": entry.get("mass", float("nan")),
            "source": entry.get("source", ""),
            "key": entry.get("key", ""),
        }
    return labels


def plot_signal_breakdown(
    is_signal: np.ndarray,
    signal_class: np.ndarray,
    class_labels: Dict[int, Dict[str, object]],
    save_path: str,
):
    background = int((is_signal == 0).sum())
    unique_classes = sorted(cid for cid in np.unique(signal_class) if cid != -1)
    counts = [int((signal_class == cid).sum()) for cid in unique_classes]
    labels = [class_labels.get(cid, {}).get("name", f"signal {cid}") for cid in unique_classes]

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(labels)), 6))
    bars = ax.bar(["background"], [background], color=COLOR_BACKGROUND, alpha=0.85, edgecolor="black")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{background:,}",
                ha="center", va="bottom", fontsize=10, weight="bold")

    x_positions = np.arange(len(labels)) + 1
    palette = plt.cm.tab20(np.linspace(0, 1, max(3, len(labels))))
    for idx, (cid, count, name) in enumerate(zip(unique_classes, counts, labels)):
        bar = ax.bar([idx + 1], [count], color=palette[idx], alpha=0.9, edgecolor="black")
        ax.text(bar.patches[0].get_x() + bar.patches[0].get_width() / 2,
                bar.patches[0].get_height(), f"{count:,}\n({count/(count+background):.2%})",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks([0] + list(x_positions))
    ax.set_xticklabels(["background"] + labels, rotation=20, ha="right")
    ax.set_ylabel("Jets")
    ax.set_title("Jet counts by class")
    ax.set_ylim(0, max([background] + counts) * 1.2 if counts else background * 1.2)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {os.path.basename(save_path)}")


def plot_feature_distributions_signal_background(
    X: np.ndarray,
    is_signal: np.ndarray,
    signal_class: np.ndarray,
    class_labels: Dict[int, Dict[str, object]],
    feature_names: List[str],
    output_dir: str,
    min_count: int = 0,
):
    os.makedirs(output_dir, exist_ok=True)
    signal_mask = is_signal == 1
    background_mask = is_signal == 0

    if not np.any(signal_mask) or not np.any(background_mask):
        print("  Skipping signal/background overlays (missing categories)")
        return

    class_ids = sorted(cid for cid in np.unique(signal_class) if cid != -1)
    if not class_ids:
        print("  No signal classes found for signal/background overlays")
        return

    sig = X[signal_mask]
    bkg = X[background_mask]
    palette = plt.cm.tab20(np.linspace(0, 1, max(len(class_ids), 3)))

    for idx, name in enumerate(feature_names):
        bkg_feat = bkg[:, idx]
        class_values = [X[signal_class == cid, idx] for cid in class_ids]
        bins_info = _compute_bins([bkg_feat] + class_values, feature_name=name)
        if bins_info is None:
            continue
        (lo, hi), bins = bins_info

        fig, ax = plt.subplots(figsize=(10, 7))
        bkg_finite = bkg_feat[np.isfinite(bkg_feat)]
        h_bkg, edges = np.histogram(bkg_finite, bins=bins)
        if h_bkg.sum() > 0:
            h_bkg = h_bkg / h_bkg.sum()
        hep.histplot((h_bkg, edges), histtype="fill", color=COLOR_BACKGROUND,
                     alpha=0.5, label=f"background (n={bkg_finite.size:,})", ax=ax)

        for color, cid in zip(palette, class_ids):
            mask = (signal_class == cid)
            values = X[mask, idx]
            finite = values[np.isfinite(values)]
            if finite.size <= min_count:
                continue
            label = class_labels.get(cid, {}).get("name", f"class {cid}")
            label = label.split(" (PID")[0]
            h_sig, _ = np.histogram(finite, bins=bins)
            if h_sig.sum() > 0:
                h_sig = h_sig / h_sig.sum()
            hep.histplot((h_sig, edges), histtype="step", linewidth=2.0, color=tuple(color),
                         label=f"{label} (n={finite.size:,})", ax=ax)

        ax.set_xlabel(name)
        ax.set_ylabel("Normalized jets")
        ax.set_xlim(lo, hi)
        ax.legend(frameon=True)
        ax.set_title(f"{name}: signal vs background")
        plt.tight_layout()
        out = os.path.join(output_dir, f"feature_sb_{idx:02d}_{safe_name(name)}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {os.path.basename(out)}")


def plot_feature_distributions_by_class(
    X: np.ndarray,
    is_signal: np.ndarray,
    signal_class: np.ndarray,
    class_labels: Dict[int, Dict[str, object]],
    feature_names: List[str],
    output_dir: str,
    min_count: int = 50,
):
    os.makedirs(output_dir, exist_ok=True)
    class_ids = sorted(cid for cid in np.unique(signal_class) if cid != -1)
    if not class_ids:
        print("  No signal classes found for per-class plots")
        return

    background_mask = is_signal == 0
    bkg = X[background_mask]
    palette = plt.cm.tab20(np.linspace(0, 1, max(len(class_ids), 3)))

    for idx, name in enumerate(feature_names):
        feat_bkg = bkg[:, idx]
        bins_info = _compute_bins([feat_bkg] + [X[signal_class == cid, idx] for cid in class_ids], feature_name=name)
        if bins_info is None:
            continue
        (lo, hi), bins = bins_info

        fig, ax = plt.subplots(figsize=(10, 7))
        bkg_finite = feat_bkg[np.isfinite(feat_bkg)]
        h_bkg, edges = np.histogram(bkg_finite, bins=bins)
        if h_bkg.sum() > 0:
            h_bkg = h_bkg / h_bkg.sum()
        hep.histplot((h_bkg, edges), histtype="fill", color=COLOR_BACKGROUND,
                     alpha=0.35, label=f"background (n={bkg_finite.size:,})", ax=ax)

        for color, cid in zip(palette, class_ids):
            mask = (signal_class == cid)
            values = X[mask, idx]
            finite = values[np.isfinite(values)]
            if finite.size < min_count:
                continue
            label = class_labels.get(cid, {}).get("name", f"class {cid}")
            label = label.split(" (PID")[0]
            h_sig, _ = np.histogram(finite, bins=bins)
            if h_sig.sum() > 0:
                h_sig = h_sig / h_sig.sum()
            hep.histplot((h_sig, edges), histtype="step", linewidth=2, color=tuple(color),
                         label=f"{label} (n={finite.size:,})", ax=ax)

        ax.set_xlabel(name)
        ax.set_ylabel("Normalized jets")
        ax.set_xlim(lo, hi)
        ax.legend(frameon=True, fontsize=10)
        ax.set_title(f"{name}: per-signal-class")
        plt.tight_layout()
        out = os.path.join(output_dir, f"feature_classes_{idx:02d}_{safe_name(name)}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {os.path.basename(out)}")


def plot_signal_mass_distributions(
    masses: np.ndarray,
    signal_class: np.ndarray,
    class_labels: Dict[int, Dict[str, object]],
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)
    class_ids = sorted(cid for cid in np.unique(signal_class) if cid != -1)
    if not class_ids:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    palette = plt.cm.tab20(np.linspace(0, 1, max(len(class_ids), 3)))

    for color, cid in zip(palette, class_ids):
        mask = (signal_class == cid)
        finite = masses[mask]
        finite = finite[np.isfinite(finite)]
        if finite.size < 5:
            continue
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        if hi <= lo:
            width = max(abs(lo) * 0.05, 0.1)
            lo = max(0.0, lo - width / 2.0)
            hi = lo + width
        bins = np.linspace(lo, hi, 41)
        hist, edges = np.histogram(finite, bins=bins)
        label = class_labels.get(cid, {}).get("name", f"class {cid}")
        label = label.split(" (PID")[0]
        hep.histplot((hist, edges), histtype="step", linewidth=2, color=tuple(color),
                     label=f"{label} (n={finite.size:,})", ax=ax)

    ax.set_xlabel("Truth mass [GeV]")
    ax.set_ylabel("Jets")
    ax.legend(frameon=True)
    ax.set_title("Signal truth mass per class")
    plt.tight_layout()
    out = os.path.join(output_dir, "signal_mass_by_class.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out)}")


def create_summary_pdf(save_dir: str, pdf_name: str = "feature_plots_summary.pdf"):
    """Collect PNGs in save_dir into a single PDF for quick browsing."""
    pngs = sorted([f for f in os.listdir(save_dir) if f.lower().endswith(".png")])
    if not pngs:
        print("  No PNG files found to assemble PDF")
        return

    pdf_path = os.path.join(save_dir, pdf_name)
    with PdfPages(pdf_path) as pdf:
        for png in pngs:
            img = plt.imread(os.path.join(save_dir, png))
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            ax.imshow(img)
            ax.axis("off")
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"  Saved: {pdf_name}, contains {len(pngs)} pages")


def main():
    parser = argparse.ArgumentParser(
        description="Plot ghost track features grouped by signal/background classes"
    )
    parser.add_argument("--input-h5", nargs="+", required=True, help="Input HDF5 files")
    parser.add_argument("--features-key", default="ghost_track_vars", help="Dataset name for features")
    parser.add_argument("--targets-key", default="targets", help="Dataset name for regression target")
    parser.add_argument("--output-dir", required=True, help="Directory for generated plots")
    parser.add_argument("--skip-signal-background", action="store_true", help="Skip aggregated signal vs background histograms")
    parser.add_argument("--skip-per-class", action="store_true", help="Skip per-signal-class feature histograms")
    parser.add_argument("--skip-mass", action="store_true", help="Skip truth mass per class plots")
    parser.add_argument("--create-pdf", action="store_true", help="Assemble all PNGs into a single summary PDF")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("Loading data...")
    print("=" * 80)
    X, y, is_signal, signal_class, class_meta = load_h5_multi(
        args.input_h5, args.features_key, args.targets_key
    )

    feature_names = [
        "nTracks",
        "deltaRLeadTrack",
        "leadTrackPt",
        "angularity_2",
        "U1_0p7",
        "M2_0p3",
        "tau2",
    ]
    if X.shape[1] != len(feature_names):
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    print(f"Feature shape: {X.shape}")
    print(f"Target shape:  {y.shape}")
    print(f"Signal jets:   {(is_signal == 1).sum():6d}")
    print(f"Background:    {(is_signal == 0).sum():6d}")

    class_labels = _build_class_labels(class_meta)
    if class_labels:
        print("\nSignal class metadata:")
        for cid in sorted(class_labels):
            info = class_labels[cid]
            mass = info.get("mass")
            mass_txt = f", mass={mass:.3f} GeV" if mass is not None and np.isfinite(mass) else ""
            print(f"  id={cid:2d}: {info.get('name')} (PID {info.get('pid', 'n/a')}{mass_txt})")

    print("\n" + "=" * 80)
    print("Creating plots...")
    print("=" * 80)

    print("\n1. Class breakdown:")
    plot_signal_breakdown(
        is_signal,
        signal_class,
        class_labels,
        os.path.join(args.output_dir, "class_breakdown.png"),
    )

    if not args.skip_signal_background:
        print("\n2. Feature distributions by class (normalized):")
        plot_feature_distributions_signal_background(
            X,
            is_signal,
            signal_class,
            class_labels,
            feature_names,
            args.output_dir,
        )
    else:
        print("\n2. Skipping class overlays")

    if not args.skip_mass:
        print("\n3. Truth mass distributions per signal class:")
        plot_signal_mass_distributions(
            y,
            signal_class,
            class_labels,
            args.output_dir,
        )
    else:
        print("\n3. Skipping mass plots")

    if args.create_pdf:
        print("\n4. Creating summary PDF:")
        create_summary_pdf(args.output_dir)

    print("\n" + "=" * 80)
    print("Plotting complete!")
    print("=" * 80)
    print(f"\nAll plots saved to: {args.output_dir}\n")


if __name__ == "__main__":
    main()
