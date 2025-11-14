#!/usr/bin/env python3
"""
Visualize regression network outputs for background and each signal class.
"""

import argparse
import os

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from train_regression_pytorch import SENTINEL, load_signal_class_lookup
except ImportError:  # pragma: no cover
    from .train_regression_pytorch import SENTINEL, load_signal_class_lookup


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot regression output distributions for signal classes vs background."
    )
    parser.add_argument("--input-h5", required=True, help="HDF5 file with regression outputs.")
    parser.add_argument(
        "--predictions-key",
        default="regression_prediction",
        help="Dataset containing regression output (default: regression_prediction).",
    )
    parser.add_argument(
        "--is-signal-key",
        default="is_signal",
        help="Dataset indicating signal (1) vs background (0).",
    )
    parser.add_argument(
        "--class-key",
        default="signal_class",
        help="Dataset containing per-jet signal class IDs.",
    )
    parser.add_argument(
        "--output",
        default="nn_plots/regression_output_overlay.png",
        help="Path for the raw-yield output PNG.",
    )
    parser.add_argument(
        "--normalized-output",
        help="Path for the normalized-to-unit-area PNG (default: append '_normalized').",
    )
    parser.add_argument("--bins", type=int, default=80, help="Number of histogram bins.")
    parser.add_argument(
        "--range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Explicit range for the histogram. Defaults to percentile-based.",
    )
    parser.add_argument("--logy", action="store_true", help="Use log scale on the y-axis.")
    return parser.parse_args()


def ensure_dir(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def main():
    args = parse_args()
    ensure_dir(args.output)
    norm_output = args.normalized_output
    if not norm_output:
        root, ext = os.path.splitext(args.output)
        norm_output = f"{root}_normalized{ext or '.png'}"
    ensure_dir(norm_output)

    with h5py.File(args.input_h5, "r") as f:
        if args.predictions_key not in f:
            raise KeyError(f"Dataset '{args.predictions_key}' missing in {args.input_h5}")
        preds = f[args.predictions_key][:].astype(np.float32)

        if args.is_signal_key not in f:
            raise KeyError(f"Dataset '{args.is_signal_key}' missing in {args.input_h5}")
        is_signal = f[args.is_signal_key][:].astype(np.int8)

        signal_class = f[args.class_key][:].astype(np.int64) if args.class_key in f else None

    mask = np.isfinite(preds) & (preds != SENTINEL)
    preds = preds[mask]
    is_signal = is_signal[mask]
    signal_class = signal_class[mask] if signal_class is not None else None

    if preds.size == 0:
        raise RuntimeError("No valid regression predictions to plot.")

    if args.range:
        xmin, xmax = args.range
    else:
        xmin, xmax = np.percentile(preds, [0.5, 99.5])
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
            xmin, xmax = np.min(preds), np.max(preds) + 1e-6

    bins = np.linspace(float(xmin), float(xmax), args.bins + 1)

    bkg = preds[is_signal <= 0]
    sig = preds[is_signal > 0]

    lookup = load_signal_class_lookup(args.input_h5)

    def _make_plot(output_path, density=False, logy=args.logy and not density):
        fig, ax = plt.subplots(figsize=(8, 7))

        if bkg.size > 0:
            ax.hist(
                bkg,
                bins=bins,
                histtype="step",
                linewidth=2,
                color="black",
                label=f"Background (n={len(bkg)})",
                density=density,
            )

        if signal_class is not None and sig.size > 0:
            unique_classes = sorted(set(signal_class[(signal_class >= 0) & (is_signal > 0)]))
            if unique_classes:
                colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(unique_classes))))
                for idx, cls in enumerate(unique_classes):
                    mask_cls = (signal_class == cls) & (is_signal > 0)
                    cls_preds = preds[mask_cls]
                    if cls_preds.size == 0:
                        continue
                    meta = lookup.get(int(cls), {})
                    label = meta.get("name") or meta.get("key") or f"class {cls}"
                    mass = meta.get("mass_GeV")
                    if isinstance(mass, (float, int)) and np.isfinite(mass):
                        label += f" (m={mass:.2f} GeV)"
                    label += f" (n={cls_preds.size})"
                    ax.hist(
                        cls_preds,
                        bins=bins,
                        histtype="step",
                        linestyle="--",
                        linewidth=2,
                        color=colors[idx % len(colors)],
                        label=label,
                        density=density,
                    )
        elif sig.size > 0:
            ax.hist(
                sig,
                bins=bins,
                histtype="step",
                linestyle="--",
                linewidth=2,
                color="tab:red",
                label=f"Signal (n={len(sig)})",
                density=density,
            )

        ax.set_xlabel("Regression output")
        ax.set_ylabel("Normalized events" if density else "Events")
        ax.set_title("Regression output distributions" + (" (normalized)" if density else ""))
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-1)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Saved regression output plot to {output_path}")

    _make_plot(args.output, density=False, logy=args.logy)
    _make_plot(norm_output, density=True, logy=False)


if __name__ == "__main__":
    main()
