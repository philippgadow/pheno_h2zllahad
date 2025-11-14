#!/usr/bin/env python3
"""Inspect HDF5 files produced by convert_to_h5.py."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

SENTINEL = -999999.0


def format_float(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.6g}"
    return str(value)


def decode_bytes(dataset):
    if dataset is None:
        return []
    out = []
    for item in dataset:
        if isinstance(item, (bytes, np.bytes_)):
            out.append(item.decode("utf-8", errors="ignore"))
        else:
            out.append(str(item))
    return out


def describe_1d_numeric(name, data):
    info = {"name": name, "dtype": str(data.dtype), "shape": data.shape, "kind": "1d"}
    values = np.asarray(data)
    finite = np.isfinite(values)
    info["finite"] = int(finite.sum())
    info["total"] = int(values.size)
    if info["finite"] > 0:
        vals = values[finite]
        info["min"] = float(np.min(vals))
        info["max"] = float(np.max(vals))
        info["mean"] = float(np.mean(vals))
        info["std"] = float(np.std(vals))
    return info


def describe_2d_numeric(name, data):
    info = {"name": name, "dtype": str(data.dtype), "shape": data.shape, "kind": "2d"}
    values = np.asarray(data)
    finite = np.isfinite(values)
    info["finite_elements"] = int(finite.sum())
    info["total_elements"] = int(values.size)
    rows = values.shape[0]
    sentinel_mask = np.any(values == SENTINEL, axis=1)
    info["rows_with_sentinel"] = int(sentinel_mask.sum())
    finite_rows = np.all(finite, axis=1)
    info["fully_valid_rows"] = int(finite_rows.sum())
    return info


def inspect_h5(file_path):
    file_path = Path(file_path)
    result = {"file": str(file_path), "datasets": {}, "signal_classes": []}

    try:
        with h5py.File(file_path, "r") as f:
            keys = list(f.keys())
            result["datasets"]["keys"] = keys

            # Handle numeric datasets
            for name in keys:
                dset = f[name]
                if isinstance(dset, h5py.Dataset) and np.issubdtype(dset.dtype, np.number):
                    if dset.ndim == 1:
                        result["datasets"][name] = describe_1d_numeric(name, dset[:])
                    elif dset.ndim == 2:
                        result["datasets"][name] = describe_2d_numeric(name, dset[:])

            # Signal class metadata
            if "is_signal" in f:
                labels = f["is_signal"][:]
                labels = labels.astype(np.int8)
                result["signal_counts"] = {
                    "signal": int((labels > 0.5).sum()),
                    "background": int((labels <= 0.5).sum()),
                }

            if "signal_class_ids" in f:
                ids = f["signal_class_ids"][:]
                names = decode_bytes(f.get("signal_class_names"))
                keys = decode_bytes(f.get("signal_class_keys"))
                masses = f.get("signal_class_mass_GeV")
                pids = f.get("signal_class_truth_pid")
                sources = decode_bytes(f.get("signal_class_source_file"))

                for idx, cid in enumerate(ids):
                    entry = {"id": int(cid)}
                    if idx < len(names):
                        entry["name"] = names[idx]
                    if idx < len(keys):
                        entry["key"] = keys[idx]
                    if masses is not None:
                        entry["mass"] = float(masses[idx])
                    if pids is not None:
                        entry["pid"] = int(pids[idx])
                    if idx < len(sources):
                        entry["source"] = sources[idx]
                    result["signal_classes"].append(entry)

    except Exception as err:
        result["error"] = str(err)

    return result


def summarize_file(file_path):
    data = inspect_h5(file_path)
    file = data.get("file", file_path)
    if "error" in data:
        print(f"✗ {file}: {data['error']}")
        return

    keys = data["datasets"].get("keys", [])
    print(f"\n{'='*80}\nFile: {file}\n{'='*80}")
    print(f"Datasets: {keys}")

    for name in ["jet_features", "ghost_track_vars", "targets", "is_signal"]:
        if name in data["datasets"]:
            stats = data["datasets"][name]
            if stats["kind"] == "1d":
                print(f"  {name}: shape={stats['shape']} finite={stats['finite']}/{stats['total']} range=[{format_float(stats.get('min'))}, {format_float(stats.get('max'))}]")
            elif stats["kind"] == "2d":
                print(f"  {name}: shape={stats['shape']} finite elems={stats['finite_elements']}/{stats['total_elements']} fully valid rows={stats['fully_valid_rows']} sentinel rows={stats['rows_with_sentinel']}")

    if "signal_counts" in data:
        sig = data["signal_counts"]
        print(f"Signal jets: {sig['signal']}  background: {sig['background']}")

    if data.get("signal_classes"):
        print("Signal classes:")
        for entry in data["signal_classes"]:
            mass_txt = f", mass={format_float(entry.get('mass'))} GeV" if entry.get("mass") is not None else ""
            pid_txt = f", PID {entry.get('pid')}" if entry.get("pid") is not None else ""
            name = entry.get("name", "")
            print(f"  id={entry['id']} {name}{pid_txt}{mass_txt}")


def main():
    parser = argparse.ArgumentParser(description="Inspect HDF5 files from convert_to_h5.py")
    parser.add_argument("files", nargs="+", help="HDF5 files to inspect")
    parser.add_argument("--summary", action="store_true", help="Print summary instead of full details")
    parser.add_argument("--output-manifest", help="Optional JSON out path containing inspection results")
    args = parser.parse_args()

    results = []
    for file_path in args.files:
        data = inspect_h5(file_path)
        results.append(data)
        if args.summary:
            file = data.get("file", file_path)
            if "error" in data:
                print(f"✗ {file}: {data['error']}")
            else:
                sig_counts = data.get("signal_counts", {})
                sig = sig_counts.get("signal", 0)
                bkg = sig_counts.get("background", 0)
                classes = len(data.get("signal_classes", []))
                print(f"{file}: jets={data['datasets'].get('jet_features', {}).get('shape', 'N/A')} signal={sig} background={bkg} classes={classes}")
        else:
            summarize_file(file_path)

    if args.output_manifest:
        path = Path(args.output_manifest)
        path.write_text(json.dumps(results, indent=2))
        print(f"Manifest written to {path}")


if __name__ == "__main__":
    main()
