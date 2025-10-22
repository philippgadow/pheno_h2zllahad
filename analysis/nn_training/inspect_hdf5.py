#!/usr/bin/env python3
"""
Inspect and validate HDF5 files produced by the Delphes converter.

Usage:
    python inspect_h5.py jet_data.h5
    python inspect_h5.py *.h5

Adds per feature float checks:
  - integer like fraction per column
  - unique counts and unique ratio
  - quantization step probe
  - sentinel and zero fractions
"""

import argparse
import h5py
import numpy as np
import sys

SENTINEL = -999999.0

def _integer_like_fraction(x: np.ndarray, eps: float = 1e-9) -> float:
    """Fraction of values that are within eps of an integer."""
    xf = x[np.isfinite(x)]
    if xf.size == 0:
        return 0.0
    return np.mean(np.abs(xf - np.round(xf)) <= eps)

def _quantization_step(x: np.ndarray) -> float:
    """
    Heuristic quantization probe.
    Tests a few candidate steps and returns the smallest step that explains most values.
    Returns 0.0 when no clear step is found.
    """
    xf = x[np.isfinite(x)]
    if xf.size < 50:
        return 0.0
    xf = xf[(xf != 0) & (xf != SENTINEL)]
    if xf.size < 50:
        return 0.0
    # Remove mean to reduce floating error
    m = np.mean(xf)
    xf = xf - m
    # Candidate steps
    steps = [1.0, 0.5, 0.25, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    best = 0.0
    best_cov = 0.0
    for s in steps:
        rem = np.mod(np.abs(xf), s)
        cov = np.mean((rem <= 1e-6) | (np.abs(rem - s) <= 1e-6))
        if cov > best_cov:
            best_cov = cov
            best = s
    return best if best_cov >= 0.98 else 0.0

def _column_report(mat: np.ndarray, names=None, max_cols: int = 32):
    """Print per column stats for a 2D float dataset."""
    n_rows, n_cols = mat.shape
    if n_cols > max_cols:
        print(f"    Note: showing first {max_cols} of {n_cols} columns")
        cols = range(max_cols)
    else:
        cols = range(n_cols)

    # Header
    print("    " + "-" * 96)
    print("    col  name                 dtype     uniques  uniq%   frac_int  frac_zero  frac_sentinel  q_step")
    print("    " + "-" * 96)

    for j in cols:
        col = mat[:, j]
        finite = np.isfinite(col)
        n_fin = finite.sum()
        if n_fin == 0:
            uniques = 0
            uniq_ratio = 0.0
            frac_int = 0.0
            frac_zero = 0.0
            frac_sentinel = np.mean(col == SENTINEL)
            q_step = 0.0
        else:
            colf = col[finite]
            uniques = len(np.unique(colf))
            uniq_ratio = uniques / max(1, n_fin)
            frac_int = _integer_like_fraction(colf)
            frac_zero = np.mean(colf == 0.0)
            frac_sentinel = np.mean(col == SENTINEL)
            q_step = _quantization_step(colf)

        nm = names[j] if names and j < len(names) else f"feat_{j}"
        print(f"    {j:>3d}  {nm[:20]:<20s}  {str(mat.dtype):<8s}  {uniques:>7d}  "
              f"{uniq_ratio:6.3f}   {frac_int:7.3f}   {frac_zero:9.3f}     {frac_sentinel:12.3f}   "
              f"{q_step if q_step>0 else 0.0:.3g}")

    print("    " + "-" * 96)
    print("    Legend:")
    print("      frac_int   fraction of values that are numerically integer")
    print("      uniq%      unique count divided by finite count")
    print("      q_step     detected quantization step, zero means none detected")
    print("      frac_sentinel includes non finite rows since sentinel can be present with NaNs elsewhere")

def _safe_attr_list(dset, key):
    try:
        v = dset.attrs.get(key, None)
        if v is None:
            return None
        # h5py may store as bytes
        out = []
        for item in v:
            if isinstance(item, bytes):
                out.append(item.decode("utf-8", errors="ignore"))
            else:
                out.append(str(item))
        return out
    except Exception:
        return None

def inspect_h5_file(filepath):
    """Inspect a single HDF5 file and print its structure."""
    print(f"\n{'='*80}")
    print(f"File: {filepath}")
    print('='*80)
    
    try:
        with h5py.File(filepath, 'r') as f:
            print("\nDatasets found:")
            print("-" * 80)
            
            datasets = {}
            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets[name] = obj
            f.visititems(visitor)
            
            if not datasets:
                print("  No datasets found!")
                return False
            
            # Print dataset information
            for name, dset in sorted(datasets.items()):
                shape = dset.shape
                dtype = dset.dtype
                print(f"\n  {name}")
                print(f"    Shape: {shape}")
                print(f"    Dtype: {dtype}")
                
                # Stats by rank
                if len(shape) == 1:
                    data = dset[:]
                    if np.issubdtype(dtype, np.floating):
                        finite_mask = np.isfinite(data)
                        n_finite = int(np.sum(finite_mask))
                        n_total = data.size
                        print(f"    Finite values: {n_finite}/{n_total}")
                        if n_finite > 0:
                            print(f"    Range: [{np.min(data[finite_mask]):.3f}, {np.max(data[finite_mask]):.3f}]")
                            print(f"    Mean ± Std: {np.mean(data[finite_mask]):.3f} ± {np.std(data[finite_mask]):.3f}")
                        # Float check
                        frac_int = _integer_like_fraction(data)
                        print(f"    Integer like fraction: {frac_int:.3f}")
                    else:
                        print(f"    Unique values: {len(np.unique(data))}")
                        
                elif len(shape) == 2:
                    data = dset[:]
                    if np.issubdtype(dtype, np.floating):
                        # Row validity
                        valid_mask = np.all(data != SENTINEL, axis=1) & np.all(np.isfinite(data), axis=1)
                        n_valid = int(np.sum(valid_mask))
                        print(f"    Valid rows: {n_valid}/{shape[0]}")
                        # Finite count
                        finite_mask = np.isfinite(data)
                        n_finite = int(np.sum(finite_mask))
                        n_total = data.size
                        print(f"    Finite elements: {n_finite}/{n_total}")
                        # Column float diagnostics
                        feat_names = _safe_attr_list(dset, "feature_names")
                        print(f"    Per feature float checks:")
                        _column_report(data, names=feat_names)
                    else:
                        # integer or other dtypes
                        uniq = "many" if data.size > 10000 else len(np.unique(data))
                        print(f"    Non float dataset, unique summary: {uniq}")
                        
                elif len(shape) == 3:
                    data = dset[:]
                    if 'mask' in name.lower():
                        n_true = int(np.sum(data))
                        print(f"    True values: {n_true}/{data.size}")
                    else:
                        finite_mask = np.isfinite(data)
                        n_finite = int(np.sum(finite_mask))
                        n_total = data.size
                        print(f"    Finite elements: {n_finite}/{n_total}")
                        # Quick float sanity
                        frac_int = _integer_like_fraction(data.reshape(-1))
                        print(f"    Integer like fraction, flattened: {frac_int:.3f}")
            
            # Regression training compatibility
            print("\n" + "="*80)
            print("Regression Training Compatibility Check:")
            print("-" * 80)
            
            required = ['ghost_track_vars', 'targets']
            optional = ['labels', 'jet_features', 'track_features', 'track_mask']
            
            all_ok = True
            for req in required:
                if req in datasets:
                    print(f"  ✓ {req:20s} - Found")
                else:
                    print(f"  ✗ {req:20s} - MISSING, required")
                    all_ok = False
            
            for opt in optional:
                if opt in datasets:
                    print(f"  ✓ {opt:20s} - Found")
                else:
                    print(f"  - {opt:20s} - Not found, optional")
            
            if all_ok:
                print("\n✓ File is compatible with regression training")
                gtv = datasets['ghost_track_vars'][:]
                targets = datasets['targets'][:]
                
                if gtv.shape[0] != targets.shape[0]:
                    print(f"\n⚠ Warning: ghost_track_vars and targets have different lengths")
                    print(f"  ghost_track_vars: {gtv.shape[0]}")
                    print(f"  targets: {targets.shape[0]}")
                    all_ok = False
                
                if gtv.shape[1] != 7:
                    print(f"\n⚠ Warning: ghost_track_vars should have 7 features, found {gtv.shape[1]}")
                    all_ok = False
                
                finite_targets = np.isfinite(targets)
                valid_features = np.all((gtv != SENTINEL) & np.isfinite(gtv), axis=1)
                usable = finite_targets & valid_features
                n_usable = int(np.sum(usable))
                
                print(f"\nUsable samples for training:")
                print(f"  Finite targets: {int(np.sum(finite_targets))}/{len(targets)}")
                print(f"  Valid features: {int(np.sum(valid_features))}/{len(gtv)}")
                print(f"  Both valid: {n_usable}/{len(targets)} ({100*n_usable/len(targets):.1f}%)")
                
                if n_usable < 100:
                    print(f"\n⚠ Warning: only {n_usable} usable samples")
                    all_ok = False
            else:
                print("\n✗ File is NOT compatible with regression training")
                print("   Missing required datasets")
            
            return all_ok
            
    except Exception as e:
        print(f"\n✗ Error reading file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Inspect HDF5 files from Delphes converter"
    )
    parser.add_argument("files", nargs="+", help="HDF5 files to inspect")
    parser.add_argument("--summary", action="store_true", 
                       help="Only show summary for each file")
    
    args = parser.parse_args()
    
    if len(args.files) == 1 and not args.summary:
        ok = inspect_h5_file(args.files[0])
        sys.exit(0 if ok else 1)
    else:
        print(f"\nInspecting {len(args.files)} files...")
        print("="*80)
        results = []
        for filepath in args.files:
            if args.summary:
                try:
                    with h5py.File(filepath, 'r') as f:
                        datasets = list(f.keys())
                        has_gtv = 'ghost_track_vars' in datasets
                        has_targets = 'targets' in datasets
                        if has_gtv and has_targets:
                            gtv_shape = f['ghost_track_vars'].shape
                            tgt_shape = f['targets'].shape
                            status = "✓ OK" if gtv_shape[1] == 7 else "⚠ WARN"
                        else:
                            gtv_shape = None
                            tgt_shape = None
                            status = "✗ MISSING"
                        results.append({
                            'file': filepath,
                            'status': status,
                            'gtv_shape': gtv_shape,
                            'target_shape': tgt_shape
                        })
                except Exception as e:
                    results.append({
                        'file': filepath,
                        'status': f"✗ ERROR: {e}",
                        'gtv_shape': None,
                        'target_shape': None
                    })
            else:
                inspect_h5_file(filepath)
        
        if args.summary:
            print("\nSummary:")
            print("-" * 80)
            print(f"{'Status':<12} {'File':<40} {'GTV Shape':<15} {'Target Shape'}")
            print("-" * 80)
            for r in results:
                gtv_str = str(r['gtv_shape']) if r['gtv_shape'] else "N/A"
                tgt_str = str(r['target_shape']) if r['target_shape'] else "N/A"
                print(f"{r['status']:12s} {r['file']:<40s} {gtv_str:<15s} {tgt_str}")

if __name__ == "__main__":
    main()
