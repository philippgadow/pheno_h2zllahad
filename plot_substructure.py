#!/usr/bin/env python3
"""
Compare substructure / ghost-track proxy variables between Herwig and Pythia.
Reads the flat branches written by hepmc_to_root_substructure.py.
Only events that passed selection (passed == True, i.e. non-NaN entries) are plotted.
"""

import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FILE1      = "herwig_ghost.root"
FILE2      = "pythia_ghost.root"
LABEL1     = "Herwig"
LABEL2     = "Pythia"
COLOR1     = "blue"
COLOR2     = "red"
TREE_NAME  = "events"

# Variables to plot: (branch_name, x_label, n_bins, (x_min, x_max), log_y)
VARIABLES = [
    ("gt_nTracks",     r"$N_\mathrm{tracks}$",             40,  (0,   40),   False),
    ("gt_leadDR",      r"$\Delta R(\mathrm{lead\,trk, jet})$", 40, (0.0, 0.4), False),
    ("gt_leadPtRatio", r"$p_T^\mathrm{lead} / p_T^\mathrm{jet}$", 40, (0.0, 1.5), False),
    ("gt_angularity2", r"Angularity $\langle\Delta R^2\rangle$", 40, (0.0, 0.4), False),
    ("gt_U1_0p7",      r"$U_1$ ($\Delta R < 0.7$)",        40,  (0.0, 0.4),  False),
    ("gt_M2_0p3",      r"$M_2$ ($\Delta R < 0.3$)",        40,  (0.0, 0.4),  False),
    ("gt_tau2",        r"$\tau_2$",                         40,  (0.0, 1.0),  False),
    ("jet1_pt",        r"Leading jet $p_T$ [GeV]",          40,  (0,   300),  False),
    ("jet1_eta",       r"Leading jet $\eta$",               40,  (-3,  3),    False),
    ("jet1_mass",      r"Leading jet mass [GeV]",           40,  (0,   30),   False),
    ("mlljet",         r"$m_{\ell\ell j}$ [GeV]",          40,  (100, 250),  False),
    ("mll",            r"$m_{\ell\ell}$ [GeV]",            40,  (81,  101),  False),
    ("lep_pt",         r"$p_T^{\ell^+}$ [GeV]",            40,  (0,   200),  False),
    ("lem_pt",         r"$p_T^{\ell^-}$ [GeV]",            40,  (0,   200),  False),
]

# ---------------------------------------------------------------------------
# Load data — read flat branches, drop NaN (failed-selection events)
# ---------------------------------------------------------------------------

def load_branch(tree, name):
    arr = tree[name].array(library="np")
    return arr[~np.isnan(arr)]

print(f"Opening {FILE1} and {FILE2} ...")
tree1 = uproot.open(FILE1)[TREE_NAME]
tree2 = uproot.open(FILE2)[TREE_NAME]

data1 = {var: load_branch(tree1, var) for var, *_ in VARIABLES}
data2 = {var: load_branch(tree2, var) for var, *_ in VARIABLES}

n1 = len(data1["mll"])
n2 = len(data2["mll"])
print(f"  {LABEL1}: {n1} events passing selection")
print(f"  {LABEL2}: {n2} events passing selection")

# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def make_comparison_plot(ax_main, ax_ratio, d1, d2, label, bins, xrange, logy):
    h1, edges = np.histogram(d1, bins=bins, range=xrange, density=True)
    h2, _     = np.histogram(d2, bins=bins, range=xrange, density=True)
    centers   = (edges[:-1] + edges[1:]) / 2
    width     = edges[1] - edges[0]

    ax_main.step(edges[:-1], h1, where='post', color=COLOR1, linewidth=1.8, label=LABEL1)
    ax_main.step(edges[:-1], h2, where='post', color=COLOR2, linewidth=1.8, label=LABEL2)
    ax_main.set_ylabel("Normalised counts", fontsize=9)
    ax_main.set_xlim(xrange)
    ax_main.legend(fontsize=8)
    ax_main.grid(True, alpha=0.3)
    ax_main.tick_params(labelbottom=False)
    if logy:
        ax_main.set_yscale('log')

    ratio = np.divide(h2, h1, out=np.ones_like(h1), where=h1 != 0)
    ax_ratio.plot(centers, ratio, 'ko', markersize=2.5)
    ax_ratio.axhline(1, color='gray', linestyle='--', linewidth=1)
    ax_ratio.set_xlabel(label, fontsize=9)
    ax_ratio.set_ylabel(f"{LABEL2}/{LABEL1}", fontsize=8)
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_xlim(xrange)
    ax_ratio.grid(True, alpha=0.3)
    ax_ratio.tick_params(axis='both', labelsize=7)

# ---------------------------------------------------------------------------
# Individual plots (one file per variable)
# ---------------------------------------------------------------------------

print("Saving individual plots ...")
for var, xlabel, bins, xrange, logy in VARIABLES:
    fig = plt.figure(figsize=(5, 5))
    gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_main  = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1])

    make_comparison_plot(ax_main, ax_ratio,
                         data1[var], data2[var],
                         xlabel, bins, xrange, logy)
    ax_main.set_title(xlabel, fontsize=10)

    fname = f"substructure_{var}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

# ---------------------------------------------------------------------------
# Combined summary plot  (7 substructure variables in one figure)
# ---------------------------------------------------------------------------

SUBSTR_VARS = [v for v in VARIABLES if v[0].startswith("gt_")]
n_sub = len(SUBSTR_VARS)   # 7

fig, axes = plt.subplots(2, n_sub, figsize=(4 * n_sub, 5),
                          gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

for col, (var, xlabel, bins, xrange, logy) in enumerate(SUBSTR_VARS):
    make_comparison_plot(axes[0, col], axes[1, col],
                         data1[var], data2[var],
                         xlabel, bins, xrange, logy)
    axes[0, col].set_title(xlabel, fontsize=9)
    if col > 0:
        axes[0, col].set_ylabel("")
        axes[1, col].set_ylabel("")

fig.suptitle(f"Ghost-track proxy variables — {LABEL1} vs {LABEL2}", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("substructure_summary.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: substructure_summary.png")

# ---------------------------------------------------------------------------
# Combined jet / lepton / mll variables in one figure
# ---------------------------------------------------------------------------

KINEM_VARS = [v for v in VARIABLES if not v[0].startswith("gt_")]
n_kin = len(KINEM_VARS)

fig, axes = plt.subplots(2, n_kin, figsize=(4 * n_kin, 5),
                          gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

for col, (var, xlabel, bins, xrange, logy) in enumerate(KINEM_VARS):
    make_comparison_plot(axes[0, col], axes[1, col],
                         data1[var], data2[var],
                         xlabel, bins, xrange, logy)
    axes[0, col].set_title(xlabel, fontsize=9)
    if col > 0:
        axes[0, col].set_ylabel("")
        axes[1, col].set_ylabel("")

fig.suptitle(f"Kinematic variables — {LABEL1} vs {LABEL2}", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("kinematics_summary.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: kinematics_summary.png")

print("\nAll done.")
