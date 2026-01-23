import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# Folder containing histograms
hfol = "hists_PhPy8_HZA"

# List of histogram base names
hists = [
    "hist_j1mass",
    "hist_jet1pt",
    "hist_lem_pt",
    "hist_lep_pt",
    "hist_mlljet",
    "hist_z_mass",
    "hist_proxy_nTracks",
    "hist_proxy_deltaRLeadTrack",
    "hist_proxy_leadTrackPtRatio",
    "hist_proxy_angularity_2",
    "hist_proxy_U1_0p7",
    "hist_proxy_M2_0p3",
    "hist_proxy_tau2"
]

# Different configurations (suffixes) 
configs = {
    "reallep": "_reallep_MPIon_ISRon_FSRon_HADon_Amass4.0",
    #"truelep": "_truelep_MPIon_ISRon_FSRon_HADon_Amass4.0",
    "MPIoff":  "_reallep_MPIoff_ISRon_FSRon_HADon_Amass4.0",
    "ISRoff": "_reallep_MPIon_ISRoff_FSRon_HADon_Amass4.0",
    "FSRoff": "_reallep_MPIon_ISRon_FSRoff_HADon_Amass4.0",
    "HADoff": "_reallep_MPIon_ISRon_FSRon_HADoff_Amass4.0",
    #"0.5GeV": "_reallep_MPIon_ISRon_FSRon_HADon_Amass0.5",
    #"2.0GeV": "_reallep_MPIon_ISRon_FSRon_HADon_Amass2.0",
    #"3.0GeV": "_reallep_MPIon_ISRon_FSRon_HADon_Amass3.0",
}

# Load all histograms for each config
data = {cfg: {} for cfg in configs}

for label, opt in configs.items():
    for h in hists:
        fname = f"{hfol}/{h}{opt}.dat"
        try:
            data[label][h] = np.loadtxt(fname)
        except FileNotFoundError:
            print(f"Missing: {fname}")
            continue

# --------------------------
# Updated plotting: top hist + ratio to "truelep" underneath
# --------------------------
with PdfPages("plot_PhPy8_HZA_comparison_with_ratios.pdf") as pdf:
    for hname in hists:
        # Check whether truelep is available for this histogram
        true_available = ('reallep' in data) and (hname in data['reallep'])
        if not true_available:
            print(f"Warning: 'reallep' missing for {hname}. Ratio subplot will be hidden for this variable.")

        # Create figure with ratio subplot (top:bottom = 2:1 -> bottom is half height of top)
        fig, (ax_top, ax_ratio) = plt.subplots(
            2, 1,
            sharex=True,
            gridspec_kw={'height_ratios': [2, 1]},
            figsize=(6, 6)  # taller to keep ratio readable; adjust if you prefer
        )

        ax_top.set_title(hname)
        ax_top.set_ylabel("Normalized entries")

        # If truelep exists, take its counts and edges as reference
        if true_available:
            true_hist = data['reallep'][hname]
            true_counts = true_hist[:-1, 1]    # bin contents (per your original scheme)
            edges = true_hist[:, 2]            # bin edges
        else:
            true_counts = None
            edges = None

        plotted_any = False
        # Keep track of ratios to determine dynamic y-limits
        all_ratio_vals = []

        for label, opt in configs.items():
            if hname not in data[label]:
                continue

            hist = data[label][hname]
            values = hist[:-1, 1]
            this_edges = hist[:, 2] if edges is None else edges  # prefer truelep edges when available

            # Top: overlay histograms as before
            ax_top.stairs(values, edges=this_edges, label=label, linewidth=1)
            plotted_any = True

            # Bottom: ratio to truelep when possible
            if true_counts is not None:
                # safe division; bins where true_counts == 0 will be set to np.nan
                ratio = np.zeros_like(values, dtype=float)
                np.divide(values, true_counts, out=ratio, where=(true_counts != 0))
                # put nan where division would be undefined (true_counts==0)
                ratio[(true_counts == 0)] = np.nan

                # record finite entries for y-limits
                finite = ratio[np.isfinite(ratio)]
                if finite.size > 0:
                    all_ratio_vals.append(finite)

                # Draw ratio (use stairs so bin-alignment matches top)
                ax_ratio.stairs(ratio, edges=this_edges, label=label)

        if not plotted_any:
            print(f"No histograms found for {hname}, skipping.")
            plt.close(fig)
            continue

        # Top cosmetics
        ax_top.legend()
        ax_top.grid(True, linestyle="--", alpha=0.5)
        #logarithmic axes
        #ax_top.set_yscale("log")

        # Ratio cosmetics and dynamic y-limits
        if true_available:
            ax_ratio.set_xlabel("Energy")
            ax_ratio.set_ylabel("Ratio / reallep")
            ax_ratio.axhline(1.0, linestyle='--', linewidth=1, zorder=-1)

            # collect all finite ratio values into one array
            if len(all_ratio_vals) > 0:
                all_finite = np.hstack(all_ratio_vals)
                # Compute padded limits around 1.0 but ensure reasonable minimum span
                rmin, rmax = np.min(all_finite), np.max(all_finite)
                # Add 10% padding relative to max deviation from 1
                dev = max(abs(rmax - 1.0), abs(1.0 - rmin), 1e-6)
                ymin = max(0.0, 1.0 - 1.2 * dev)
                ymax = 1.0 + 1.2 * dev
                # enforce a minimum span if data too close to 1
                if (ymax - ymin) < 0.5:
                    center = 1.0
                    ymin = max(0.0, center - 0.25)
                    ymax = center + 0.25
                ax_ratio.set_ylim(ymin, ymax)
            else:
                # fallback if no finite ratio values
                ax_ratio.set_ylim(0, 2)

            ax_ratio.grid(True, linestyle="--", alpha=0.5)
            # Smaller legend for ratio (optional)
            ax_ratio.legend(ncol=2, fontsize="small")
            #set limits to ratio to (-3,3)
            ax_ratio.set_ylim(0, 3)
        #else:
            # hide ratio axis if we can't compute it
            #ax_ratio.set_visible(False)
            # set xlabel on top instead (optional)
            #ax_top.set_xlabel("Variable")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

print("Comparison plots saved to plot_PhPy8_HZA_comparison_ratios.pdf")
