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
    #"reallep": "_reallep_MPIon_ISRon_FSRon_HADon_Amass4.0",
    #"truelep": "_truelep_MPIon_ISRon_FSRon_HADon_Amass4.0",
    #"MPIoff":  "_reallep_MPIoff_ISRon_FSRon_HADon_Amass4.0",
    #"ISRoff": "_reallep_MPIon_ISRoff_FSRon_HADon_Amass4.0",
    #"FSRoff": "_reallep_MPIon_ISRon_FSRoff_HADon_Amass4.0",
    #"HADoff": "_reallep_MPIon_ISRon_FSRon_HADoff_Amass4.0",
    "0.5GeV": "_reallep_MPIon_ISRon_FSRon_HADon_Amass0.5",
    "2.0GeV": "_reallep_MPIon_ISRon_FSRon_HADon_Amass2.0",
    "3.0GeV": "_reallep_MPIon_ISRon_FSRon_HADon_Amass3.0",
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

# Plot all histograms overlaid for each variable
with PdfPages("plot_PhPy8_HZA_comparison.pdf") as pdf:
    for hname in hists:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.set_title(hname)
        #ax.set_xlabel("Bin center")
        ax.set_ylabel("Normalized entries")
        
        # Loop over configs and plot them
        for label, opt in configs.items():
            if hname in data[label]:
                hist = data[label][hname]
                ax.stairs(hist[:-1, 1], edges=hist[:, 2], label=label)
        
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        pdf.savefig(fig)
        plt.close(fig)

print("Comparison plots saved to plot_PhPy8_HZA_comparison.pdf")

