import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

hfol = "hists_PhPy8_HZA"

hists = [
    "hist_j1mass",
    "hist_jet1pt",
    "hist_lem_pt",
    "hist_lep_pt",
    "hist_mlljet",
    "hist_z_mass"
]

OPT = "_reallep_MPIon_ISRon_FSRon_HADon"

hh = {}
for h in hists:
    hh[h] = np.loadtxt(hfol+"/"+h+OPT+".dat")

with PdfPages("plot_PhPy8_HZA_dohist.pdf") as pdf:
    for hname, hist in hh.items():
        fig, ax = plt.subplots()
        ax.set_title(hname)
        ax.stairs(hist[:-1,1], edges=hist[:,2])
        pdf.savefig(fig)
        plt.close(fig)
