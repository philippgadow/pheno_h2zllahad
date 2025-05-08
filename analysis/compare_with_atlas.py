from coffea.nanoevents import NanoEventsFactory, DelphesSchema  # type: ignore
import hist
import uproot  # type: ignore
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep


atlas_file = "HZa_TRUTH3/mc15_13TeV.600973.PhPy8EG_HZetac.DAOD_TRUTH3.root"
delphes_file = "output/ggH_2HDM/delphes_output_HZetac.root"


def load_events(filename):
    # Load events from a ROOT file
    events = NanoEventsFactory.from_root(
        filename,
        schemaclass=DelphesSchema,
        entry_stop=1000,
        metadata={"exclude": ["Particle.fBits", "Particle.fUniqueID"]},

    ).events()
    return events


def load_atlas(filename):
    f = uproot.open(filename)
    t = f["CollectionTree"]
    arrays = ["TruthMuonsAuxDyn.px", "TruthMuonsAuxDyn.py", "TruthMuonsAuxDyn.pz", "TruthMuonsAuxDyn.e"]
    # Load the arrays from the ROOT file
    return t.arrays(arrays, library="ak",)


events_atlas = load_atlas(atlas_file)
events_delphes = load_events(delphes_file)

# plot muon pt
MeV_to_GeV = 0.001
muon_atlas = ak.zip({
    "px": events_atlas["TruthMuonsAuxDyn.px"] * MeV_to_GeV,
    "py": events_atlas["TruthMuonsAuxDyn.py"] * MeV_to_GeV,
    "pz": events_atlas["TruthMuonsAuxDyn.pz"] * MeV_to_GeV,
    "e": events_atlas["TruthMuonsAuxDyn.e"] * MeV_to_GeV,
}, with_name="TLorentzVector")

muon_delphes = events_delphes.Particle[events_delphes.Particle.PID == 13]
muon_delphes = ak.zip({
    "px": muon_delphes.Px.compute(),
    "py": muon_delphes.Py.compute(),
    "pz": muon_delphes.Pz.compute(),
    "e": muon_delphes.E.compute(),
}, with_name="TLorentzVector")

# plot first muon in each event
first_muon_atlas = ak.firsts(muon_atlas)
first_muon_delphes = ak.firsts(muon_delphes)


# compute pt (not implemented in TLorentzVector)
def pt(muon):
    return np.sqrt(muon.px**2 + muon.py**2)


first_muon_atlas._pt = pt(first_muon_atlas)
first_muon_delphes._pt = pt(first_muon_delphes)

# Replace missing values with 0.0 and compute to numpy
lead_pt_atlas = ak.fill_none(first_muon_atlas._pt, 0.0)
lead_pt_delphes = ak.fill_none(first_muon_delphes._pt, 0.0)

# Define histogram axes
axis = hist.axis.Regular(
    40, 10, 210,
    name="pt",
    label=r"$p_{T}$ [GeV]"
)
# Create histograms with Double storage
h_atlas = hist.Hist(axis, storage=hist.storage.Double())
h_delphes = hist.Hist(axis, storage=hist.storage.Double())

# Compute weights for unit-area normalization
weights_atlas = np.ones_like(lead_pt_atlas)
weights_delphes = np.ones_like(lead_pt_delphes)

# Fill histograms with weights
h_atlas.fill(pt=lead_pt_atlas, weight=weights_atlas)
h_delphes.fill(pt=lead_pt_delphes, weight=weights_delphes)

# Plot
hep.style.use("CMS") 

# Create figure with ratio subplot
fig, (ax_main, ax_ratio) = plt.subplots(
    2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}, sharex=True
)

# Plot histograms using mplhep
hep.histplot(h_atlas, ax=ax_main, label="ATLAS", histtype="step")
hep.histplot(h_delphes, ax=ax_main, label="Delphes", histtype="step")

# Decorate main axis
ax_main.set_ylabel("Entries")
ax_main.legend()
ax_main.grid(True)

# Compute ratio (Delphes / ATLAS)
ratio = h_delphes.values() / h_atlas.values()
ratio[np.isnan(ratio)] = 0  # avoid division by zero
ratio[np.isinf(ratio)] = 0

# Plot ratio
bin_edges = h_atlas.axes[0].edges
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
ax_ratio.step(bin_centers, ratio, where='mid', color='black')
ax_ratio.axhline(1.0, color='red', linestyle='--')
ax_ratio.set_ylim(0, 2)
ax_ratio.set_xlabel(r"$p_{T}$ [GeV]")
ax_ratio.set_ylabel("Ratio")
ax_ratio.grid(True)

# Save figure
fig.savefig("plots/compare_muon_pt_with_ratio_mplhep.png")
fig.savefig("plots/compare_muon_pt_with_ratio_mplhep.pdf")
plt.close(fig)