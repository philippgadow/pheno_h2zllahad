import os
import re
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import mplhep as hep
import awkward as ak
import numpy as np
from coffea.nanoevents import NanoEventsFactory, DelphesSchema  # type: ignore
import hist


def load_events(filename):
    """
    Load events from a ROOT file using DelphesSchema.
    """
    events = NanoEventsFactory.from_root(
        filename,
        schemaclass=DelphesSchema,
        entry_stop=1000,
        metadata={"exclude": ["Particle.fBits", "Particle.fUniqueID"]},
    ).events()
    return events


def plot(events_sig, events_bkg, plot_type, output_dir):
    """
    Plot leading jet pT distributions for signal and background,
    normalized to unit area using hist package with weighted fills.
    """
    # Extract jagged arrays of jet pT
    jets_sig = events_sig.Jet
    jets_bkg = events_bkg.Jet

    # Get first jet pT per event (None if no jets)
    first_pt_sig = ak.firsts(jets_sig.PT)
    first_pt_bkg = ak.firsts(jets_bkg.PT)

    # Replace missing values with 0.0 and compute to numpy
    lead_pt_sig = ak.fill_none(first_pt_sig, 0.0).compute()
    lead_pt_bkg = ak.fill_none(first_pt_bkg, 0.0).compute()

    # Define histogram axes
    axis = hist.axis.Regular(
        40, 10, 210,
        name="pt",
        label=r"$p_{T}$ [GeV]"
    )

    # Create histograms with Double storage
    h_sig = hist.Hist(axis, storage=hist.storage.Double())
    h_bkg = hist.Hist(axis, storage=hist.storage.Double())

    # Compute weights for unit-area normalization
    weights_sig = np.ones_like(lead_pt_sig)
    weights_bkg = np.ones_like(lead_pt_bkg)

    # Fill histograms with weights
    h_sig.fill(pt=lead_pt_sig, weight=weights_sig)
    h_bkg.fill(pt=lead_pt_bkg, weight=weights_bkg)

    # Plot
    hep.style.use(hep.style.CMS)
    fig, ax = plt.subplots()
    h_sig.plot(ax=ax, histtype='step', density=False, label="Signal (eta_C)")
    h_bkg.plot(ax=ax, histtype='step', density=False, label="Background (Z)")
    ax.legend()
    ax.set_xlabel(r'$p_{T}$ [GeV]')
    ax.set_ylabel('Normalized entries')
    ax.grid(True)

    # Save plot
    output_path = os.path.join(output_dir, f'{plot_type}_comp_eta_vs_Z_lead_jetpt.png')
    fig.savefig(output_path)
    print(f"Normalized plot saved to {output_path}")


    # --- 2) Leading-jet invariant mass ---
    # pull out the mass of the first jet in each event
    first_mass_sig = ak.firsts(jets_sig.Mass)
    first_mass_bkg = ak.firsts(jets_bkg.Mass)
    lead_mass_sig = ak.fill_none(first_mass_sig, 0.0).compute()
    lead_mass_bkg = ak.fill_none(first_mass_bkg, 0.0).compute()

    # define a mass axis (e.g. 0–200 GeV in 40 bins)
    mass_axis = hist.axis.Regular(
        40, 2, 42,
        name="mass",
        label=r"Leading jet mass [GeV]"
    )

    # create & fill mass histograms
    h_mass_sig = hist.Hist(mass_axis, storage=hist.storage.Double())
    h_mass_bkg = hist.Hist(mass_axis, storage=hist.storage.Double())
    h_mass_sig.fill(mass=lead_mass_sig, weight=weights_sig)
    h_mass_bkg.fill(mass=lead_mass_bkg, weight=weights_bkg)

    # plot mass
    fig2, ax2 = plt.subplots()
    h_mass_sig.plot(ax=ax2, histtype='step', label="Signal (eta_C)")
    h_mass_bkg.plot(ax=ax2, histtype='step', label="Background (Z)")
    ax2.legend()
    ax2.set_xlabel(r'Leading jet mass [GeV]')
    ax2.set_ylabel('Entries')
    ax2.grid(True)
    out2 = os.path.join(output_dir, f'{plot_type}_comp_eta_vs_Z_lead_jet_mass.png')
    fig2.savefig(out2)
    print(f"Saved leading-jet mass comparison to {out2}")


def main():
    parser = ArgumentParser(description="Compare leading jet pT for signal vs background (unit-area normalized).")
    parser.add_argument("input_file_signal", help="Path to the input ROOT file with signal")
    parser.add_argument("input_file_bkg", help="Path to the input ROOT file with background")
    parser.add_argument(
        "--output_dir",
        default="plots",
        help="Directory to save the output plots",
    )
    args = parser.parse_args()

    print(f"Running over {args.input_file_signal}, {args.input_file_bkg}")

    # Extract plot type from filename
    match = re.search(r"delphes_output_(.*?)\.root", args.input_file_signal)
    if not match:
        raise ValueError("Plot type not found in signal filename.")
    plot_type = match.group(1)

    # Load events
    events_sig = load_events(args.input_file_signal)
    events_bkg = load_events(args.input_file_bkg)

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Make the normalized plot
    plot(events_sig, events_bkg, plot_type, args.output_dir)

if __name__ == "__main__":
    main()
