import os
import re
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, DelphesSchema  # type: ignore
import hist


def load_events(filename):
    events = NanoEventsFactory.from_root(
        filename,
        schemaclass=DelphesSchema,
        entry_stop=1000,
        metadata={"exclude": ["Particle.fBits", "Particle.fUniqueID"]},
    ).events()
    return events


def extract_plot_type(filename):
    match = re.search(r"delphes_output_(.*?)\.root", filename)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract plot type from: {filename}")


def make_plot(data_dict, xlabel, filename, bins, range_, ylabel="Entries"):
    hep.style.use(hep.style.CMS)
    fig, ax = plt.subplots()
    for label, values in data_dict.items():
        h = hist.Hist(hist.axis.Regular(bins, *range_, name=xlabel, label=label))
        h.fill(values)
        h.plot(ax=ax, histtype='step', label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    fig.savefig(filename)
    plt.close(fig)


def plot_all(input_files, output_dir):
    # PID dictionaries
    data = {
        "Z_mass": {},
        "etac_mass": {},
        "etac_pt": {},
        "etac_eta": {},
        "jpsi_mass": {},
        "jpsi_pt": {},
        "jpsi_eta": {},
        "A_mass": {},
        "A_pt": {},
        "A_eta": {},
    }

    for input_file in input_files:
        plot_type = extract_plot_type(input_file)
        label = plot_type
        print(f"Processing {input_file} as {label}")

        events = load_events(input_file)
        particles = events.Particle

        # Z boson (PID 23)
        particles_Z = particles[particles.PID == 23]
        if ak.num(particles_Z, axis=0).compute().sum() > 0:
            mass_Z = ak.flatten(particles_Z.mass.compute()).to_numpy()
            data["Z_mass"][label] = mass_Z

        # eta_c (PID 441)
        particles_441 = particles[particles.PID == 441]
        if ak.num(particles_441, axis=0).compute().sum() > 0:
            data["etac_mass"][label] = ak.flatten(particles_441.mass.compute()).to_numpy()
            data["etac_pt"][label] = ak.flatten(particles_441.PT.compute()).to_numpy()
            data["etac_eta"][label] = ak.flatten(particles_441.Eta.compute()).to_numpy()

        # J/psi (PID 443)
        particles_443 = particles[particles.PID == 443]
        if ak.num(particles_443, axis=0).compute().sum() > 0:
            data["jpsi_mass"][label] = ak.flatten(particles_443.mass.compute()).to_numpy()
            data["jpsi_pt"][label] = ak.flatten(particles_443.PT.compute()).to_numpy()
            data["jpsi_eta"][label] = ak.flatten(particles_443.Eta.compute()).to_numpy()

        # A (PID 36)
        particles_36 = particles[particles.PID == 36]
        if ak.num(particles_36, axis=0).compute().sum() > 0:
            data["A_mass"][label] = ak.flatten(particles_36.mass.compute()).to_numpy()
            data["A_pt"][label] = ak.flatten(particles_36.PT.compute()).to_numpy()
            data["A_eta"][label] = ak.flatten(particles_36.Eta.compute()).to_numpy()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot everything
    if data["Z_mass"]:
        make_plot(data["Z_mass"], "Invariant Mass [GeV]", f"{output_dir}/Z_mass_overlay.png", 100, (76, 106))

    if data["etac_mass"]:
        make_plot(data["etac_mass"], "Invariant Mass [GeV]", f"{output_dir}/eta_c_mass_overlay.png", 100, (2.7, 3.2))
        make_plot(data["etac_pt"], "$p_T$ [GeV]", f"{output_dir}/eta_c_pt_overlay.png", 100, (0, 100))
        make_plot(data["etac_eta"], "$\eta$", f"{output_dir}/eta_c_eta_overlay.png", 100, (-5, 5))

    if data["jpsi_mass"]:
        make_plot(data["jpsi_mass"], "Invariant Mass [GeV]", f"{output_dir}/jpsi_mass_overlay.png", 100, (3.095, 3.100))
        make_plot(data["jpsi_pt"], "$p_T$ [GeV]", f"{output_dir}/jpsi_pt_overlay.png", 100, (0, 100))
        make_plot(data["jpsi_eta"], "$\eta$", f"{output_dir}/jpsi_eta_overlay.png", 100, (-5, 5))

    if data["A_mass"]:
        make_plot(data["A_mass"], "Invariant Mass [GeV]", f"{output_dir}/A_mass_overlay.png", 100, (0, 10))
        make_plot(data["A_pt"], "$p_T$ [GeV]", f"{output_dir}/A_pt_overlay.png", 100, (0, 100))
        make_plot(data["A_eta"], "$\eta$", f"{output_dir}/A_eta_overlay.png", 100, (-5, 5))


def main():
    parser = ArgumentParser()
    parser.add_argument("input_files", nargs='+', help="Input ROOT files (space-separated)")
    parser.add_argument("--output_dir", default="plots", help="Directory to save output plots")
    args = parser.parse_args()

    print(f"Running over {len(args.input_files)} file(s)")
    plot_all(args.input_files, args.output_dir)


if __name__ == "__main__":
    main()
