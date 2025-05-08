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
    # Load events from a ROOT file
    events = NanoEventsFactory.from_root(
        filename,
        schemaclass=DelphesSchema,
        entry_stop=1000,
        metadata={"exclude": ["Particle.fBits", "Particle.fUniqueID"]},

    ).events()
    return events


def plot(events, plot_type, output_dir):
    output_path_base = os.path.join(output_dir, f"{plot_type}")

    particles = events.Particle

    # Z boson mass and kinematics
    particles_Z = particles[particles.PID == 23]
    mass_Z = particles_Z.mass.compute()
    mass_Z = ak.flatten(mass_Z).to_numpy()
    label_Z = r'$Z$ (PID 23)'

    h = hist.Hist(
        hist.axis.Regular(
            100, 76, 106,
            name="Invariant Mass [GeV]",
            label=label_Z)
        )
    h.fill(mass_Z)

    hep.style.use(hep.style.CMS)
    fig, ax = plt.subplots()
    h.plot(ax=ax, histtype='step')
    ax.legend([label_Z])
    ax.set_xlabel('Invariant Mass [GeV]')
    ax.set_ylabel('Entries')
    ax.grid(True)
    fig.savefig(f'{output_path_base}_Z_mass.png')

    if "HZetac" in plot_type:
        particles_pid441 = particles[particles.PID == 441]
        label_etac = r'$\eta_{c}$ (PID 441)'

        # mass
        mass_pid441 = particles_pid441.mass.compute()
        mass_pid441 = ak.flatten(mass_pid441).to_numpy()
        h = hist.Hist(
            hist.axis.Regular(
                100, 2.7, 3.2,
                name="Invariant Mass [GeV]",
                label=label_etac)
            )
        h.fill(mass_pid441)

        hep.style.use(hep.style.CMS)
        fig, ax = plt.subplots()
        h.plot(ax=ax, histtype='step')
        ax.legend(label_etac)
        ax.set_xlabel('Invariant Mass [GeV]')
        ax.set_ylabel('Entries')
        ax.grid(True)
        fig.savefig(f'{output_path_base}_eta_c_mass.png')

        # transverse momentum
        pt_pid441 = particles_pid441.PT.compute()
        pt_pid441 = ak.flatten(pt_pid441).to_numpy()
        h = hist.Hist(
            hist.axis.Regular(
                100, 0, 100,
                name="$p_{T}$ [GeV]",
                label=label_etac)
            )
        h.fill(pt_pid441)
        hep.style.use(hep.style.CMS)
        fig, ax = plt.subplots()
        h.plot(ax=ax, histtype='step')
        ax.legend([label_etac])
        ax.set_xlabel('$p_{T}$ [GeV]')
        ax.set_ylabel('Entries')
        ax.grid(True)
        fig.savefig(f'{output_path_base}_eta_c_pt.png')

        # eta
        eta_pid441 = particles_pid441.Eta.compute()
        eta_pid441 = ak.flatten(eta_pid441).to_numpy()
        h = hist.Hist(
            hist.axis.Regular(
                100, -5, 5,
                name="Pseudo-Rapidity",
                label=label_etac)
            )
        h.fill(eta_pid441)
        hep.style.use(hep.style.CMS)
        fig, ax = plt.subplots()
        h.plot(ax=ax, histtype='step')
        ax.legend([label_etac])
        ax.set_xlabel('$\eta$')
        ax.set_ylabel('Entries')
        ax.grid(True)
        fig.savefig(f'{output_path_base}_eta_c_eta.png')

    if "HZJpsi" in plot_type:
        particles_pid443 = particles[particles.PID == 443]
        label_jpsi = r'$J/\psi$ (PID 443)'

        # mass
        mass_pid443 = particles_pid443.mass.compute()
        mass_pid443 = ak.flatten(mass_pid443, axis=-1).to_numpy()
        h = hist.Hist(
            hist.axis.Regular(
                100, 3.095, 3.100,
                name="Invariant Mass [GeV]",
                label=label_jpsi)
        )
        h.fill(mass_pid443)
        hep.style.use(hep.style.CMS)
        fig, ax = plt.subplots()
        h.plot(ax=ax, histtype='step')
        ax.legend([label_jpsi])
        ax.set_xlabel('Invariant Mass [GeV]')
        ax.set_ylabel('Entries')
        ax.grid(True)
        fig.savefig(f'{output_path_base}_jpsi_mass.png')

        # transverse momentum
        pt_pid443 = particles_pid443.PT.compute()
        pt_pid443 = ak.flatten(pt_pid443).to_numpy()
        h = hist.Hist(
            hist.axis.Regular(
                100, 0, 100,
                name="Transverse Momentum [GeV]",
                label=label_jpsi)
        )
        h.fill(pt_pid443)
        hep.style.use(hep.style.CMS)
        fig, ax = plt.subplots()
        h.plot(ax=ax, histtype='step')
        ax.legend([label_jpsi])
        ax.set_xlabel('$p_{T}$ [GeV]')
        ax.set_ylabel('Entries')
        ax.grid(True)
        fig.savefig(f'{output_path_base}_jpsi_pt.png')

        # eta
        eta_pid443 = particles_pid443.Eta.compute()
        eta_pid443 = ak.flatten(eta_pid443).to_numpy()
        h = hist.Hist(
            hist.axis.Regular(
                100, -5, 5,
                name="Pseudo-Rapidity",
                label=label_jpsi)
        )
        h.fill(eta_pid443)
        hep.style.use(hep.style.CMS)
        fig, ax = plt.subplots()
        h.plot(ax=ax, histtype='step')
        ax.legend([label_jpsi])
        ax.set_xlabel('$\eta$')
        ax.set_ylabel('Entries')
        ax.grid(True)
        fig.savefig(f'{output_path_base}_jpsi_eta.png')

    if "HZA" in plot_type:
        particles_pid36 = particles[particles.PID == 36]
        label_A = r'$A$ (PID 36)'
        # mass
        mass_pid36 = particles_pid36.mass.compute()
        mass_pid36 = ak.flatten(mass_pid36).to_numpy()
        h = hist.Hist(
            hist.axis.Regular(
                100, 0, 10,
                name="Invariant Mass [GeV]",
                label=label_A)
        )
        h.fill(mass_pid36)
        hep.style.use(hep.style.CMS)
        fig, ax = plt.subplots()
        h.plot(ax=ax, histtype='step')
        ax.legend([label_A])
        ax.set_xlabel('Invariant Mass [GeV]')
        ax.set_ylabel('Entries')
        ax.grid(True)
        fig.savefig(f'{output_path_base}_A_mass.png')

        # transverse momentum
        pt_pid36 = particles_pid36.PT.compute()
        pt_pid36 = ak.flatten(pt_pid36).to_numpy()
        h = hist.Hist(
            hist.axis.Regular(
                100, 0, 100,
                name="Transverse Momentum [GeV]",
                label=label_A)
        )
        h.fill(pt_pid36)
        hep.style.use(hep.style.CMS)
        fig, ax = plt.subplots()
        h.plot(ax=ax, histtype='step')
        ax.legend([label_A])
        ax.set_xlabel('$p_{T}$ [GeV]')
        ax.set_ylabel('Entries')
        ax.grid(True)
        fig.savefig(f'{output_path_base}_A_pt.png')

        # eta
        eta_pid36 = particles_pid36.Eta.compute()
        eta_pid36 = ak.flatten(eta_pid36).to_numpy()
        h = hist.Hist(
            hist.axis.Regular(
                100, -5, 5,
                name="Pseudo-Rapidity",
                label=label_A)
        )
        h.fill(eta_pid36)
        hep.style.use(hep.style.CMS)
        fig, ax = plt.subplots()
        h.plot(ax=ax, histtype='step')
        ax.legend([label_A])
        ax.set_xlabel('$\eta$')
        ax.set_ylabel('Entries')
        ax.grid(True)
        fig.savefig(f'{output_path_base}_A_eta.png')


def main():
    parser = ArgumentParser()
    parser.add_argument("input_file", help="Path to the input ROOT file")
    parser.add_argument(
        "--output_dir",
        default="plots",
        help="Directory to save the output plots",
    )
    args = parser.parse_args()

    print(f"Running over {args.input_file}")

    # extract plot_type from filename
    plot_type = re.search(r"delphes_output_(.*?)\.root", args.input_file)
    if plot_type:
        plot_type = plot_type.group(1)
    else:
        raise ValueError("Plot type not found in filename.")
    events = load_events(args.input_file)
    
    # create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    plot(events, plot_type, args.output_dir)

if __name__ == "__main__":
    main()
