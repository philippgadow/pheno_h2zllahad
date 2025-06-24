import uproot
import awkward as ak
import numpy as np
import fastjet


def getJetsFromATLAS(file, events=10):
    """
    Read stable truth particles from ATLAS TRUTH1 ROOT file and cluster jets.
    """
    tree = file["CollectionTree"]

    # Read particle-level data
    truth = tree.arrays(
        [
            "TruthParticlesAuxDyn.px",
            "TruthParticlesAuxDyn.py",
            "TruthParticlesAuxDyn.pz",
            "TruthParticlesAuxDyn.e",
            "TruthParticlesAuxDyn.pdgId",
            "TruthParticlesAuxDyn.status"
        ],
        entry_stop=events
    )

    # Selection: status == 1 and not a neutrino
    # status == 1 means visible particles at the end of the showering/hadronisation chain
    status = truth["TruthParticlesAuxDyn.status"]
    pdgId = truth["TruthParticlesAuxDyn.pdgId"]
    abs_pdgId = abs(pdgId)
    neutrino_mask = (abs_pdgId == 12) | (abs_pdgId == 14) | (abs_pdgId == 16)
    mask = (status == 1) & (~neutrino_mask)

    # ATLAS stores in MeV, convert to GeV
    px = truth["TruthParticlesAuxDyn.px"][mask] * 0.001
    py = truth["TruthParticlesAuxDyn.py"][mask] * 0.001
    pz = truth["TruthParticlesAuxDyn.pz"][mask] * 0.001
    e = truth["TruthParticlesAuxDyn.e"][mask] * 0.001

    particles = ak.zip(
        {
            "px": px,
            "py": py,
            "pz": pz,
            "E": e
        },
        with_name="Momentum4D"
    )

    # Convert to fastjet inputs (per event)
    fj_inputs = [
        [fastjet.PseudoJet(float(p.px), float(p.py), float(p.pz), float(p.E)) for p in event]
        for event in particles
    ]

    # Cluster jets for first event
    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    jets = []
    for event_idx, event_particles in enumerate(fj_inputs):
        if len(event_particles) > 0:
            cs = fastjet.ClusterSequence(event_particles, jet_def)
            event_jets = cs.inclusive_jets(ptmin=20.0)
            for jet in event_jets:
                jet.set_user_index(event_idx)
            jets.extend(event_jets)

    return jets


def getJetsFromDelphes(file, events=10):
    """
    Read stable truth particles from Delphes ROOT file and cluster jets.
    """
    tree = file["Delphes"]

    # Read particle-level data
    particles_data = tree.arrays(
        [
            "Particle.Px",
            "Particle.Py", 
            "Particle.Pz",
            "Particle.E",
            "Particle.PID",
            "Particle.Status"
        ],
        entry_stop=events
    )

    # Selection: status == 1 and not a neutrino
    # In Delphes, status == 1 also means stable particles
    status = particles_data["Particle.Status"]
    pdgId = particles_data["Particle.PID"]
    abs_pdgId = abs(pdgId)
    neutrino_mask = (abs_pdgId == 12) | (abs_pdgId == 14) | (abs_pdgId == 16)
    mask = (status == 1) & (~neutrino_mask)

    px = particles_data["Particle.Px"][mask]
    py = particles_data["Particle.Py"][mask]
    pz = particles_data["Particle.Pz"][mask]
    e = particles_data["Particle.E"][mask]

    particles = ak.zip(
        {
            "px": px,
            "py": py,
            "pz": pz,
            "E": e
        },
        with_name="Momentum4D"
    )

    # Convert to fastjet inputs (per event)
    fj_inputs = [
        [fastjet.PseudoJet(float(p.px), float(p.py), float(p.pz), float(p.E)) for p in event]
        for event in particles
    ]

    # Cluster jets for first event
    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    jets = []
    for event_idx, event_particles in enumerate(fj_inputs):
        if len(event_particles) > 0:
            cs = fastjet.ClusterSequence(event_particles, jet_def)
            event_jets = cs.inclusive_jets(ptmin=20.0)
            for jet in event_jets:
                jet.set_user_index(event_idx)
            jets.extend(event_jets)

    return jets


if __name__ == "__main__":
    # ATLAS file
    atlas_file = uproot.open("atlas_truth1/mc15_13TeV.600973.PhPy8EG_HZetac.DAOD_TRUTH1.root")
    atlas_jets = getJetsFromATLAS(atlas_file)
    
    print("ATLAS Jets:")
    for j in atlas_jets:
        print(f"Jet: pt = {j.pt():.2f}, eta = {j.eta():.2f}, phi = {j.phi():.2f}, m = {j.m():.2f}")
    
    # Delphes file
    delphes_file = uproot.open("output/ggH_2HDM/delphes_output_HZetac.root")
    delphes_jets = getJetsFromDelphes(delphes_file)
    
    print("\nDelphes Jets:")
    for j in delphes_jets:
        print(f"Jet: pt = {j.pt():.2f}, eta = {j.eta():.2f}, phi = {j.phi():.2f}, m = {j.m():.2f}")