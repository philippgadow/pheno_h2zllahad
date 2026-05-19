#!/bin/bash


# Run all steps of event generation in one go

# 1) Run hard-scattering generation with POWHEG
./run_powheg.sh

# Optional legacy MG5 Z+jets generation (disabled by default)
if [[ "${USE_MG5_ZJETS:-0}" == "1" ]]; then
	./run_madgraph.sh
fi

# 2) Run showering and hadronization with Pythia
#    -> this is where the decay H -> Z(ll) a(had) is done
./run_pythia_signal.sh
./run_pythia_zjets.sh

# 3) Run detector simulation with Delphes
./run_delphes.sh
