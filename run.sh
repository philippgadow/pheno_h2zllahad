#!/bin/bash

# Run all steps of event generation in one go

# 1) Run calculation of hard scattering process ggH with Powheg and MadGraph
./run_powheg.sh
./run_madgraph.sh

# 2) Run showering and hadronization with Pythia
#    -> this is where the decay H -> Z(ll) a(had) is done
./run_pythia_signal.sh
./run_pythia_zjets.sh

# 3) Run detector simulation with Delphes
./run_delphes.sh
