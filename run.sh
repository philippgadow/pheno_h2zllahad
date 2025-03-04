#!/bin/bash

# Run all steps of event generation in one go

# 1) Run calculation of hard scattering process ggH with Powheg
./run_powheg.sh

# 2) Run showering and hadronization with Pythia
#    -> this is where the decay H -> Z(ll) a(had) is done
./run_pythia.sh

# 3) Run detector simulation with Delphes
./run_delphes.sh
