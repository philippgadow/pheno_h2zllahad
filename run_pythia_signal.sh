#!/bin/bash

# Set output directory
OUTPUTDIR=output/ggH_2HDM

# install PDF sets with LHAPDF
lhapdf install NNPDF23_lo_as_0130_qed

# Function to compile pythia
compile_pythia() {
    local cc_file=$1
    local output_file=${cc_file%.cc}
    g++ -std=c++11 $cc_file \
        $(pythia8-config --cxxflags --libs) \
        -lHepMC \
        -o $output_file
}

pushd generators/pythia
    # Compile pythia
    compile_pythia main_PhPy8_HZetac.cc
    compile_pythia main_PhPy8_HZJpsi.cc
    compile_pythia main_PhPy8_HZA.cc

    # Copy LHE input files
    cp ../../${OUTPUTDIR}/pwgevents.lhe input.lhe

    # Run pythia for SM processes
    ./main_PhPy8_HZetac
    ./main_PhPy8_HZJpsi

    # Run BSM events: need to hack LHE file to switch SM Higgs (25) to BSM Higgs (35) for pythia decays to work
    python hack_lhe.py input.lhe

    # Run pythia for BSM processes
    ./main_PhPy8_HZA 0.5
    ./main_PhPy8_HZA 0.75
    ./main_PhPy8_HZA 1.0
    ./main_PhPy8_HZA 1.5
    ./main_PhPy8_HZA 2.0
    ./main_PhPy8_HZA 2.5
    ./main_PhPy8_HZA 3.0
    ./main_PhPy8_HZA 3.5
    ./main_PhPy8_HZA 4.0
    ./main_PhPy8_HZA 8.0

    # Move the HepMC output files to the output directory
    mv *.hepmc ../../${OUTPUTDIR}/

    # Clean up
    rm input.lhe
popd
