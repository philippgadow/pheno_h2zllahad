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

    # Copy LHE input files
    cp ../../${OUTPUTDIR}/pwgevents.lhe input.lhe

    # Run pythia
    ./main_PhPy8_HZetac
    ./main_PhPy8_HZJpsi

    # Move the HepMC output files to the output directory
    mv *.hepmc ../../${OUTPUTDIR}/

    # Clean up
    rm input.lhe
popd
