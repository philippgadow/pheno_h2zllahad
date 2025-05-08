#!/bin/bash

# Set output directory
OUTPUTDIR=output/Zjets

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
    compile_pythia main_MG5Py8_Zjets.cc

    # Copy LHE and decompress input files
    cp ../../${OUTPUTDIR}/unweighted_events.lhe.gz unweighted_events.lhe.gz
    gzip -d unweighted_events.lhe.gz
    mv unweighted_events.lhe input.lhe

    # Run pythia for SM processes
    ./main_MG5Py8_Zjets

    # Move the HepMC output files to the output directory
    mv *.hepmc ../../${OUTPUTDIR}/

    # Clean up
    rm input.lhe
popd
