#!/bin/bash

# Set output directory
OUTPUTDIR=output/Zjets

# install PDF sets with LHAPDF when available under this exact alias
lhapdf install CTEQ6L1 || true

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
    compile_pythia main_PowhegPy8_Zjets.cc

    run_powheg_lhe() {
        local lhe_input=$1
        local hepmc_output=$2
        echo "Showering ${lhe_input} -> ${hepmc_output}"
        ./main_PowhegPy8_Zjets "${lhe_input}" "${hepmc_output}" "${ZJETS_PYTHIA_MAXEVENTS:-0}"
    }

    # POWHEG channels (preferred): Zee and Zmumu
    if [[ -f ../../${OUTPUTDIR}/pwgevents_Zee_powheg.lhe ]]; then
        run_powheg_lhe ../../${OUTPUTDIR}/pwgevents_Zee_powheg.lhe hepmc_output_Zjets_Zee.hepmc
    fi

    if [[ -f ../../${OUTPUTDIR}/pwgevents_Zmumu_powheg.lhe ]]; then
        run_powheg_lhe ../../${OUTPUTDIR}/pwgevents_Zmumu_powheg.lhe hepmc_output_Zjets_Zmumu.hepmc
    fi

    # Move the HepMC output files to the output directory
    mv *.hepmc ../../${OUTPUTDIR}/

    # Clean up
    rm -f input.lhe
popd
