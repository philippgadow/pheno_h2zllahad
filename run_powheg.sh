#!/bin/bash

# Set output directory
OUTPUTDIR=output/ggH_2HDM

# Create output directory
mkdir -p ${OUTPUTDIR}

# Running the event generation
echo "Running ggH 2HDM POWHEG process to generate events"
pushd generators/POWHEG-BOX-V2/gg_H_2HDM
    pushd testrun-lhc-h
        # Run the POWHEG process
        ./../pwhg_main
        # Move the LHE file to the output directory
        mv pwgevents.lhe ../../../../${OUTPUTDIR}/pwgevents.lhe
    popd
popd
