#!/bin/bash

# Set output directory
OUTPUTDIR=output/Zjets

# Create output directory
mkdir -p ${OUTPUTDIR}

# Running the event generation
echo "Running Z+jets in madgraph to generate events"
pushd generators/MG5_aMC_v3_6_2
        # Run MadGraph
        ./bin/mg5_aMC ../mg5cards/zjets.process
        # Copy the LHE file to the output directory
        cp ZJets/Events/unweighted_events.lhe.gz ../../${OUTPUTDIR}/unweighted_events.lhe.gz
popd
