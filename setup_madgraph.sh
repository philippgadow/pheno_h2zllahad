#!/bin/bash

MG5_URL="https://launchpad.net/mg5amcnlo/3.0/3.6.x/+download/MG5_aMC_v3.6.2.tar.gz"
MG5_TAR=$(basename $MG5_URL)
MG5_DIR=$(echo $MG5_TAR | sed 's/.tar.gz//; s/\./_/g')

pushd generators
    wget $MG5_URL
    tar xvf $MG5_TAR
    rm $MG5_TAR
    # pushd $MG5_DIR
    #     # Set pythia8 path and install interface in madgraph
    #     echo "install mg5amc_py8_interface" | ./bin/mg5_aMC
    # popd
popd
