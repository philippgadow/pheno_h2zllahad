#!/bin/bash

# SIMPLEANADIR=/opt/SimpleAnalysis
SIMPLEANADIR=$PWD/simpleanalysis/simple-analysis/
# inside docker container
source /release_setup.sh
cp simpleanalysis/HZa_2018.cxx $SIMPLEANADIR/SimpleAnalysisCodes/src/

mkdir -p simpleanalysis/build/
cd simpleanalysis/build/
cmake $SIMPLEANADIR/
make 
source x*/setup.sh
cd ../..

# run outside of the container first
# python simpleanalysis/Delphes2SA.py output/ggH_2HDM/delphes_output_HZA_mA8.00GeV.root sa_delphes_output_HZA_mA8.00GeV.root

# ./simpleanalysis/build/x86_64-centos7-gcc8-opt/bin/simpleAnalysis -a HZa2018 --input-files $PWD/atlas_truth3/mc15_13TeV.361106.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zee.DAOD_TRUTH3.root
./simpleanalysis/build/x86_64-centos7-gcc8-opt/bin/simpleAnalysis -a HZa2018 --input-files $PWD/sa_delphes_output_HZA_mA8.00GeV.root