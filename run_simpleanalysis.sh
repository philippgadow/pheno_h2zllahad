#!/bin/bash

# inside docker container
source /release_setup.sh
cp simpleanalysis/HZa_2018.cxx /opt/SimpleAnalysis/SimpleAnalysisCodes/src/

mkdir -p simpleanalysis/build/
cd simpleanalysis/build/
cmake /opt/SimpleAnalysis/
make 
cd ../..

./simpleanalysis/build/x86_64-centos7-gcc8-opt/bin/simpleAnalysis -a HZa2018 --input-files $PWD/atlas_truth3/mc15_13TeV.361106.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zee.DAOD_TRUTH3.root