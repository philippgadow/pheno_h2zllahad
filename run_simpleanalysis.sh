#!/bin/bash

# inside docker container
source /release_setup.sh
cp simpleanalysis/HZa_2018.cxx /opt/SimpleAnalysis/SimpleAnalysisCodes/src/

mkdir -p simpleanalysis/build/
cd simpleanalysis/build/
cmake /opt/SimpleAnalysis/
make 

# simpleAnalysis [-a listOfAnalysis] <inputFile1> [inputFile2]...