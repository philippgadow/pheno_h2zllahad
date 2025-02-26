#!/bin/bash

# mapyde run madgraph configs/mapyde/user.toml

rm main

g++ -std=c++11 configs/pythia/main.cc \
    -I$CONDA_PREFIX/include \
    $(pythia8-config --cxxflags --libs) \
    -L$CONDA_PREFIX/lib -lHepMC \
    -o main
./main

