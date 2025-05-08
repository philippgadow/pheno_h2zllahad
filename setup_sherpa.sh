#!/bin/bash

# Install sherpa
pushd generators
wget https://gitlab.com/sherpa-team/sherpa/-/archive/v3.0.1/sherpa-v3.0.1.tar.gz
tar xvf sherpa-v3.0.1.tar.gz
rm sherpa-v3.0.1.tar.gz

    pushd sherpa-v3.0.1
    cmake -S . -B build/
    cmake --build build/ -j 8
    cmake --install build/

    popd
popd 