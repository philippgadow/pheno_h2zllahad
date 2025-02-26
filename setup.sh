#!/bin/bash

# check if conda exists, otherwise install miniconda
if ! conda --version >/dev/null 2>&1; then
    echo 'Error: conda is not installed.' >&2
    echo 'Installing miniconda...'
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b
    rm Miniconda3-latest-Linux-x86_64.sh
    export PATH="$HOME/miniconda3/bin:$PATH"
fi

if ! conda env list | grep -q "h2zllahad"; then
    conda create -y --name h2zllahad python=3.9
fi

conda activate h2zllahad
    conda install -y conda-forge::subversion conda-forge::fortran-compiler
    conda install -y conda-forge::pythia8 conda-forge::hepmc2 conda-forge::hepmc3 conda-forge::lhapdf
