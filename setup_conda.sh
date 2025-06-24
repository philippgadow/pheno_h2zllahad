#!/bin/bash

# Check if micromamba is installed
if [ ! -f ./bin/micromamba ]; then
    echo "micromamba could not be found, installing micromamba..."
    mkdir -p ./bin

    OS=$(uname -s)
    ARCH=$(uname -m)

    echo "OS: $OS"
    echo "ARCH: $ARCH"

    if [ "$OS" = "Linux" ]; then
        if [ "$ARCH" = "x86_64" ]; then
            URL="https://micro.mamba.pm/api/micromamba/linux-64/latest"
        elif [ "$ARCH" = "aarch64" ]; then
            URL="https://micro.mamba.pm/api/micromamba/linux-aarch64/latest"
        elif [ "$ARCH" = "ppc64le" ]; then
            URL="https://micro.mamba.pm/api/micromamba/linux-ppc64le/latest"
        else
            echo "Unsupported architecture: $ARCH"
            exit 1
        fi
    elif [ "$OS" = "Darwin" ]; then
        if [ "$ARCH" = "x86_64" ]; then
            URL="https://micro.mamba.pm/api/micromamba/osx-64/latest"
        elif [ "$ARCH" = "arm64" ]; then
            URL="https://micro.mamba.pm/api/micromamba/osx-arm64/latest"
        else
            echo "Unsupported architecture: $ARCH"
            exit 1
        fi
    else
        echo "Unsupported operating system: $OS"
        exit 1
    fi

    curl -Ls $URL | tar -xvj -C ./ bin/micromamba
fi

export MAMBA_ROOT_PREFIX=$PWD/micromamba
export MAMBA_EXE=$PWD/bin/micromamba

# Initialize shell correctly
if [ -n "$ZSH_VERSION" ]; then
    echo "Initializing micromamba for zsh"
    eval "$($MAMBA_EXE shell hook --shell zsh)"
elif [ -n "$BASH_VERSION" ]; then
    echo "Initializing micromamba for bash"
    eval "$($MAMBA_EXE shell hook --shell bash)"
else
    echo "Unsupported shell"
    exit 1
fi

# Create environment if it doesn't exist
if ! $MAMBA_EXE env list | grep -q "micromamba_h2zllahad"; then
    echo "Creating micromamba environment"
    $MAMBA_EXE create -y --name micromamba_h2zllahad python=3.9
fi

# Activate environment properly
micromamba activate micromamba_h2zllahad
micromamba install -y conda-forge::subversion conda-forge::fortran-compiler conda-forge::sed conda-forge::libzip conda-forge::cmake
micromamba install -y conda-forge::root
micromamba install -y conda-forge::pythia8 conda-forge::hepmc2 conda-forge::hepmc3 conda-forge::lhapdf
micromamba install -y conda-forge::delphes
micromamba install -y conda-forge::fastjet
pip install -r requirements.txt
python -m pip install fastjet
