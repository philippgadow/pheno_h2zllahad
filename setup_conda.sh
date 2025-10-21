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

# Use a project local condarc that only points to conda forge
export CONDARC="$PWD/condarc-project.yaml"
cat > "$CONDARC" <<'YAML'
channels:
  - conda-forge
default_channels: []
channel_priority: strict
auto_activate_base: false
always_yes: true
YAML

# Initialize shell
if [ -n "${ZSH_VERSION:-}" ]; then
    echo "Initializing micromamba for zsh"
    eval "$($MAMBA_EXE --rc-file "$CONDARC" shell hook --shell zsh)"
elif [ -n "${BASH_VERSION:-}" ]; then
    echo "Initializing micromamba for bash"
    eval "$($MAMBA_EXE --rc-file "$CONDARC" shell hook --shell bash)"
else
    echo "Unsupported shell"
    exit 1
fi

ENV_NAME="micromamba_h2zllahad"

# Create environment if it does not exist, force only conda forge
env_exists=1
if "$MAMBA_EXE" --rc-file "$CONDARC" env list --json > /dev/null 2>&1; then
    # parse JSON to find environment names (jq is not guaranteed available),
    # use python -c to parse the JSON safely
    if "$MAMBA_EXE" --rc-file "$CONDARC" env list --json | \
       python -c "import sys,json;print(any(e.get('name')==\"$ENV_NAME\" for e in json.load(sys.stdin)))" 2>/dev/null | grep -q True; then
        env_exists=0
    else
        env_exists=1
    fi
else
    # fallback to plain text parsing (escape for zsh-safe character classes)
    if "$MAMBA_EXE" --rc-file "$CONDARC" env list | grep -q "^$ENV_NAME[[:space:]]"; then
        env_exists=0
    else
        env_exists=1
    fi
fi

if [ "$env_exists" -ne 0 ]; then
    echo "Creating micromamba environment"
    "$MAMBA_EXE" --rc-file "$CONDARC" create -y --name "$ENV_NAME" --override-channels -c conda-forge python=3.9
fi

# Activate environment + install
micromamba activate "$ENV_NAME"

micromamba install -y --override-channels -c conda-forge subversion fortran-compiler sed libzip cmake wget
micromamba install -y --override-channels -c conda-forge root
micromamba install -y --override-channels -c conda-forge pythia8 hepmc2 hepmc3 lhapdf
micromamba install -y --override-channels -c conda-forge delphes
micromamba install -y --override-channels -c conda-forge fastjet
micromamba install -y --override-channels -c conda-forge pytorch

pip install -r requirements.txt
