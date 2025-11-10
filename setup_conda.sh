#!/bin/bash

setup_conda() {
    local env_name="micromamba_h2zllahad"
    local conda_packages=(
        subversion
        fortran-compiler
        sed
        libzip
        cmake
        wget
        root
        pythia8
        hepmc2
        hepmc3
        lhapdf
        delphes
        fastjet
        pytorch
        cython
        autoconf
    )
    
    if [ ! -f ./bin/micromamba ]; then
        echo "micromamba could not be found, installing micromamba..."
        mkdir -p ./bin
        local os arch url
        os=$(uname -s)
        arch=$(uname -m)
        echo "OS: $os"
        echo "ARCH: $arch"
        
        if [ "$os" = "Linux" ]; then
            case "$arch" in
                x86_64) url="https://micro.mamba.pm/api/micromamba/linux-64/latest";;
                aarch64) url="https://micro.mamba.pm/api/micromamba/linux-aarch64/latest";;
                ppc64le) url="https://micro.mamba.pm/api/micromamba/linux-ppc64le/latest";;
                *) echo "Unsupported architecture: $arch" >&2; return 1;;
            esac
        elif [ "$os" = "Darwin" ]; then
            case "$arch" in
                x86_64) url="https://micro.mamba.pm/api/micromamba/osx-64/latest";;
                arm64) url="https://micro.mamba.pm/api/micromamba/osx-arm64/latest";;
                *) echo "Unsupported architecture: $arch" >&2; return 1;;
            esac
        else
            echo "Unsupported operating system: $os" >&2
            return 1
        fi
        
        curl -Ls "$url" | tar -xvj -C ./ bin/micromamba
    fi
    
    export MAMBA_ROOT_PREFIX=$PWD/micromamba
    export MAMBA_EXE=$PWD/bin/micromamba
    export CONDARC="$PWD/condarc-project.yaml"
    
    cat > "$CONDARC" <<'YAML'
channels:
  - conda-forge
default_channels: []
channel_priority: strict
auto_activate_base: false
always_yes: true
YAML
    
    if [ -n "${ZSH_VERSION:-}" ]; then
        echo "Initializing micromamba for zsh"
        eval "$($MAMBA_EXE --rc-file "$CONDARC" shell hook --shell zsh)"
    elif [ -n "${BASH_VERSION:-}" ]; then
        echo "Initializing micromamba for bash"
        eval "$($MAMBA_EXE --rc-file "$CONDARC" shell hook --shell bash)"
    else
        echo "Unsupported shell" >&2
        return 1
    fi
    
    # Improved environment existence check
    local env_exists=1
    if "$MAMBA_EXE" --rc-file "$CONDARC" env list -n "$env_name" 2>/dev/null | grep -q "$env_name"; then
        env_exists=0
        echo "Environment '$env_name' already exists."
    else 
        echo "Environment '$env_name' does not exist."
    fi
    
    if [ "$env_exists" -ne 0 ]; then
        echo "Creating micromamba environment"
        "$MAMBA_EXE" --rc-file "$CONDARC" create -y --name "$env_name" --override-channels -c conda-forge python=3.9
    fi
    
    echo "Checking required conda packages in environment '$env_name'..."
    local existing_pkgs missing_pkgs pkg pkg_name
    existing_pkgs=$("$MAMBA_EXE" --rc-file "$CONDARC" list -n "$env_name" 2>/dev/null | awk 'NR>3 && NF {print $1}')
    missing_pkgs=()
    
    for pkg in "${conda_packages[@]}"; do
        pkg_name=${pkg%%=*}
        if ! echo "$existing_pkgs" | grep -Fxq "$pkg_name"; then
            missing_pkgs+=("$pkg")
        fi
    done
    
    if [ ${#missing_pkgs[@]} -gt 0 ]; then
        echo "Installing missing packages: ${missing_pkgs[*]}"
        "$MAMBA_EXE" --rc-file "$CONDARC" install -y -n "$env_name" --override-channels -c conda-forge "${missing_pkgs[@]}"
    else
        echo "All requested conda packages already present."
    fi
    
    micromamba activate "$env_name"
    pip install -r requirements.txt
    
    return 0
}

setup_conda "$@"