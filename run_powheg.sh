#!/bin/bash

# -----------------------------------------------------------------------------
# ggH signal production (existing workflow)
# -----------------------------------------------------------------------------
SIGNAL_OUTPUTDIR=output/ggH_2HDM
mkdir -p "${SIGNAL_OUTPUTDIR}"

echo "Running ggH 2HDM POWHEG process to generate events"
pushd generators/POWHEG-BOX-V2/gg_H_2HDM
    pushd testrun-lhc-h
        ./../pwhg_main
        mv pwgevents.lhe ../../../../${SIGNAL_OUTPUTDIR}/pwgevents.lhe
    popd
popd

# -----------------------------------------------------------------------------
# Z+jets background production with POWHEG Z process (ATLAS-inspired settings)
# -----------------------------------------------------------------------------
ZJETS_OUTPUTDIR=output/Zjets
POWHEG_Z_DIR=generators/POWHEG-BOX-V2/Z
ZJETS_EVENTS=${POWHEG_ZJETS_EVENTS:-11000}

mkdir -p "${ZJETS_OUTPUTDIR}"

run_powheg_z_channel() {
    local decaymode="$1"
    local channel_tag="$2"
    local run_dir="${POWHEG_Z_DIR}/testrun-lhc-13TeV-z${channel_tag}-aznlo"

    mkdir -p "${run_dir}"

    cat > "${run_dir}/powheg.input" <<EOF
! Z -> ${channel_tag}, POWHEG Z process, AZNLO-inspired setup
vdecaymode ${decaymode}

numevts ${ZJETS_EVENTS}
ih1 1
ih2 1
ebeam1 6500d0
ebeam2 6500d0

! historic CT10 central value used in ATLAS AZNLO setup
lhans1 10800
lhans2 10800

! allow grid reuse if files exist in this run directory
use-old-grid 1
use-old-ubound 1

! integration setup (ATLAS-inspired)
ncall1 400000
itmx1 10
ncall2 400000
itmx2 20
nubound 1000000
foldcsi 2
foldphi 1
foldy 1

! generation / EW scheme inputs
ptsqmin 4
mass_low 60
mass_high 1d20
alphaem 0.00781653
Wmass 79.958059
EOF

    echo "Running POWHEG Z process for Z->${channel_tag} background"
    pushd "${run_dir}"
        ../pwhg_main
        mv pwgevents.lhe ../../../../${ZJETS_OUTPUTDIR}/pwgevents_Z${channel_tag}_powheg.lhe
    popd
}

# vdecaymode in POWHEG Z process: 1 -> e+e-, 2 -> mu+mu-
run_powheg_z_channel 1 ee
run_powheg_z_channel 2 mumu
