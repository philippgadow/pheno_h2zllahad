#!/bin/bash

# Set output directory
OUTPUTDIR=output/ggH_2HDM

# Set the cards directory
CARDS=micromamba/envs/micromamba_h2zllahad/cards

# Function to run Delphes
run_delphes() {
    local hepmc_file=$1
    local name_identifier=$(basename "$hepmc_file" .hepmc)
    name_identifier=${name_identifier#hepmc_output_}
    local output_file="${OUTPUTDIR}/delphes_output_${name_identifier}.root"

    rm -f "$output_file"
    DelphesHepMC2 ${CARDS}/delphes_card_CMS.tcl "$output_file" "$hepmc_file"
}


# Run Delphes with CMS card
run_delphes ${OUTPUTDIR}/hepmc_output_HZetac.hepmc
run_delphes ${OUTPUTDIR}/hepmc_output_HZJpsi.hepmc
