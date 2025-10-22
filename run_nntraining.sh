#!/bin/bash

# convert to hdf5 format
python analysis/nn_training/convert_to_h5.py -i delphes -o nn_training_input

# inspect files
python analysis/nn_training/inspect_hdf5.py nn_training_input/jet_data.h5

# plot inputs to training
python analysis/nn_training/plot_features.py --input-h5 nn_training_input/jet_data.h5 --output-dir plots_nn

# train network
