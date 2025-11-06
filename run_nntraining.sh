#!/bin/bash

CONVERTER=analysis/nn_training/convert_to_h5.py
INSPECTOR=analysis/nn_training/inspect_hdf5.py
PLOTTER=analysis/nn_training/plot_features.py
REGRESSION=analysis/nn_training/train_regression_pytorch.py
OUTPUT_DIR=nn_training_input
PLOTS_DIR=nn_plots
REG_OUTPUT=nn_training_output/regression

mkdir -p "${OUTPUT_DIR}" "${PLOTS_DIR}"  "${REG_OUTPUT}"

echo "[1/3] Converting Delphes ROOT files to HDF5..."
if [ ! -f "${OUTPUT_DIR}/jet_data.h5" ]; then
    python "${CONVERTER}" -i delphes -o "${OUTPUT_DIR}"
else
    echo "HDF5 file already exists, skipping conversion..."
fi

echo "[2/3] Inspecting generated HDF5 files..."
python "${INSPECTOR}" --summary "${OUTPUT_DIR}/jet_data.h5"

echo "[3/3] Plotting feature distributions across all samples..."
python "${PLOTTER}" --input-h5 "${OUTPUT_DIR}/jet_data.h5" --output-dir "${PLOTS_DIR}"

# train regression network
python ${REGRESSION} \
  --input-h5 "${OUTPUT_DIR}/jet_data.h5" \
  --output-dir "$REG_OUTPUT"
