#!/bin/bash

CONVERTER=analysis/nn_training/convert_to_h5.py
INSPECTOR=analysis/nn_training/inspect_hdf5.py
PLOTTER=analysis/nn_training/plot_features.py
REGRESSION=analysis/nn_training/train_regression_pytorch.py
REG_APPLIER=analysis/nn_training/apply_regression_model.py
REG_PLOTTER=analysis/nn_training/plot_regression_outputs.py
CLASSIFIER=analysis/nn_training/train_classification_pytorch.py
OUTPUT_DIR=nn_training_input
PLOTS_DIR=nn_plots
REG_OUTPUT=nn_training_output/regression
CLS_OUTPUT=nn_training_output/classification
BASE_H5="${OUTPUT_DIR}/jet_data.h5"
AUG_H5="${OUTPUT_DIR}/jet_data_augmented.h5"

mkdir -p "${OUTPUT_DIR}" "${PLOTS_DIR}" "${REG_OUTPUT}" "${CLS_OUTPUT}"

echo "[1/7] Converting Delphes ROOT files to HDF5..."
if [ ! -f "${BASE_H5}" ]; then
    python "${CONVERTER}" -i delphes -o "${OUTPUT_DIR}"
else
    echo "HDF5 file already exists, skipping conversion..."
fi

echo "[2/7] Inspecting generated HDF5 files..."
python "${INSPECTOR}" --summary "${BASE_H5}"

echo "[3/7] Plotting feature distributions across all samples..."
python "${PLOTTER}" --input-h5 "${BASE_H5}" --output-dir "${PLOTS_DIR}"

# train regression network
echo "[4/7] Training regression network..."
python ${REGRESSION} \
  --input-h5 "${BASE_H5}" \
  --output-dir "$REG_OUTPUT"

echo "[5/7] Applying regression model back to jet_data.h5..."
python ${REG_APPLIER} \
  --input-h5 "${BASE_H5}" \
  --output-h5 "${AUG_H5}" \
  --regression-run-dir "$REG_OUTPUT" \
  --overwrite

echo "[6/7] Plotting regression outputs..."
python ${REG_PLOTTER} \
  --input-h5 "${AUG_H5}" \
  --output "${PLOTS_DIR}/regression_output_overlay.png"

echo "[7/7] Training classification network..."
python ${CLASSIFIER} \
  --input-h5 "${AUG_H5}" \
  --output-dir "$CLS_OUTPUT" \
  --features-key ghost_track_vars_with_reg
