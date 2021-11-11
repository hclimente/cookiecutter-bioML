#!/usr/bin/env python
"""
Input variables:
    - PARAMS: model parameters
    - Y_TEST: path to numpy array with validation Y vector.
    - Y_PRED: path to numpy array with prediction vector.
Output files:
    - prediction_stats: path to a single-line tsv with the TSV results.
"""

import numpy as np
from sklearn.metrics import mean_squared_error

import utils as u

_, y, _ = u.read_data("${TEST_NPZ}", "")
y_pred = np.load("${PRED_NPZ}")["preds"]

if len(y_pred):
    mse = mean_squared_error(y, y_pred)
else:
    mse = "NA"

u.save_analysis_tsv(run="${PARAMS}", metric=["mse"], value=[mse])
