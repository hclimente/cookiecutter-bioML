#!/usr/bin/env python
"""
Input variables:
    - PARAMS: model parameters
    - TEST: path to numpy array with validation Y vector.
    - Y_PROBA: path to numpy array with prediction vector.
Output files:
    - prediction_stats: path to a single-line tsv with the TSV results.
"""

import csv
import numpy as np
from sklearn.metrics import roc_auc_score

import utils as u

_, y, _ = u.read_data("${TEST_NPZ}")
y_proba = np.load("${Y_PROBA}")["proba"]

try:
    auc = roc_auc_score(y, y_proba)
except ValueError:
    auc = "NA"

row = ["${PARAMS}", auc]

with open("prediction_stats", "w", newline="") as f_output:
    tsv_output = csv.writer(f_output, delimiter="\t")
    tsv_output.writerow(row)
