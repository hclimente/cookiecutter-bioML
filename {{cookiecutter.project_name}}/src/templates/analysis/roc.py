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


def load_data(path):
    data = np.load(path, allow_pickle=True)

    X = data["X"].transpose()
    Y = data["Y"]
    genes = data["genes"]

    return X, Y, genes


X_test, y_test, genes_test = load_data("${TEST}")
y_proba = np.load("${Y_PROBA}")

try:
    auc = roc_auc_score(y_test, y_proba)
except ValueError:
    auc = "NA"

row = ["${PARAMS}", auc]

with open("prediction_stats", "w", newline="") as f_output:
    tsv_output = csv.writer(f_output, delimiter="\t")
    tsv_output.writerow(row)
