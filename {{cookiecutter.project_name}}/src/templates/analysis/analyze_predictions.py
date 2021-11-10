#!/usr/bin/env python
"""
Input variables:
    - PARAMS: model parameters
    - Y_TEST: path to numpy array with validation Y vector.
    - Y_PRED: path to numpy array with prediction vector.
Output files:
    - prediction_stats: path to a single-line tsv with the TSV results.
"""

import csv
import numpy as np
from sklearn.metrics import confusion_matrix


def load_data(path):
    data = np.load(path, allow_pickle=True)

    X = data["X"].transpose()
    Y = data["Y"]
    genes = data["genes"]

    return X, Y, genes


X_test, y_test, genes_test = load_data("${TEST}")
y_pred = np.load("${PRED}")

score = np.nan
if len(y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[-1, 1]).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

else:
    tpr = fpr = "NA"

row = ["${PARAMS}", tpr, fpr]

with open("prediction_stats", "w", newline="") as f_output:
    tsv_output = csv.writer(f_output, delimiter="\t")
    tsv_output.writerow(row)
