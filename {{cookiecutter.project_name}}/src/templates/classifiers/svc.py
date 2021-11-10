#!/usr/bin/env python
"""
Input variables:
    - SELECTED_FEATURES: path to the selected features.
    - TRAIN: path to numpy array with train X matrix.
    - TEST: path to numpy array with train Y vector.
Output files:
    - y_pred.npy
"""
import sys
import traceback

import numpy as np
from sklearn import svm


def load_data(path):
    data = np.load(path, allow_pickle=True)

    X = data["X"]
    Y = data["Y"]
    genes = data["genes"]

    return X, Y, genes


X_train, y_train, genes = load_data("${TRAIN}")
X_test, y_test, _ = load_data("${TEST}")
with open("${SELECTED_FEATURES}") as f:
    selected = f.read().splitlines()
    selected = selected[1:]

selected = [x in selected for x in genes]

# filter matrix by extracted features
try:
    if not sum(selected):
        raise IndexError("No selected features")
    X_train = X_train[:, selected]
    X_test = X_test[:, selected]
except IndexError:
    traceback.print_exc()
    np.save("y_proba.npy", np.array([]))
    sys.exit(77)

# cv, build model and predict
clf = svm.SVC(gamma="scale", class_weight="balanced", random_state=42, probability=True)

clf.fit(X_train, y_train)

y_proba = clf.predict_proba(X_test)
np.save("y_proba.npy", y_proba[:, 1])
