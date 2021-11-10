#!/usr/bin/env python
"""
Input variables:
    - TRAIN: path of a numpy array with x.
    - TEST: path of a numpy array with x.
    - NET: pickled file with the adjacency matrix
Output files:
    - selected.npy
"""

import numpy as np
from scipy.sparse import load_npz

from galore import Galore


np.random.seed(0)

train_data = np.load("${TRAIN}", allow_pickle=True)

X_train = train_data["X"]
y_train = train_data["Y"]

A = load_npz("${NET}")

# train
lambda_1 = float("${LAMBDA_1}")
lambda_2 = float("${LAMBDA_2}")
galore = Galore(A, lambda_1, lambda_2)
galore.fit(X_train, y_train)

# test
test_data = np.load("${TEST}")

X_test = test_data["X"]

y_proba = galore.predict_proba(X_test)
np.save("y_proba.npy", y_proba[:, 1])
