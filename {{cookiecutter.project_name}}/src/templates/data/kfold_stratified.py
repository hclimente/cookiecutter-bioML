#!/usr/bin/env python
"""
Input variables:
    - DATA: path of a numpy array with x.
    - SEED: random seed.
Output files:
    - xy_train.npy
    - xy_test.npy
"""
from itertools import islice

import numpy as np
from sklearn.model_selection import StratifiedKFold

split = int("${I}")
splits = int("${SPLITS}")

with open("${DATA}", "rb") as a_file:
    input_data = np.load(a_file, allow_pickle=True)
    X = input_data["X"]
    y = input_data["Y"]
    genes = input_data["genes"]

skf = StratifiedKFold(n_splits=splits)
train_index, test_index = next(islice(skf.split(X, y), split, None))

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

np.savez("Xy_train.npz", X=X_train, Y=y_train, genes=genes)
np.savez("Xy_test.npz", X=X_test, Y=y_test, genes=genes)
