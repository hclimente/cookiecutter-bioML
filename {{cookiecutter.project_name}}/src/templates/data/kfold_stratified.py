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

import utils as u

# Read data
############################
X, y, featnames = u.read_data("${DATA_NPZ}")

# Split data
############################
split = int("${I}")
splits = int("${SPLITS}")

skf = StratifiedKFold(n_splits=splits)
train_index, test_index = next(islice(skf.split(X, y), split, None))

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

# Save data
############################
np.savez("Xy_train.npz", X=X_train, y=y_train, featnames=featnames)
np.savez("Xy_test.npz", X=X_test, y=y_test, featnames=featnames)
