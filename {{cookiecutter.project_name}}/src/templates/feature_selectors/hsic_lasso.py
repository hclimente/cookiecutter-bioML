#!/usr/bin/env python
"""
Input variables:
    - X_TRAIN: path of a numpy array with x.
    - Y_TRAIN: path of a numpy array with y.
    - COVARS: path of a numpy array with covariates.
    - FEATNAMES: path of a numpy array with feature names.
    - MODE: regression or classification.
    - HL_SELECT: number of features to select.
    - HL_B: size of the block.
    - HL_M: number of permutations.
Output files:
    - features_hl.npy: numpy array with the 0-based index of
    the selected features.
"""

import numpy as np
from pyHSICLasso import HSICLasso

import utils as u

hl = HSICLasso()

np.random.seed(0)
hl.X_in = np.load("${X_TRAIN}").T
hl.Y_in = np.load("${Y_TRAIN}").T
hl.Y_in = np.expand_dims(hl.Y_in, 0)
hl.featname = np.load("${FEATNAMES}")
covars = np.load("${COVARS}")

# TODO parameters as dictionary
C = int("${HL_SELECT}")
block_size = int("${HL_B}")
permutations = int("${HL_M}")

try:
    hl.classification(C, B=block_size, M=permutations, covars=covars)
except MemoryError:
    u.custom_error(file="features_hl.npy", content=np.array([]))

np.save("features_hl.npy", hl.A)
