#!/usr/bin/env python
"""
Input variables:
  - TRAIN_NPZ: path to a .npz file containing three elements: an X matrix, a y vector,
    and a featnames vector (optional)
  - PARAMS_JSON: path to a json file with the hyperparameters
    - num_feat
    - B
    - M
    - covars
Output files:
  - selected.npz: contains the featnames of the selected features, their scores and the
    hyperparameters selected by cross-validation
  - selected.tsv: like selected.npz, but in tsv format.
"""
import numpy as np
from pyHSICLasso import HSICLasso

import utils as u

u.set_random_state()

# Read data
############################
X, y, featnames = u.read_data("${TRAIN_NPZ}")
param_grid = u.read_parameters("${PARAMS_FILE}")

# Run algorithm
############################
hl = HSICLasso()
hl.input(X, y, featnames)

try:
    hl.classification(**param_grid)
except MemoryError:
    u.custom_error(file="scores.npz", content=np.array([]))

# Save selected features
############################
u.save_selected_npz(hl.A, featnames, param_grid)
u.save_selected_tsv(hl.A, featnames, param_grid)
