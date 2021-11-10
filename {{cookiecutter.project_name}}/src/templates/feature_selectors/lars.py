#!/usr/bin/env python
"""
Input variables:
  - TRAIN_NPZ: path to a .npz file containing three elements: an X matrix, a y vector,
    and a featnames vector (optional)
  - PARAMS_JSON: path to a json file with the hyperparameters
    - n_nonzero_coefs
Output files:
  - selected.npz: contains the featnames of the selected features, their scores and the
    hyperparameters selected by cross-validation
  - selected.tsv: like selected.npz, but in tsv format.
"""
import numpy as np
from sklearn.linear_model import LarsCV
from sklearn.feature_selection import SelectFromModel

import utils as u

# Read data
############################
X, y, featnames = u.read_data("${TRAIN_NPZ}")
param_grid = u.read_parameters("${PARAMS_FILE}")

# Run algorithm
############################
clf = LarsCV(**param_grid)
clf.fit(X, y)

sfm = SelectFromModel(clf.best_estimator_, prefit=True)
features = np.where(sfm.get_support())[0]

# Save selected features
############################
u.save_selected_npz(features, param_grid)
u.save_selected_tsv(features, param_grid)
