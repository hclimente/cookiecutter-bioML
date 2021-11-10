#!/usr/bin/env python
"""
Input variables:
  - TRAIN_NPZ: path to a .npz file containing the train set. It must contain three
    elements: an X matrix, a y vector, and a featnames vector (optional)
  - TEST_NPZ: path to a .npz file containing the test set. It must contain three
    elements: an X matrix, a y vector, and a featnames vector (optional)
  - PARAMS_JSON: path to a json file with the hyperparameters
    - None
Output files:
  - y_pred.npz: predictions on the test set.
  - scores.npz: contains the featnames, wether each feature was selected, their scores
    and the hyperparameters selected by cross-validation
  - scores.tsv: like scores.npz, but in tsv format
"""
from sklearn.linear_model import LassoCV

import utils as u

# Train model
############################
X, y, featnames = u.read_data("${TRAIN_NPZ}")
param_grid = u.read_parameters("${PARAMS_FILE}")

clf = LassoCV(**param_grid)
clf.fit(X, y)

# Predict test
############################
X_test, _, _ = u.read_data("${TEST_NPZ}")

y_pred = clf.predict(X_test)
u.save_preds_npz(y_pred)

# Active features
############################
selected = clf.coef_ != 0

u.save_scores_npz(featnames, selected, clf.coef_, param_grid)
u.save_scores_tsv(featnames, selected, clf.coef_, param_grid)
