#!/usr/bin/env python
"""
Input variables:
  - TRAIN_NPZ: path to a .npz file containing the train set. It must contain three
    elements: an X matrix, a y vector, and a featnames vector (optional)
  - TEST_NPZ: path to a .npz file containing the test set. It must contain three
    elements: an X matrix, a y vector, and a featnames vector (optional)
  - NET_NPZ: path to a .npz file with the adjacency matrix
  - PARAMS_JSON: path to a json file with the hyperparameters
    - n_nonzero_coefs
Output files:
  - y_proba.npz: predictions on the test set.
  - scores.npz: contains the featnames, wether each feature was selected, their scores
    and the hyperparameters selected by cross-validation
  - scores.tsv: like scores.npz, but in tsv format
"""
from galore import LogisticGraphLasso
from sklearn.model_selection import GridSearchCV

import utils as u

# Train model
############################
X, y, featnames = u.read_data("${TRAIN_NPZ}")
A = u.read_adjacency("${NET_NPZ}")
param_grid = u.read_parameters("${PARAMS_FILE}")

gl = LogisticGraphLasso(A, 0, 0)
clf = GridSearchCV(estimator=gl, param_grid=param_grid, scoring="roc_auc", n_jobs=5)
clf.fit(X, y)

# Predict test
############################
X_test, _, _ = u.read_data("${TEST_NPZ}")

y_proba = clf.predict_proba(X_test)
u.save_proba_npz(y_proba)

# Score features
############################
Wp = clf.best_estimator_.get_W("p").sum(axis=1)
Wn = clf.best_estimator_.get_W("n").sum(axis=1)
scores = Wp - Wn
selected = scores != 0

u.save_scores_npz(featnames, selected, scores, param_grid)
u.save_scores_tsv(featnames, selected, scores, param_grid)
