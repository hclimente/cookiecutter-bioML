#!/usr/bin/env python
"""
Input variables:
  - TRAIN_NPZ: path to a .npz file containing three elements: an X matrix, a y vector,
    and a featnames vector (optional)
  - NET_NPZ: path to a .npz file with the adjacency matrix
  - PARAMS_JSON: path to a json file with the hyperparameters
    - lambda_1
    - lambda_2
Output files:
  - scores.npz: contains the score computed for each feature, the featnames and the
    hyperparameters selected by cross-validation
  - scores.tsv: like scores.npz, but in tsv format
"""

from galore import Galore
from sklearn.model_selection import GridSearchCV

import utils as u

u.set_random_state()

# Read data
############################
X, y, featnames = u.read_data("${TRAIN_NPZ}")
A = u.read_adjacency("${NET_NPZ}")
# TODO check if we can pass a grid
param_grid = u.read_parameters("${PARAMS_JSON}")

# Run algorithm
############################
galore = Galore(A, 0, 0)
clf = GridSearchCV(estimator=galore, param_grid=param_grid, scoring="roc_auc", n_jobs=5)
clf.fit(X, y)

# Save scores
############################
Wp = clf.best_estimator_.get_W("p").sum(axis=1)
Wn = clf.best_estimator_.get_W("n").sum(axis=1)
scores = Wp - Wn

best_hyperparams = {k: clf.best_params_[k] for k in param_grid.keys()}

u.save_scores_npz(featnames, scores, best_hyperparams)
u.save_scores_tsv(featnames, scores, best_hyperparams)
