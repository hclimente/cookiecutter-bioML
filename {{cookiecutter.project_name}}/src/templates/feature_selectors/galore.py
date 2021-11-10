#!/usr/bin/env python
"""
Input variables:
  - TRAIN_NPZ: path to a .npz file containing three elements: an X matrix, a y vector,
    and a featnames vector
  - NET_NPZ: path to a .npz file with the adjacency matrix
Output files:
  - scores.npz: contains the score computed for each feature, the featnames and the
    hyperparameters selected by cross-validation
  - scores.tsv: like scores.npz, but in tsv format
"""

from galore import Galore
from sklearn.model_selection import GridSearchCV

import utils as u

# TODO allow user to change the parameters
param_grid = {
    "lambda_1": [0.05, 0.1, 0.15, 0.2, 0.25],
    "lambda_2": [0.05, 0.1, 0.15, 0.2, 0.25],
}

u.set_random_state()

# Read data
############################
X, y, featnames = u.read_data("${TRAIN_NPZ}")
A = u.read_adjacency("${NET_NPZ}")

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

best_hyperparams = {
    "lambda_1": clf.best_params_["lambda_1"],
    "lambda_2": clf.best_params_["lambda_2"],
}

u.save_scores_npz(scores, featnames, best_hyperparams)
u.save_scores_tsv(scores, featnames, best_hyperparams)
