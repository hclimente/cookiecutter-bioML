#!/usr/bin/env python
"""
Input variables:
  - TRAIN_NPZ: path to a .npz file containing three elements: an X matrix, a y vector,
    and a featnames vector (optional)
  - PARAMS_JSON: path to a json file with the hyperparameters
    - n_estimators
    - max_features
    - max_depth
    - criterion
Output files:
  - scores.npz: contains the score computed for each feature, the featnames and the
    hyperparameters selected by cross-validation
  - scores.tsv: like scores.npz, but in tsv format
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import utils as u

u.set_random_state()

# Train model
############################
X, y, featnames = u.read_data("${TRAIN_NPZ}", "${SELECTED}")
param_grid = u.read_parameters("${PARAMS_FILE}")

rf = RandomForestClassifier()
clf = GridSearchCV(rf, param_grid)
clf.fit(X, y)

best_hyperparams = {k: clf.best_params_[k] for k in param_grid.keys()}

# Predict test
############################
X_test, _, _ = u.read_data("${TEST_NPZ}", "${SELECTED}")

y_pred = clf.predict(X_test)
u.save_proba_npz(y_pred, best_hyperparams)

# Feature importance
############################
scores = clf.best_params_.feature_importances_

u.save_scores_npz(featnames, scores, best_hyperparams)
u.save_scores_tsv(featnames, scores, best_hyperparams)
