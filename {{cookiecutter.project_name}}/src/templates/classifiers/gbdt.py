#!/usr/bin/env python
"""
Input variables:
  - TRAIN_NPZ: path to a .npz file containing the train set. It must contain three
    elements: an X matrix, a y vector, and a featnames vector (optional)
  - TEST_NPZ: path to a .npz file containing the test set. It must contain three
    elements: an X matrix, a y vector, and a featnames vector (optional)
  - PARAMS_JSON: path to a json file with the hyperparameters
    - n_nonzero_coefs
Output files:
  - y_proba.npz: predictions on the test set.
  - scores.npz: contains the featnames, wether each feature was selected, their scores
    and the hyperparameters selected by cross-validation
  - scores.tsv: like scores.npz, but in tsv format
"""
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

import utils as u

# Train model
############################
X, y, featnames = u.read_data("${TRAIN_NPZ}")
param_grid = u.read_parameters("${PARAMS_FILE}")

gbdt = LGBMClassifier(objective="binary", n_estimators=1000)
clf = GridSearchCV(gbdt, param_grid)
clf.fit(X, y)

# Predict test
############################
X_test, _, _ = u.read_data("${TEST_NPZ}")

y_proba = clf.predict_proba(X_test)
u.save_proba_npz(y_proba)

# Active features
############################
selected = [True for _ in featnames]

u.save_scores_npz(featnames, selected, clf.feature_importances_, param_grid)
u.save_scores_tsv(featnames, selected, clf.feature_importances_, param_grid)