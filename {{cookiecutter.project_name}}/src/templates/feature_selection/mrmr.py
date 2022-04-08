#!/usr/bin/env python
"""
Input variables:
  - TRAIN_NPZ: path to a .npz file containing the train set. It must contain three
    elements: an X matrix, a y vector, and a featnames vector (optional)
  - TEST_NPZ: path to a .npz file containing the test set. It must contain three
    elements: an X matrix, a y vector, and a featnames vector (optional)
  - PARAMS_FILE: path to a YAML file with the hyperparameters
    - n_nonzero_coefs
Output files:
  - y_proba.npz: predictions on the test set.
  - scores.npz: contains the featnames, wether each feature was selected, their scores
    and the hyperparameters selected by cross-validation
  - scores.tsv: like scores.npz, but in tsv format
"""
import numpy as np
import subprocess

import utils as u

# Prepare data
############################
X, y, featnames = u.read_data("${TRAIN_NPZ}")
ds = np.hstack((np.expand_dims(y, axis=1), X))
cols = "y," + ",".join(featnames)

np.savetxt("dataset.csv", ds, header=cols, fmt="%1.3f", delimiter=",", comments="")
discretization = "-t 0" if "${MODE}" == "regression" else ""

# Run mRMR
############################
samples, features = X.shape
param_grid = u.read_parameters("${PARAMS_FILE}", "feature_selection", "mrmr")

out = subprocess.check_output(
    [
        "mrmr",
        "-i",
        "dataset.csv",
        discretization,
        "-n",
        param_grid["num_features"],
        "-s",
        str(samples),
        "-v",
        str(features),
    ]
)

# Get selected features
############################
flag = False
selected = []
for line in out.decode("ascii").split("\\n"):
    if flag and "Name" not in line:
        if not line:
            flag = False
        else:
            f = line.split("\\t")[2].strip()
            selected.append(int(f))
    elif "mRMR features" in line:
        flag = True

scores = [0 for _ in selected]
selected = np.array(selected)

u.save_scores_npz(featnames, selected, scores, param_grid)
u.save_scores_tsv(featnames, selected, scores, param_grid)
