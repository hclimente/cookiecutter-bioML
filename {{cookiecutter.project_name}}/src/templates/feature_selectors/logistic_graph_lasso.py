#!/usr/bin/env python
"""
Input variables:
    - TRAIN: path of a numpy array with x.
    - NET: pickled file with the adjacency matrix
Output files:
    - selected.npy
"""

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.model_selection import GridSearchCV

from galore import LogisticGraphLasso

param_grid = {
    "lambda_1": [0.05, 0.1, 0.15, 0.2, 0.25],
    "lambda_2": [0.05, 0.1, 0.15, 0.2, 0.25],
}

np.random.seed(0)

train_data = np.load("${TRAIN}")

X = train_data["X"]
y = train_data["Y"]
genes = train_data["genes"]

A = load_npz("${NET}")

gl = LogisticGraphLasso(A, 0, 0)
clf = GridSearchCV(estimator=gl, param_grid=param_grid, scoring="roc_auc", n_jobs=5)
clf.fit(X, y)

Wp = clf.best_estimator_.get_W("p").sum(axis=1)
Wn = clf.best_estimator_.get_W("n").sum(axis=1)

with open("scored_genes.logistic_graph_lasso.tsv", "a") as f:
    f.write("# lambda_1: {}\\n".format(clf.best_params_["lambda_1"]))
    f.write("# lambda_2: {}\\n".format(clf.best_params_["lambda_2"]))
    pd.DataFrame({"gene": genes, "score_p": Wp, "score_n": Wn}).to_csv(
        f, sep="\t", index=False
    )
