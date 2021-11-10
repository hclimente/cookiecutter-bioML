#!/usr/bin/env python
"""
Input variables:
    - DATA: path of a numpy array with x.
    - GXG: path to the PPIN
    - PHENO: index of the phenotype
Output files:
    - Xy.npz
    - A.npz
"""

import numpy as np

from templates.io.makeA import makeA

idx = int("${PHENO.value}")
controls = int("${WHICH_CONTROLS.value}")
subgroup = int("${WHICH_GROUP.value}")

with open("${DATA}", "rb") as a_file:
    # pos = 1, neg = 0, NA = -999
    input_data = np.load(a_file)
    X = input_data["X"].T
    Y = input_data["Y"]
    y = Y[:, idx]
    genes = input_data["genes"]

    if controls != idx:
        wt = Y[:, controls]
        y[np.logical_and(wt > 0, y != 1)] = -999
        y[wt == 0] = 0

    if subgroup != idx:
        subgroup = Y[:, subgroup]
        y[subgroup < 1] = -999

    X = X[y >= 0, :]
    y = y[y >= 0]
    y = y.astype("bool")

n = X.shape[0]
perm = np.random.permutation(n)
X = X[perm, :]
y = y[perm]
y = y * 2 - 1

if "${GXG}":
    from scipy.sparse import save_npz

    # read network
    A, X, genes = makeA("${GXG}", X, genes)

    save_npz("A.npz", A)

# save data
np.savez("Xy.npz", X=X, Y=y, genes=genes)
