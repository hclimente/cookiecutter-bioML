#!/usr/bin/env python
"""
Input variables:
    - TRAIN: path of a numpy array with x.
    - LAMBDA: regularisation scalar for the L1 penalty
Output files:
    - selected.npy
"""
import numpy as np
from sklearn.utils.validation import check_array
from spams import fistaFlat


def logreg(X, y, lambda_1):

    weights_0 = np.zeros((X.shape[1], 1), dtype="float32", order="F")

    X = check_array(X, order="F", dtype="float32")
    y = np.expand_dims(y, 1)
    y = check_array(y, order="F", dtype="float32")

    weights, optim_info = fistaFlat(
        y,
        X,
        weights_0,
        True,
        verbose=True,
        max_it=500,
        L0=0.1,
        tol=1e-3,
        loss="weighted-logistic",
        regul="l1",
        lambda1=lambda_1,
    )

    print(
        "mean loss: %f, mean relative duality_gap: %f, number of iterations: %f"
        % (
            np.mean(optim_info[0, :], 0),
            np.mean(optim_info[2, :], 0),
            np.mean(optim_info[3, :], 0),
        )
    )

    weights = np.squeeze(weights, 1)

    return weights


np.random.seed(0)

train_data = np.load("${TRAIN}", allow_pickle=True)

X_train = train_data["X"]
y_train = train_data["Y"]

lambda_1 = float("${LAMBDA}")
weights = logreg(X_train, y_train, lambda_1)

# test
test_data = np.load("${TEST}")

X_test = test_data["X"]

y_proba = 1 / (1 + np.exp(-np.dot(X_test, weights)))
np.save("y_proba.npy", y_proba)
