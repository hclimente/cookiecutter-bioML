#!/usr/bin/env python
"""
Input variables:
    - TRAIN: path of a numpy array with x.
Output files:
    - selected.npy
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import spams


def logreg(X, y, lambda_1):

    weights_0 = np.zeros((X.shape[1], 1), dtype="float32", order="F")

    X = np.asfortranarray(X)
    y = np.expand_dims(y, 1)
    y = np.asfortranarray(y)
    y = y.astype("float32")

    weights, optim_info = spams.fistaFlat(
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


def model_train_cv_l1(X, y, lambda_1):

    n_splits = 2
    kf = KFold(n_splits=n_splits)

    score_cv = np.zeros(n_splits)
    for i, (train_index, val_index) in enumerate(kf.split(X)):
        X_train = X[train_index, :]
        y_train = y[train_index]

        X_val = X[val_index, :]
        y_val = y[val_index]

        weights = logreg(X_train, y_train, lambda_1)

        # Prediction (Training)
        y_pred = 1 / (1 + np.exp(-np.dot(X_val, weights)))

        is_auc = 1
        if is_auc:
            score_cv[i] = roc_auc_score(y_val, y_pred)
        else:
            pos_class = y_pred > 0.5
            neg_class = y_pred <= 0.5
            y_pred[pos_class] = 1
            y_pred[neg_class] = -1
            score_cv[i] = accuracy_score(y_val, y_pred)

    return score_cv.mean()


np.random.seed(0)

train_data = np.load("${TRAIN}")

X = train_data["X"]
y = train_data["Y"]
genes = train_data["genes"]

lambda_1s = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
max_score = 0
best_lambda_1 = 0
for lambda_1 in lambda_1s:
    score = model_train_cv_l1(X, y, lambda_1)

    if score > max_score:
        best_lambda_1 = lambda_1
        max_score = score

weights = logreg(X, y, best_lambda_1)

selected_index = np.nonzero(weights)[0]
selected_genes = genes[selected_index]
selected_weights = weights[selected_index]

output = np.stack([selected_genes, selected_weights], axis=1)

np.save("selected.npy", output)
with open("scored_genes.logreg.tsv", "a") as f:
    f.write("# lambda_1: {}\\n".format(best_lambda_1))
    pd.DataFrame({"gene": selected_genes, "weight": selected_weights}).to_csv(
        f, sep="\t", index=False
    )
