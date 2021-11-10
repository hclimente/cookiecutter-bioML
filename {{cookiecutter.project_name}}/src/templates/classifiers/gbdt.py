#!/usr/bin/env python
"""
Input variables:
    - TRAIN: path of a numpy array with x.
    - ALPHA
    - LAMBDA
    - NUM_LEAVES
Output files:
    - selected.npy
"""

import numpy as np
import lightgbm as lgb


def gbdt(X, y, num_leaves, alpha, lambda_):
    gbdt_params = {
        # fixed
        "objective": "binary",
        "boosting": "gbdt",
        "boosting_type": "gbdt",
        "max_depth": -1,
        "learning_rate": 0.1,
        "n_estimators": 1000,
        "metric": "auc",
        # variable
        "num_leaves": num_leaves,
        "reg_alpha": alpha,
        "reg_lambda": lambda_,
    }

    num_round = 10
    data = lgb.Dataset(X, label=y)
    bst = lgb.train(gbdt_params, data, num_round)

    return bst


np.random.seed(0)

train_data = np.load("${TRAIN}", allow_pickle=True)

X_train = train_data["X"]
y_train = train_data["Y"]

num_leaves = int("${NUM_LEAVES}")
alpha = float("${ALPHA}")
lambda_ = float("${LAMBDA}")


model = gbdt(X_train, y_train, num_leaves, alpha, lambda_)

# test
test_data = np.load("${TEST}")

X_test = test_data["X"]
print("getting the probabilities")

y_proba = model.predict(X_test)

np.save("y_proba.npy", y_proba)
