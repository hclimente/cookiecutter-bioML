#!/usr/bin/env python
"""
Input variables:
    - TRAIN: path of a numpy array with x.
Output files:
    - selected.npy
"""
import pandas as pd
import numpy as np

import itertools
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score

from joblib import Parallel, delayed


def model_train_cv_lgb(X, y, hp):
    num_leave, alpha, lambda_ = hp
    num_round = 10

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
        "num_leaves": num_leave,
        "reg_alpha": alpha,
        "reg_lambda": lambda_,
    }

    n_splits = 2
    kf = KFold(n_splits=n_splits)

    score_cv = np.zeros(n_splits)

    for i, (train_index, val_index) in enumerate(kf.split(X)):
        X_train = X[train_index, :]
        y_train = y[train_index]

        X_val = X[val_index, :]
        y_val = y[val_index]

        X_train = np.asfortranarray(X_train)
        X_val = np.asfortranarray(X_val)
        y_train = np.asfortranarray(y_train)
        y_val = np.asfortranarray(y_val)

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        bst = lgb.train(gbdt_params, train_data, num_round, valid_sets=[val_data])

        # Prediction (Training)
        yhat_val = bst.predict(X_val)

        is_auc = 1
        if is_auc:
            score_cv[i] = roc_auc_score(y_val, yhat_val)
        else:
            pos_class = yhat_val > 0.5
            neg_class = yhat_val <= 0.5
            yhat_val[pos_class] = 1
            yhat_val[neg_class] = -1
            score_cv[i] = accuracy_score(y_val, yhat_val)

    return {"pair": hp, "score": score_cv.mean()}


def gbdt(X, y, num_leave, alpha, lambda_):
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
        "num_leaves": num_leave,
        "reg_alpha": alpha,
        "reg_lambda": lambda_,
    }

    num_round = 10
    data = lgb.Dataset(X, label=y)
    bst = lgb.train(gbdt_params, data, num_round)

    return bst


np.random.seed(0)

train_data = np.load("${TRAIN}")

X = train_data["X"]
y = train_data["Y"]
genes = train_data["genes"]

num_leaves = [20, 40, 60, 80, 100]
alphas = [0.001, 0.01, 0.1]
lambdas = [0.001, 0.01, 0.1]

max_score = 0
best_hyperparameter = None
list_hyperparameter = list(itertools.product(num_leaves, alphas, lambdas))

processes = [delayed(model_train_cv_lgb)(X, y, hp) for hp in list_hyperparameter]
result = Parallel(n_jobs=-1)(processes)

for r in result:
    if r["score"] > max_score:
        best_hyperparameter = r["pair"]
        max_score = r["score"]

model = gbdt(X, y, *best_hyperparameter)
feature_importance = model.feature_importance()

# filter out unselected genes
selected_index = np.nonzero(feature_importance)[0]
selected_genes = genes[selected_index]
selected_importance = feature_importance[selected_index]

# output = np.stack([selected_genes, selected_importance], axis=1)


with open("scored_genes.gbdt.tsv", "a") as f:
    f.write("# num_leaves: {}\\n".format(best_hyperparameter[0]))
    f.write("# alpha: {}\\n".format(best_hyperparameter[1]))
    f.write("# lambda: {}\\n".format(best_hyperparameter[2]))

    pd.DataFrame({"gene": selected_genes, "weight": selected_importance}).to_csv(
        f, sep="\\t", index=False
    )
