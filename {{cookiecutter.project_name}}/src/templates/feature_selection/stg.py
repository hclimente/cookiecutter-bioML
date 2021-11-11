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
import torch
from stg import STG
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
archi = [60, 20]
n_epochs = 1000


def model_train_cv_stg(X, y, hp, device=device):
    learning_rate, sigma, lambda_ = hp

    n_splits = 2
    kf = KFold(n_splits=n_splits)

    score_cv = np.zeros(n_splits)

    for i, (train_index, val_index) in enumerate(kf.split(X)):
        X_train = X[train_index, :]
        y_train = y[train_index]

        X_val = X[val_index, :]
        y_val = y[val_index]

        model = STG(
            task_type="classification",
            input_dim=X.shape[1],
            output_dim=2,
            hidden_dims=archi,
            activation="tanh",
            optimizer="SGD",
            learning_rate=learning_rate,
            batch_size=X.shape[0],
            feature_selection=True,
            sigma=sigma,
            lam=lambda_,
            random_state=1,
            device=device,
        )

        model.fit(
            X_train,
            y_train,
            nr_epochs=n_epochs,
            valid_X=X_val,
            valid_y=y_val,
            shuffle=True,
            print_interval=200,
        )
        # Prediction (Training)
        feed_dict = {"input": torch.from_numpy(X_val).float()}
        yhat_val = model._model.forward(feed_dict)["prob"].detach().numpy()[:, 1]

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


def stg_model(X, y, learning_rate, sigma, lam, device=device):

    model = STG(
        task_type="classification",
        input_dim=X.shape[1],
        output_dim=2,
        hidden_dims=archi,
        activation="tanh",
        optimizer="SGD",
        learning_rate=learning_rate,
        batch_size=X.shape[0],
        feature_selection=True,
        sigma=sigma,
        lam=lam,
        random_state=1,
        device=device,
    )
    # we have to give valid_X and valid_y because it gives an error
    # however, no EARLY STOPPING :-)
    model.fit(
        X, y, nr_epochs=n_epochs, shuffle=True, valid_X=X, valid_y=y, print_interval=200
    )
    return model


np.random.seed(0)

train_data = np.load("${TRAIN}")

X = train_data["X"]
y = train_data["Y"]
y = ((y + 1) / 2).astype(int)
genes = train_data["genes"]

learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
sigmas = [0.1, 0.2, 0.3, 0.5, 1.0]
lambdas = [5e-5, 5e-4, 5e-3, 5e-2, 5e-1]

max_score = 0
best_hyperparameter = None
list_hyperparameter = list(itertools.product(learning_rates, sigmas, lambdas))

for hp in list_hyperparameter:
    r = model_train_cv_stg(X, y, hp, device)
    if r["score"] > max_score:
        best_hyperparameter = r["pair"]
        max_score = r["score"]

model = stg_model(X, y, *best_hyperparameter)
feature_importance = model.get_gates(mode="prob")

# filter out unselected genes
selected_index = np.nonzero(feature_importance)[0]
selected_genes = genes[selected_index]
selected_importance = feature_importance[selected_index]

# output = np.stack([selected_genes, selected_importance], axis=1)


with open("scored_genes.stg.tsv", "a") as f:
    f.write("# num_leaves: {}\\n".format(best_hyperparameter[0]))
    f.write("# alpha: {}\\n".format(best_hyperparameter[1]))
    f.write("# lambda: {}\\n".format(best_hyperparameter[2]))

    pd.DataFrame({"gene": selected_genes, "weight": selected_importance}).to_csv(
        f, sep="\\t", index=False
    )
