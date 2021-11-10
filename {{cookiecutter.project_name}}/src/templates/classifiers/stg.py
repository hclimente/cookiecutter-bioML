#!/usr/bin/env python
"""
Input variables:
    - TRAIN: path of a numpy array with x.
    - SIGMA
    - LAMBDA
    - LR
Output files:
    - selected.npy
"""
import pandas

# the import pandas module is necessary, there is an import
# error with STG...
import numpy as np
import torch
from stg import STG


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epochs = 1000
archi = [60, 20]


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

train_data = np.load("${TRAIN}", allow_pickle=True)

X_train = train_data["X"]
y_train = train_data["Y"]
y_train = ((y_train + 1) / 2).astype(int)

lr = float("${LR}")
sigma = float("${SIGMA}")
lambda_ = float("${LAMBDA}")


model = stg_model(X_train, y_train, lr, sigma, lambda_)

if torch.cuda.is_available():
    # I had devices and data in different locations when
    # gpu is used
    model._model.cpu()
# test
test_data = np.load("${TEST}")

X_test = test_data["X"]
print("getting the probabilities")
feed_dict = {"input": torch.from_numpy(X_test).float()}
y_proba = model._model.forward(feed_dict)["prob"].detach().numpy()[:, 1]

np.save("y_proba.npy", y_proba)
