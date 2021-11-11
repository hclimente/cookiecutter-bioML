import os
import random
import sys
import traceback

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from scipy.sparse import load_npz


# Input functions
###########################
def read_data(data_npz: str, selected_npz: str = ""):
    data = np.load(data_npz, allow_pickle=True)

    X = data["X"]
    y = data["y"]

    if "featnames" in data.keys():
        featnames = data["featnames"]
    else:
        featnames = np.arange(X.shape[1])

    if selected_npz != "":
        selected = np.load(selected_npz)["selected"]

        if not sum(selected):
            custom_error()

    return X, y, featnames


def read_adjacency(A_npz: str):

    return load_npz(A_npz)


def read_parameters(params_yaml: str) -> dict:

    try:
        clf_name = os.path.basename(__file__)
        clf_name = os.path.splitext(clf_name)[0]

        f = open(params_yaml)
        return yaml.load(f)[clf_name]
    except FileNotFoundError:
        return {}


# Output functions
##########################
def save_scores_npz(
    featnames: npt.ArrayLike,
    selected: npt.ArrayLike,
    scores: npt.ArrayLike = None,
    hyperparams: dict = None,
):
    np.savez(
        "scores.npz",
        featnames=featnames,
        scores=sanitize_vector(scores),
        selected=sanitize_vector(selected),
        hyperparams=hyperparams,
    )


def save_scores_tsv(
    featnames: npt.ArrayLike,
    selected: npt.ArrayLike,
    scores: npt.ArrayLike = None,
    hyperparams: dict = {},
):
    features_dict = {"feature": featnames, "selected": sanitize_vector(selected)}
    if scores is not None:
        features_dict["score"] = sanitize_vector(scores)

    with open("scores.tsv", "a") as FILE:
        for key, value in hyperparams.items():
            FILE.write("# {}: {}\\n".format(key, value))
        pd.DataFrame(features_dict).to_csv(FILE, sep="\t", index=False)


def save_preds_npz(preds: npt.ArrayLike = None, hyperparams: dict = None):
    np.savez("y_pred.npz", preds=sanitize_vector(preds), hyperparams=hyperparams)


def save_proba_npz(proba: npt.ArrayLike = None, hyperparams: dict = None):
    np.savez("y_proba.npz", proba=sanitize_vector(proba), hyperparams=hyperparams)


def save_analysis_tsv(**kwargs):

    metrics_dict = locals()["kwargs"]

    with open("performance.tsv", "w", newline="") as FILE:
        pd.DataFrame(metrics_dict).to_csv(FILE, sep="\t", index=False)


# Other functions
##########################
def set_random_state(seed=0):
    np.random.seed(seed)
    random.seed(seed)


def custom_error(error: int = 77, file: str = None, content=None):
    traceback.print_exc()
    np.save(file, content)
    sys.exit(error)


def sanitize_vector(x: npt.ArrayLike):
    if x is not None:
        x = np.array(x)
        x = x.flatten()

    return x
