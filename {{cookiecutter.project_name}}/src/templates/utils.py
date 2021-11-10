import json
import random
import sys
import traceback

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import load_npz


# Input functions
###########################
def read_data(data_npz: str, selected_npz: str = ""):
    data = np.load(data_npz)

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


def read_parameters(json_path: str) -> dict:
    f = open(json_path)

    return json.load(f)


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
        scores=scores,
        selected=selected,
        hyperparams=hyperparams,
    )


def save_scores_tsv(
    featnames: npt.ArrayLike,
    selected: npt.ArrayLike,
    scores: npt.ArrayLike = None,
    hyperparams: dict = {},
):
    features_dict = {"feature": featnames, "selected": selected}
    if scores is not None:
        features_dict["score"] = scores

    with open("scores.tsv", "a") as FILE:
        for key, value in hyperparams.items():
            FILE.write("# {}: {}\\n".format(key, value))
        pd.DataFrame(features_dict).to_csv(FILE, sep="\t", index=False)


def save_preds_npz(preds: npt.ArrayLike = None, hyperparams: dict = None):
    np.savez("y_pred.npz", preds=preds, hyperparams=hyperparams)


def save_proba_npz(proba: npt.ArrayLike = None, hyperparams: dict = None):
    np.savez("y_proba.npz", proba=proba, hyperparams=hyperparams)


# Other functions
##########################
def set_random_state(seed=0):
    np.random.seed(seed)
    random.seed(seed)


def custom_error(error: int = 77, file: str = None, content=None):
    traceback.print_exc()
    np.save(file, content)
    sys.exit(error)
