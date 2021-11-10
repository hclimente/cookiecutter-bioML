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
def read_data(npz_path: str):
    data = np.load(npz_path)

    X = data["X"]
    y = data["y"]

    if "featnames" in data.keys():
        featnames = data["featnames"]
    else:
        featnames = np.arange(X.shape[1])

    return X, y, featnames


def read_adjacency(npz_path: str):

    return load_npz(npz_path)


def read_parameters(json_path: str) -> dict:

    return json.load(json_path)


# Output functions
##########################
def save_scores_npz(
    featnames: npt.ArrayLike, scores: npt.ArrayLike, hyperparams: dict = None
):
    np.savez("scores.npz", featnames=featnames, scores=scores, hyperparams=hyperparams)


def save_scores_tsv(
    featnames: npt.ArrayLike, scores: npt.ArrayLike, hyperparams: dict = {}
):
    features_dict = {"feature": featnames, "score": scores}

    with open("scores.tsv", "a") as FILE:
        for key, value in hyperparams.items():
            FILE.write("# {}: {}\\n".format(key, value))
        pd.DataFrame(features_dict).to_csv(FILE, sep="\t", index=False)


def save_selected_npz(
    featnames: npt.ArrayLike, scores: npt.ArrayLike = None, hyperparams: dict = None
):
    np.savez(
        "selected.npz", featnames=featnames, scores=scores, hyperparams=hyperparams
    )


def save_selected_tsv(
    featnames: npt.ArrayLike, scores: npt.ArrayLike = None, hyperparams: dict = {}
):

    features_dict = {"feature": featnames}
    if scores:
        features_dict["score"] = scores

    with open("selected.tsv", "a") as FILE:
        for key, value in hyperparams.items():
            FILE.write("# {}: {}\\n".format(key, value))
        pd.DataFrame(features_dict).to_csv(FILE, sep="\t", index=False)


# Other functions
##########################
def set_random_state(seed=0):
    np.random.seed(seed)
    random.seed(seed)


def custom_error(error: int = 77, file: str = None, content=None):
    traceback.print_exc()
    np.save(file, content)
    sys.exit(error)
