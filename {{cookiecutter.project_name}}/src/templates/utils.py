import random
import sys
import traceback

import json
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


def read_yaml_parameters(params_yaml: str, algo_type: str, algo_name: str) -> dict:

    f = open(params_yaml)
    params = {"cv_parameters": {}, "fixed_parameters": {}}

    for x in yaml.load(f, Loader=yaml.Loader)[algo_type]:
        if x["name"] == algo_name:
            params["cv_parameters"] = x.get("cv_parameters", {})
            params["fixed_parameters"] = x.get("fixed_parameters", {})

    return sanitize_parameters(params)


def read_json_parameters(params: str):

    params = json.loads(params)
    return sanitize_parameters(params)


def sanitize_parameters(params: dict):

    to_delete = set()
    for k, v in params["cv_parameters"].items():
        if isinstance(v, list) and len(v) == 1:
            params["fixed_parameters"][k] = v[0]
            to_delete.add(k)
        else:
            params["fixed_parameters"][k] = v
            to_delete.add(k)

    params["cv_parameters"] = {
        k: v for k, v in params["cv_parameters"].items() if k not in to_delete
    }

    if set(params["fixed_parameters"].items()) & set(params["cv_parameters"].items()):
        raise Exception("Repeated parameters in fixed_parameters and cv_parameters.")

    return params


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
            FILE.write(f"# {key}: {value}\n")
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


def custom_error(error: int = 77, file: str = "error.txt", content=None):
    traceback.print_exc()
    np.save(file, content)
    sys.exit(error)


def sanitize_vector(x: npt.ArrayLike):
    if x is not None:
        x = np.array(x)
        x = x.flatten()

    return x
