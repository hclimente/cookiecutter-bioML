import random
import sys
import traceback

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import load_npz


def read_data(npz_path: str):
    data = np.load(npz_path)

    return data["X"], data["Y"], data["featnames"]


def read_adjacency(npz_path: str):

    return load_npz(npz_path)


def save_scores_npz(
    scores: npt.ArrayLike, featnames: npt.ArrayLike, hyperparams: dict = None
):
    np.savez("scores.npz", scores=scores, featnames=featnames, hyperparams=hyperparams)


def save_scores_tsv(
    scores: npt.ArrayLike, featnames: npt.ArrayLike, hyperparams: dict = {}
):
    with open("scores.tsv", "a") as FILE:
        for key, value in hyperparams.items():
            FILE.write("# {}: {}\\n".format(key, value))
        pd.DataFrame({"feature": featnames, "score": scores}).to_csv(
            FILE, sep="\t", index=False
        )


def set_random_state(seed=0):
    np.random.seed(seed)
    random.seed(seed)


def custom_error(error: int = 77, file: str = None, content=None):
    traceback.print_exc()
    np.save(file, content)
    sys.exit(error)
