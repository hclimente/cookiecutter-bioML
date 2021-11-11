#!/usr/bin/env python
"""
Input variables:
  - TRAIN_NPZ: path to a .npz file containing the train set. It must contain three
    elements: an X matrix, a y vector, and a featnames vector (optional)
  - TEST_NPZ: path to a .npz file containing the test set. It must contain three
    elements: an X matrix, a y vector, and a featnames vector (optional)
  - NET_NPZ: path to a .npz file with the adjacency matrix
  - PARAMS_JSON: path to a json file with the hyperparameters
    - n_nonzero_coefs
Output files:
  - y_proba.npz: predictions on the test set.
  - scores.npz: contains the featnames, wether each feature was selected, their scores
    and the hyperparameters selected by cross-validation
  - scores.tsv: like scores.npz, but in tsv format
"""
from galore import LogisticGraphLasso

from base.sklearn import SklearnModel
import utils as u


class LogisticGraphLassoModel(SklearnModel):
    def __init__(self, adjacency_npz) -> None:
        A = u.read_adjacency(adjacency_npz)
        lgl = LogisticGraphLasso(A, 0, 0)
        super().__init__(lgl)

    def score_features(self):
        Wp = self.clf.best_estimator_.get_W("p").sum(axis=1)
        Wn = self.clf.best_estimator_.get_W("n").sum(axis=1)

        return Wp - Wn

    def select_features(self, scores):
        return scores != 0


if __name__ == "__main__":
    model = LogisticGraphLassoModel("${NET_NPZ}")
    model.train("${TRAIN_NPZ}", "${SCORES_NPZ}", "${PARAMS_FILE}")
    model.predict_proba("${TEST_NPZ}")
    model.predict("${TEST_NPZ}", "${SCORES_NPZ}")
