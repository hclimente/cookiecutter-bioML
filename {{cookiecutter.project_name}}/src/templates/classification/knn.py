#!/usr/bin/env python
"""
Input variables:
  - TRAIN_NPZ: path to a .npz file containing the train set. It must contain three
    elements: an X matrix, a y vector, and a featnames vector (optional)
  - TEST_NPZ: path to a .npz file containing the test set. It must contain three
    elements: an X matrix, a y vector, and a featnames vector (optional)
  - PARAMS_JSON: path to a json file with the hyperparameters
    - n_neighbors
Output files:
  - y_proba.npz: predictions on the test set.
  - scores.npz: contains the featnames, wether each feature was selected, their scores
    and the hyperparameters selected by cross-validation
  - scores.tsv: like scores.npz, but in tsv format
"""
from sklearn.neighbors import KNeighborsClassifier

from base.sklearn import SklearnModel


class kNNModel(SklearnModel):
    def __init__(self) -> None:
        knn = KNeighborsClassifier(weights="distance")
        super().__init__(knn)

    def score_features(self):
        return [1 for _ in range(self.clf.n_features_in_)]

    def select_features(self, scores):
        return [True for _ in scores]


if __name__ == "__main__":
    model = kNNModel()
    model.train("${TRAIN_NPZ}", "${SCORES_NPZ}", "${PARAMS_FILE}")
    model.predict_proba("${TEST_NPZ}", "${SCORES_NPZ}")
