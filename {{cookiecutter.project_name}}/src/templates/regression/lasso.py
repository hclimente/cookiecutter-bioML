#!/usr/bin/env python
"""
Input variables:
  - TRAIN_NPZ: path to a .npz file containing the train set. It must contain three
    elements: an X matrix, a y vector, and a featnames vector (optional)
  - TEST_NPZ: path to a .npz file containing the test set. It must contain three
    elements: an X matrix, a y vector, and a featnames vector (optional)
  - PARAMS_JSON: path to a json file with the hyperparameters
    - None
Output files:
  - y_pred.npz: predictions on the test set.
  - scores.npz: contains the featnames, wether each feature was selected, their scores
    and the hyperparameters selected by cross-validation
  - scores.tsv: like scores.npz, but in tsv format
"""
from sklearn.linear_model import Lasso

from base.sklearn import SklearnModel


class LassoModel(SklearnModel):
    def __init__(self) -> None:
        lasso = Lasso()
        super().__init__(lasso, "lasso")

    def score_features(self):
        return self.clf.best_estimator_.coef_

    def select_features(self, scores):
        return scores != 0


if __name__ == "__main__":
    model = LassoModel()
    model.train("${TRAIN_NPZ}", "${SCORES_NPZ}", "${PARAMS_FILE}")
    model.predict("${TEST_NPZ}", "${SCORES_NPZ}")
