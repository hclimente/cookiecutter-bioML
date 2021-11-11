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
  - y_proba.npz: predictions on the test set.
  - scores.npz: contains the featnames, wether each feature was selected, their scores
    and the hyperparameters selected by cross-validation
  - scores.tsv: like scores.npz, but in tsv format
"""
from sklearn.svm import SVC

from base.sklearn import SklearnModel


class SVCModel(SklearnModel):
    def __init__(self) -> None:
        svc = SVC(gamma="scale", class_weight="balanced", probability=True)
        super().__init__(svc)

    def score_features(self):
        if self.clf.get_params()["estimator__kernel"] == "linear":
            return self.clf.best_estimator_.coef_
        else:
            return [1 for _ in range(self.clf.n_features_in_)]

    def select_features(self, scores):
        return [True for _ in scores]


if __name__ == "__main__":
    model = SVCModel()
    model.train("${TRAIN_NPZ}", "${SCORES_NPZ}", "${PARAMS_FILE}")
    model.predict_proba("${TEST_NPZ}", "${SCORES_NPZ}")
    model.predict("${TEST_NPZ}", "${SCORES_NPZ}")
