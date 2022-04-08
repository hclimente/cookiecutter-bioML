#!/usr/bin/env python
"""
Input variables:
  - TRAIN_NPZ: path to a .npz file containing three elements: an X matrix, a y vector,
    and a featnames vector (optional)
  - PARAMS_FILE: path to a YAML file with the hyperparameters
    - n_estimators
    - max_features
    - max_depth
    - criterion
Output files:
  - scores.npz: contains the score computed for each feature, the featnames and the
    hyperparameters selected by cross-validation
  - scores.tsv: like scores.npz, but in tsv format
"""
from sklearn.ensemble import RandomForestClassifier

from base.sklearn import SklearnModel


class RandomForestModel(SklearnModel):
    def __init__(self) -> None:
        rf = RandomForestClassifier()
        super().__init__(rf, "prediction", "random_forest")

    def score_features(self):
        return self.clf.best_estimator_.feature_importances_

    def select_features(self, scores):
        return scores != 0


if __name__ == "__main__":
    model = RandomForestModel()
    model.train("${TRAIN_NPZ}", "${SCORES_NPZ}", "${PARAMS_FILE}")
    model.predict_proba("${TEST_NPZ}", "${SCORES_NPZ}")
    model.predict("${TEST_NPZ}", "${SCORES_NPZ}")
