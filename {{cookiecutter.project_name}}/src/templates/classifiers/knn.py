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
from sklearn.linear_model import KNeighborsClassifier

from base.sklearn import SklearnModel
import utils as u

class kNNModel(SklearnModel):
    def __init__(self) -> None:
        knn = KNeighborsClassifier(weights="distance")
        super().__init__(knn)
        
    def score_features(self):
        return self.clf.coef_
    
    def select_features(self, scores):
        return scores != 0

if __name__ == "__main__":
    model = kNNModel()
    model.train("${TRAIN_NPZ}", "${SELECTED_NPZ}", "${PARAMS_FILE}")
    model.predict_proba("${TEST_NPZ}")
