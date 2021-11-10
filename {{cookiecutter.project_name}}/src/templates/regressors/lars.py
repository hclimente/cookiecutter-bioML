#!/usr/bin/env python
"""
Input variables:
  - TRAIN_NPZ: path to a .npz file containing the train set. It must contain three
    elements: an X matrix, a y vector, and a featnames vector (optional)
  - TEST_NPZ: path to a .npz file containing the test set. It must contain three
    elements: an X matrix, a y vector, and a featnames vector (optional)
  - PARAMS_JSON: path to a json file with the hyperparameters
    - n_nonzero_coefs
Output files:
  - y_pred.npz: predictions on the test set.
  - scores.npz: contains the featnames, wether each feature was selected, their scores
    and the hyperparameters selected by cross-validation
  - scores.tsv: like scores.npz, but in tsv format
"""
from sklearn.linear_model import Lars

from base.sklearn import SklearnModel

class LarsModel(SklearnModel):
    def __init__(self) -> None:
        lars = Lars()
        super().__init__(lars)
        
    def score_features(self):
        return self.clf.coef_
    
    def select_features(self, scores):
        return scores != 0

if __name__ == "__main__":
    model = LarsModel()
    model.train("${TRAIN_NPZ}", "${SELECTED_NPZ}", "${PARAMS_FILE}")
    model.predict("${TEST_NPZ}")
