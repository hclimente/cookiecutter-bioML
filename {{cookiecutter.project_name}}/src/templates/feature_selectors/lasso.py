#!/usr/bin/env python
"""
Input variables:
    - X_TRAIN: path of a numpy array with x.
    - Y_TRAIN: path of a numpy array with y.
    - FEATNAMES: path of a numpy array with feature names.
    - MODE: regression or classification.
Output files:
    - features_lasso.npy: numpy array with the 0-based index of
    the selected features.
"""

import numpy as np
from sklearn import linear_model
from sklearn.feature_selection import SelectFromModel

x_train = np.load("${X_TRAIN}")
y_train = np.load("${Y_TRAIN}")
featnames = np.load("${FEATNAMES}")

if "${MODE}" == "regression":
    clf = linear_model.LassoCV()
elif "${MODE}" == "classification":
    clf = linear_model.LogisticRegressionCV()

clf.fit(x_train, y_train)

sfm = SelectFromModel(clf, prefit=True)
features = featnames[sfm.get_support()]
np.save("features_lasso.npy", features)
