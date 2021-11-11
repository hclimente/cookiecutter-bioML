#!/usr/bin/env python

import numpy as np

from base.simulator import Simulator

class Categorical1(Simulator):
    
    def __init__(self, num_samples, num_features, correlated=False, binarize=False) -> None:
        super().__init__(num_samples, num_features, correlated, binarize)
        
    def formula(self, X):
        
        y = np.exp(X[:, 0:10].sum(axis=1))

        return y 
    
    def noise(self, num_features):
        return np.zeros(num_features)
    
    def binarize(self, y):
        return ((y / (y + 1)) > 0.5).astype(float)
    
if __name__ == "__main__":
    Categorical1(int("${NUM_SAMPLES}"), int("${NUM_FEATURES}", False, True))
