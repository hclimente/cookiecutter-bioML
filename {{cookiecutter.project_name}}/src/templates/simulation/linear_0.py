#!/usr/bin/env python

import numpy as np

from base.simulator import Simulator

class LinearSimulator(Simulator):
    
    def __init__(self, num_samples, num_features) -> None:
        super().__init__(num_samples, num_features)
        
    def formula(self):
        
        x1 = self.X[:, 0]
        x2 = 2*self.X[:, 1]
        x3 = 4*self.X[:, 2]
        x4 = 8*self.X[:, 3]
        
        eps = np.random.normal(loc=0.0, scale=0.1, size=self.X.shape[0])
        
        y = x1 + x2 + x3 + x4 + eps
        y = np.where(y > 0, 1, -1)

        return y 
    
if __name__ == "__main__":
    LinearSimulator(int("${NUM_SAMPLES}"), int("${NUM_FEATURES}"))
