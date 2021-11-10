from abc import abstractmethod

import numpy as np

class Simulator:

    def __init__(self, num_samples, num_features):
        mean = np.zeros(num_features)
        sigma = np.eye(num_features)
        self.X = np.random.multivariate_normal(mean, sigma, size=num_samples)
        self.y = self.formula()
        self.featnames = np.arange(num_features)
        
        np.savez("simulation.npz", X=self.X, y=self.y, featnames=self.featnames)

    @abstractmethod
    def formula(self):
        raise NotImplementedError
