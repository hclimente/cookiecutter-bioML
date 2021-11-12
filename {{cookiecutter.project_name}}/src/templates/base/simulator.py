from abc import abstractmethod

import numpy as np

import utils as u


class Simulator:
    def __init__(self, num_samples, num_features, correlated, binarize):
        self.X = self.make_X(num_samples, num_features, correlated)
        self.y = self.formula(self.X) + self.noise(num_samples)
        self.featnames = np.arange(num_features)

        u.save_scores_npz(self.featnames, [f in self.causal for f in self.featnames])

        if binarize:
            self.y = self.binarize(self.y)

        np.savez("simulation.npz", X=self.X, y=self.y, featnames=self.featnames)

    @abstractmethod
    def formula(self, X):
        raise NotImplementedError

    def make_X(self, num_samples, num_features, correlated):
        mean = np.zeros(num_features)
        cov = self.make_covariance(num_features, correlated)
        X = np.random.multivariate_normal(mean, cov, size=num_samples)

        return X

    def noise(self, num_samples):
        return np.random.normal(loc=0.0, scale=1, size=num_samples)

    def binarize(self, y):
        return np.where(y > 0, 1, -1)

    def make_covariance(self, num_features, correlated, power=2):
        if correlated:
            dist_diagonal = np.zeros(shape=(num_features, num_features))
            for i in range(1, num_features):
                l_idx = np.arange(num_features - i)
                r_idx = l_idx + i
                dist_diagonal[l_idx, r_idx] = i
            cov = 1 / np.power(power, dist_diagonal + dist_diagonal.T)
        else:
            cov = np.eye(num_features)

        return cov
