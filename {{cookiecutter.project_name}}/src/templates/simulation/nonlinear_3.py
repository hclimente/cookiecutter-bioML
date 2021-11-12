#!/usr/bin/env python

import numpy as np

from base.simulator import Simulator


class NonLinear3(Simulator):
    def __init__(
        self, num_samples, num_features, correlated=False, binarize=False
    ) -> None:
        super().__init__(num_samples, num_features, correlated, binarize)

    def formula(self, X):

        lam = np.exp(X[:, 0:100:10].sum(axis=1))
        y = np.random.poisson(lam=lam, size=X.shape[0]).astype(float)

        return y

    def noise(self, num_samples):
        return np.zeros(num_samples)


if __name__ == "__main__":
    NonLinear3(int("${NUM_SAMPLES}"), int("${NUM_FEATURES}"), True)
