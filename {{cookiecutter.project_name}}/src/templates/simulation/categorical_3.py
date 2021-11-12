#!/usr/bin/env python

import numpy as np

from base.simulator import Simulator


class Categorical3(Simulator):
    def __init__(
        self, num_samples, num_features, correlated=False, binarize=False
    ) -> None:
        super().__init__(num_samples, num_features, correlated, binarize)

    def formula(self, X):

        self.causal = np.array(range(0, 91, 10))
        X = X[:, self.causal]

        y = X.sum(axis=1) + 10 ** 0.5

        return y

    def noise(self, num_samples):
        return np.zeros(num_samples)

    def binarize(self, y):
        def f(x):
            if x < 0:
                y = 0
            elif x < 2:
                y = 1
            elif x < 4:
                y = 2
            elif x < 6:
                y = 3
            elif x < 8:
                y = 4
            else:
                y = 5

            return y

        y = np.vectorize(f)(y)

        return y


if __name__ == "__main__":
    Categorical3(int("${NUM_SAMPLES}"), int("${NUM_FEATURES}"), True, True)
