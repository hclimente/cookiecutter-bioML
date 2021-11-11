#!/usr/bin/env python

import numpy as np

from base.simulator import Simulator


class NonLinear5(Simulator):
    def __init__(
        self, num_samples, num_features, correlated=False, binarize=False
    ) -> None:
        super().__init__(num_samples, num_features, correlated, binarize)

    def formula(self, X):

        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]

        t1 = 5 * (x2 + x3) ** -3
        t2 = np.exp(1 + 10 * np.sin(np.pi * x1 / 2) + 5 * x4)

        y = 1 - t1 * t2

        return y


if __name__ == "__main__":
    NonLinear5(int("${NUM_SAMPLES}"), int("${NUM_FEATURES}"), True)
