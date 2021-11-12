#!/usr/bin/env python

import numpy as np

from base.simulator import Simulator


class NonLinear1(Simulator):
    def __init__(
        self, num_samples, num_features, correlated=False, binarize=False
    ) -> None:
        super().__init__(num_samples, num_features, correlated, binarize)

    def formula(self, X):

        x1 = 5 * X[:, 0]
        x2 = 2 * np.sin(np.pi * X[:, 10] / 2)
        x3 = 2 * X[:, 20] * (X[:, 20] > 0).astype(int)
        x4 = 2 * np.exp(5 * X[:, 30])

        y = x1 + x2 + x3 + x4

        return y


if __name__ == "__main__":
    NonLinear1(int("${NUM_SAMPLES}"), int("${NUM_FEATURES}"), True)
