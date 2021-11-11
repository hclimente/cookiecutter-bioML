#!/usr/bin/env python

import numpy as np

from base.simulator import Simulator


class Linear3(Simulator):
    def __init__(
        self, num_samples, num_features, correlated=False, binarize=False
    ) -> None:
        super().__init__(num_samples, num_features, correlated, binarize)

    def formula(self, X):

        y = X[:, 0:10].sum(axis=1)

        return y

    def noise(self, num_samples):
        return np.random.standard_t(2, size=num_samples)


if __name__ == "__main__":
    Linear3(int("${NUM_SAMPLES}"), int("${NUM_FEATURES}"), True)
