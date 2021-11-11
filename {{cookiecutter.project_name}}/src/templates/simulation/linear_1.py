#!/usr/bin/env python

from base.simulator import Simulator


class Linear1(Simulator):
    def __init__(
        self, num_samples, num_features, correlated=False, binarize=False
    ) -> None:
        super().__init__(num_samples, num_features, correlated, binarize)

    def formula(self, X):

        x1 = X[:, 0]
        x2 = 2 * X[:, 1]
        x3 = 4 * X[:, 2]
        x4 = 8 * X[:, 3]

        y = x1 + x2 + x3 + x4

        return y


if __name__ == "__main__":
    Linear1(int("${NUM_SAMPLES}"), int("${NUM_FEATURES}"), True)
