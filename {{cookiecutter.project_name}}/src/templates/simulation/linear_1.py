#!/usr/bin/env python

from base.simulator import Simulator


class Linear1(Simulator):
    def __init__(
        self, num_samples, num_features, correlated=False, binarize=False
    ) -> None:
        super().__init__(num_samples, num_features, correlated, binarize)

    def formula(self, X):

        x1 = X[:, 0]
        x2 = X[:, 10]
        x3 = X[:, 20]
        x4 = X[:, 30]

        y = x1 + 2 * x2 + 4 * x3 + 8 * x4

        return y


if __name__ == "__main__":
    Linear1(int("${NUM_SAMPLES}"), int("${NUM_FEATURES}"), True)
