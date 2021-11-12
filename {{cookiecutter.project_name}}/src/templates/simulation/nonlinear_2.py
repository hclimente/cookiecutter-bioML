#!/usr/bin/env python

from base.simulator import Simulator


class NonLinear2(Simulator):
    def __init__(
        self, num_samples, num_features, correlated=False, binarize=False
    ) -> None:
        super().__init__(num_samples, num_features, correlated, binarize)

    def formula(self, X):

        self.causal = list(range(0, 31, 10))
        X = X[:, self.causal]

        x1 = 3 * X[:, 0]
        x2 = 3 * X[:, 1] ** 3
        x3 = 3 * X[:, 2] ** -1
        x4 = 5 * (X[:, 3] > 0).astype(int)

        y = x1 + x2 + x3 + x4

        return y


if __name__ == "__main__":
    NonLinear2(int("${NUM_SAMPLES}"), int("${NUM_FEATURES}"), True)
