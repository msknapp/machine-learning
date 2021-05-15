import numpy as np
import math


class DistanceCalculator:
    def __init__(self, weights: [float] = None):
        if weights is None:
            weights = [100., 20., 30., 0.01, 3., 5., 0.1, 1.0]
        self.weights = weights

    def home_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        t = []
        # The first column is the neighborhood, it gets some special treatment.
        # If they are the same neighborhood then the distance is zero, otherwise the
        # distance is one regardless of what combination of neighborhoods are found.
        # It seems like the estimator may be standardizing or averaging some things so the
        # values are not exact.
        if math.fabs(a[0] - b[0]) > 0.4:
            t.append(self.weights[0] * 1.)
        else:
            t.append(0.)
        for i in range(1, len(self.weights)):
            weight = self.weights[i]
            va = a[i]
            vb = b[i]
            t.append(weight * math.fabs(vb - va))
        v = np.array(t)
        distance = np.linalg.norm(v)
        return distance
