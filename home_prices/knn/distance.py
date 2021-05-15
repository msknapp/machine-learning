import numpy as np


class DistanceFunc:
    def __init__(self, weights: dict, noncontinuous: [int]):
        self.weights: dict = weights
        self.noncontinuous = noncontinuous

    def distance(self, a: np.ndarray, b: np.ndarray):
        c = []
        for k in self.weights:
            va: float = a[k]
            vb: float = b[k]
            d = float(va-vb)
            if k in self.noncontinuous and d > 0.5:
                d = 1.0
            d *= self.weights[k]
            c.append(d)
        v = np.array(c)
        distance = np.linalg.norm(v)
        return distance
