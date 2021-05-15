

class FeatureCorrelation:
    def __init__(self, first: str, second: str, covariance: float):
        self.first = first
        self.second = second
        self.covariance = covariance

    def __str__(self):
        return "{} {} {}".format(self.first, self.second, self.covariance)
