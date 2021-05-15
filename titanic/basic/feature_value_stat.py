
class FeatureValueStat:

    def __init__(self):
        self.total = 0
        self.total_positive = 0

    def increment(self, positive):
        self.total += 1
        if positive:
            self.total_positive += 1

    def probability_of(self, positive) -> float:
        if self.total == 0:
            return 0.0
        num = self.total_positive if positive else self.total - self.total_positive
        return float(num) / float(self.total)

    def __str__(self):
        return "{}/{}".format(self.total_positive, self.total)
