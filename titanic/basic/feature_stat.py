from titanic.basic.feature_value_stat import FeatureValueStat


class FeatureStat:
    def __init__(self):
        self.values = {}

    def get_feature_value_stats(self, value) -> FeatureValueStat:
        if value not in self.values:
            return None
        return self.values[value]

    def increment(self, value, positive):
        if value not in self.values:
            self.values[value] = FeatureValueStat()
        fvs = self.values[value]
        fvs.increment(positive)

    def total_events(self) -> int:
        sum = 0
        for value in self.values:
            fvs: FeatureValueStat = self.values[value]
            sum += fvs.total
        return sum

    def total_positive_events(self) -> int:
        sum = 0
        for value in self.values:
            fvs: FeatureValueStat = self.values[value]
            sum += fvs.total_positive
        return sum

    def probability_of_value(self, value) -> float:
        if value not in self.values:
            return -1
        fvs: FeatureValueStat = self.values[value]
        num = fvs.total
        den = self.total_events()
        if den == 0:
            return -2
        return float(num) / float(den)

    def probability_of_value_positive(self, value, positive) -> float:
        fvs: FeatureValueStat = self.get_feature_value_stats(value)
        if fvs is None:
            return 0.0
        return fvs.probability_of(positive)

    def __str__(self):
        x = []
        col = 0
        for v in self.values:
            x = self.values[v]
            s = str(x)
            x.append("{}=>{}".format(col, s))
            col += 1
        return ",".join(x)

    def probability_of_value_given_positive(self, value, param):
        fvs = self.get_feature_value_stats(value)
        if fvs is None:
            return -1
        fvs_pos = fvs.total_positive
        all_pos = self.total_positive_events()
        if all_pos == 0:
            return -2.0
        return float(fvs_pos) / float(all_pos)

