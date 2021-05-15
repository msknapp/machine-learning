from titanic.basic.feature_stat import FeatureStat
import json
from titanic.basic.event import Event


# create a naive bayes classifier.
class NaiveBayesModel:

    def __init__(self):
        self.total_events = 0
        self.total_positive_events = 0
        self.feature_statistics = {}

    def estimate_probability(self, event: Event):
        probability_positive = float(self.total_positive_events) / float(self.total_events)
        column = 0
        cumulative_numerator = 1.0
        cumulative_denominator = 1.0
        for value in event.features:
            stat: FeatureStat = self.feature_statistics[column]

            prob_of_value_given_positive = stat.probability_of_value_given_positive(value, True)
            if prob_of_value_given_positive < -0.001:
                continue
            prob_of_value = stat.probability_of_value(value)
            if prob_of_value < -0.001:
                continue

            cumulative_denominator *= prob_of_value
            cumulative_numerator *= prob_of_value_given_positive
            column += 1
        cumulative_numerator *= probability_positive
        p = cumulative_numerator / cumulative_denominator
        return p

    def predict(self, event: Event):
        p = self.estimate_probability(event)
        if p > 0.500:
            return 1
        return 0

    def consider(self, event: Event):
        # requirement, all input columns should have a few discrete sets of values, nothing continuous.
        self.total_events += 1
        column = 0
        positive = event.is_positive()
        if positive:
            self.total_positive_events += 1
        for feature_value in event.features:
            if column not in self.feature_statistics:
                self.feature_statistics[column] = FeatureStat()
            feature_stat: FeatureStat = self.feature_statistics[column]
            feature_stat.increment(feature_value, positive)
            column += 1

    def reset(self):
        self.total_events = 0
        self.total_positive_events = 0
        self.feature_statistics = {}

    def dump(self):
        return json.dumps(self)
