from titanic.basic.event import Event


class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def reset(self):
        for t in self.models:
            t.reset()

    def estimate_probability(self, event: Event) -> float:
        sum = 0.0
        for t in self.models:
            sum += t.estimate_probability(event)
        return sum / float(len(self.trees))

    def predict(self, event: Event):
        p = self.estimate_probability(event)
        if p > 0.500:
            return 1
        return 0

    def consider(self, event: Event):
        for t in self.trees:
            t.consider(event)
