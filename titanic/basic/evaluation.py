from titanic.basic.event import Event


class ModelEvaluation:
    def __init__(self):
        self.actual_labels = {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}}

    def increment(self, actual_label, predicted_label):
        if actual_label not in self.actual_labels:
            self.actual_labels[actual_label] = {}
        x = self.actual_labels[actual_label]
        if predicted_label not in x:
            x[predicted_label] = 0
        x[predicted_label] += 1

    def total_predictions(self) -> int:
        sum = 0
        for actual_label in self.actual_labels:
            predictions = self.actual_labels[actual_label]
            for predicted_label in predictions:
                sum += predictions[predicted_label]
        return sum

    def total_correct_predictions(self) -> int:
        sum = 0
        for actual_label in self.actual_labels:
            predicted = self.actual_labels[actual_label]
            sum += predicted[actual_label]
        return sum

    def accuracy(self) -> float:
        correct = self.total_correct_predictions()
        tot = self.total_predictions()
        return float(correct) / float(tot)


def evaluate(events: [Event]) -> ModelEvaluation:
    eval = ModelEvaluation()
    for event in events:
        actual = event.label
        predicted = event.predicted_label
        eval.increment(actual, predicted)
    return eval
