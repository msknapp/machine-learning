from datetime import datetime


class Event:
    def __init__(self, features: [int], label=-1, identity=-1, predicted_label=0, probability=0.0):
        self.label = label
        self.features = features
        self.identity = identity
        self.predicted_label = predicted_label
        self.probability = probability

    def is_label(self, label) -> bool:
        return self.label == label

    def is_positive(self) -> bool:
        return self.label == 1

    def get(self, col) -> int:
        return self.features[col]

    def size(self) -> int:
        return len(self.features)


def load_events(filename: str, test=False) -> [Event]:
    out: [Event] = []
    with open("data/prepped/{}.csv".format(filename)) as inputstream:
        for row in inputstream:
            event_s = row.split(",")
            event = [int(x) for x in event_s]
            identity = -1
            positive = -1
            if test:
                identity = event[len(event) - 1]
                features = event[:len(event) - 1]
            else:
                positive = event[0]
                features = event[1:]
            x = Event(features=features, identity=identity, label=positive)
            out.append(x)
    return out


def export_scores(events: [Event], filename: str = None, include_label=False, include_probability=False,
                  include_features=False):
    if filename is None:
        filename = datetime.utcnow().strftime("%Y-%m-%d_%H:%M:%S")
    with open("data/predictions/{}.csv".format(filename), 'x') as outputstream:
        anything_written = False
        for event in events:
            if anything_written:
                outputstream.write("\n")
            outputstream.write(str(event.identity))
            outputstream.write(",")
            outputstream.write(str(event.score))
            if include_label:
                outputstream.write(",")
                outputstream.write(str(event.positive))
            if include_probability:
                outputstream.write(",")
                outputstream.write(str(event.probability))
            if include_features:
                outputstream.write(",")
                outputstream.write(",".join([str(x) for x in event.features]))
            anything_written = True
