from titanic.basic.event import Event
import math


class NeighboringEvent:
    def __init__(self, distance, event):
        self.distance = distance
        self.event = event


class NeighboringEvents:
    def __init__(self, max_size=5):
        self.neighboring_events = []
        self.max_size = max_size
        self.longest_distance = -1

    def recalculate_longest_distance(self):
        farthest_distance = -1
        for n in self.neighboring_events:
            if n.distance > farthest_distance:
                farthest_distance = n.distance
        self.longest_distance = farthest_distance

    def remove_farthest_neighbor(self):
        farthest_distance = -1
        farthest_index = -1
        index = 0
        for n in self.neighboring_events:
            if n.distance > farthest_distance:
                farthest_index = index
                farthest_distance = n.distance
            index += 1
        self.neighboring_events.pop(farthest_index)
        self.recalculate_longest_distance()

    def consider(self, event: NeighboringEvent):
        if len(self.neighboring_events) < self.max_size:
            self.neighboring_events.append(event)
            if event.distance > self.longest_distance:
                self.longest_distance = event.distance
            return
        elif event.distance < self.longest_distance:
            self.remove_farthest_neighbor()
            self.neighboring_events.append(event)
            if event.distance > self.longest_distance:
                self.longest_distance = event.distance

    def get_labels(self):
        labels = {}
        for n in self.neighboring_events:
            lbl = n.event.label
            if lbl not in labels:
                labels[lbl] = 1
            else:
                labels[lbl] += 1
        return labels


class NearestNeighborsModel:
    def __init__(self, distance_function, k=5):
        self.distance_function = distance_function
        self.events = []
        self.k = k

    def reset(self):
        self.events = []

    def estimate_probability(self, event: Event) -> float:
        nearest = NeighboringEvents(max_size=self.k)
        threshold = -1
        for known in self.events:
            dist = self.distance_function(known, event)
            x = NeighboringEvent(distance=dist, event=known)
            nearest.consider(x)
        labels = nearest.get_labels()
        if 1 not in labels:
            return 0.0
        count = labels[1]
        return float(count) / float(self.k)

    def predict(self, event: Event):
        p = self.estimate_probability(event)
        if p > 0.500:
            return 1
        return 0

    def consider(self, event: Event):
        self.events.append(event)


def distance_between(event1: Event, event2: Event) -> float:
    features_count = len(event1.features)
    sum_squares = 0.0
    for i in range(0, features_count):
        val1 = float(event1.get(i))
        val2 = float(event2.get(i))
        diff = val2 - val1
        diff_squared = math.pow(diff, 2)
        sum_squares += diff_squared
    return math.sqrt(sum_squares)
