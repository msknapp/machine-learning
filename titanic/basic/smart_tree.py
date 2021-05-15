import numpy as np
from titanic.basic.event import Event
from scipy import stats
import math


class SmartNode:
    def __init__(self):
        self.splits_on_feature = None
        self.split_on_value = 0
        self.left = None
        self.right = None
        self.label = -1

    def estimate_probability(self, event: Event) -> float:
        if self.splits_on_feature is not None:
            vlu = event.get(self.splits_on_feature)
            if vlu < self.split_on_value:
                return self.left.estimate_probability(event)
            else:
                return self.right.estimate_probability(event)
        else:
            return self.label


class LearningNode:
    def __init__(self, events: [Event] = None, categorical_indices: [int] = None):
        self.splits_on_feature = None
        self.split_on_value = 0
        self.split_relation = None
        self.events = events if events is not None else []
        self.categorical_indices = categorical_indices
        self.left = None
        self.right = None

    def split(self, min_leaves = 10):
        if len(self.events) < min_leaves:
            return
        survival = np.array([t.label for t in self.events])
        best_corr = -1.0
        best_feature_split = -1
        best_corr_data = None
        for index in range(0, self.events[0].size()):
            data = np.asarray([t.get(index) for t in self.events])
            this_corr_m = np.corrcoef(data, survival)
            this_corr = math.fabs(this_corr_m[0,1])
            if this_corr > best_corr:
                best_corr = this_corr
                best_feature_split = index
                best_corr_data = data
        # now we know what feature we want to split on, but where should we split it?
        # For now just split by average.
        if best_corr_data is None:
            return
        avg = best_corr_data.mean()
        left = []
        right = []
        e: Event
        for e in self.events:
            vlu = e.get(best_feature_split)
            if vlu < avg:
                left.append(e)
            else:
                right.append(e)
        self.splits_on_feature = best_feature_split
        self.split_on_value = avg
        self.left = LearningNode(left)
        self.right = LearningNode(right)
        self.left.split(min_leaves)
        self.right.split(min_leaves)

    def reset(self):
        self.splits_on_feature = None
        self.split_on_value = 0
        self.split_relation = None
        self.events = []
        self.left = None
        self.right = None

    def consider(self, event: Event):
        self.events.append(event)

    def finish_training(self):
        self.split(10)
        return self.export()

    def export(self):
        node = SmartNode()
        self.update(node)
        return node

    def update(self, node: SmartNode):
        node.splits_on_feature = self.splits_on_feature
        node.split_on_value = self.split_on_value
        if self.left is not None:
            node.left = SmartNode()
            self.left.update(node.left)
        if self.right is not None:
            node.right = SmartNode()
            self.right.update(node.right)
        x = np.array([s.label for s in self.events])
        md = stats.mode(x, axis=None)
        k = md[0][0]
        node.label = k

