from titanic.basic.event import Event


class Node:
    def __init__(self, pivot_feature=None):
        self.sub_nodes = None
        self.total = 0
        self.totals_by_label = None

    def has_subnodes(self) -> bool:
        if self.sub_nodes is None:
            return False
        return len(self.sub_nodes) > 0

    def get_subnode(self, feature_value: int):
        if self.sub_nodes is None:
            self.sub_nodes = {}
        if feature_value not in self.sub_nodes:
            self.sub_nodes[feature_value] = Node()
        return self.sub_nodes[feature_value]

    def increment(self, label):
        if self.totals_by_label is None:
            self.totals_by_label = {}
        self.total += 1
        if label not in self.totals_by_label:
            self.totals_by_label[label] = 1
        else:
            self.totals_by_label[label] += 1

    def get_probability_of(self, label) -> float:
        if self.totals_by_label is None:
            return 0.0
        if label not in self.totals_by_label:
            return 0.0
        n = self.totals_by_label[label]
        return float(n) / float(self.total)


class DecisionTreeModel:
    def __init__(self, pivot_feature_indices: [int]):
        self.root_node = Node()
        self.pivot_feature_indices = pivot_feature_indices

    def reset(self):
        self.root_node = Node()

    def estimate_probability(self, event: Event) -> float:
        node = self.root_node
        for pivot_feature_index in self.pivot_feature_indices:
            feature_value = event.get(pivot_feature_index)
            node = node.get_subnode(feature_value)
        return node.get_probability_of(event.label)

    def predict(self, event: Event):
        p = self.estimate_probability(event)
        if p > 0.500:
            return 1
        return 0

    def consider(self, event: Event):
        node = self.root_node
        for pivot_feature_index in self.pivot_feature_indices:
            feature_value = event.get(pivot_feature_index)
            node = node.get_subnode(feature_value)
        node.increment(event.label)
