from titanic.basic.decision_tree import DecisionTreeModel
from titanic.basic.event import Event


class RandomForestModel:
    def __init__(self, trees: [DecisionTreeModel]):
        self.trees = trees

    def reset(self):
        for t in self.trees:
            t.reset()

    def estimate_probability(self, event: Event) -> float:
        sum = 0.0
        for t in self.trees:
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


def new_random_forest(tree_indices) -> RandomForestModel:
    trees = []
    for pivot_feature_indices in tree_indices:
        dt = DecisionTreeModel(pivot_feature_indices=pivot_feature_indices)
        trees.append(dt)
    return RandomForestModel(trees)
