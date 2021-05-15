from titanic.basic.naive_bayes_model import NaiveBayesModel
from titanic.basic.event import load_events, export_scores
from titanic.basic.runtime import run_kfold_testing, test_model
from titanic.basic.decision_tree import DecisionTreeModel
from titanic.basic.forest import RandomForestModel, new_random_forest
from titanic.basic.nearest_neighbors import NearestNeighborsModel, distance_between
from titanic.basic.smart_tree import LearningNode, SmartNode


model_type = "smart-tree"
if model_type == "bayes":
    model = NaiveBayesModel()
elif model_type == "tree":
    indices = [0, 1, 3]
    model = DecisionTreeModel(pivot_feature_indices=indices)
elif model_type == "forest":
    tree_indices = [
        [0, 1, 2],
        [0, 1],
        [0, 1, 3]
    ]
    model = new_random_forest(tree_indices=tree_indices)
elif model_type == "neighbors":
    model = NearestNeighborsModel(k=12, distance_function=distance_between)
elif model_type == "smart-tree":
    model = LearningNode()

training_events = load_events("train-basic", test=False)
evaluations = run_kfold_testing(model, training_events, 5)
for e in evaluations:
    print(str(e.accuracy()))

# testing_events = load_events('test-basic', test=True)
# test_model(model, testing_events)
# export_scores(testing_events)
