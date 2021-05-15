from sklearn import tree, linear_model,
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np



column_names = ["survived", "male", "age", "fare_dollar", "embark_num", "their_class", "parents_or_children"]
data = pd.read_csv("data/prepped/train-basic.csv", names=column_names)

labels = data["survived"]
rows = data.drop("survived", axis=1)
# [["male", "age", "fare_dollar", "embark_num", "their_class", "parents_or_children"]]
# clf = tree.DecisionTreeClassifier()
# clf.fit(rows, labels)
# output_score = clf.score(rows, labels)
# print(output_score)

sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(rows, labels)
output_score = sgd.score(rows, labels)
print(output_score)



# scores = cross_val_score(clf, rows, labels, cv=5)
# print(scores)
# print("mean: "+str(scores.mean()))
# print("standard deviation: "+str(scores.std()))

# k = 5
# holdback = int(data.shape[0] / k)
# for i in range(0,k):
#     start_index = holdback * i
#     end_index = holdback * (i + 1)
#     training_data = data[:start_index]
#     right = data[end_index:]
#     training_data.append(right)
#     labels = training_data["survived"]
#     rows = training_data[["male", "age", "fare_dollar", "embark_num", "their_class", "parents_or_children"]]
#     clf = tree.DecisionTreeClassifier()
#     clf.fit(rows, labels)
#
#     tree.plot_tree(clf)
