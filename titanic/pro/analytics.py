import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("data/original/train.csv")
r = df["Parch"].value_counts()
fig, ax = plt.subplots()
ax.bar(r.keys(), r.values)
plt.show()
