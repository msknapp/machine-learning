from sklearn.cluster import AgglomerativeClustering
from home_prices.dataload import load_training_data, load_feature_names
import numpy as np
from home_prices.transforms.assign_ordinal import CategoricalTransformer, load_categorical_mapping
from sklearn.impute import SimpleImputer


# TODO find clusters for features.


class FeatureGetter:
    def __init__(self, feature_names: [str] = None):
        if feature_names is None:
            feature_names = load_feature_names()
        self.feature_names = feature_names

    def column_number(self, feature_name: str) -> int:
        return self.feature_names.index(feature_name)

    def get_column(self, feature_name: str, x: np.ndarray) -> np.ndarray:
        num = self.column_number(feature_name)
        return x[:, num]


feature_names = load_feature_names()


def get_feature_distance_matrix():
    getter = FeatureGetter(feature_names)
    # agg = FeatureAgglomeration(n_clusters=12)
    categorical_mapping = load_categorical_mapping()
    categorical_transform = CategoricalTransformer(feature_names=getter.feature_names, mapping=categorical_mapping)
    x, _ = load_training_data(as_numpy=True)
    x = categorical_transform.transform(x)
    imputer = SimpleImputer(strategy='most_frequent')
    x = imputer.fit_transform(x)
    distance_matrix = []
    for row_feature_name in getter.feature_names:
        row = []
        for col_feature_name in getter.feature_names:
            if row_feature_name == col_feature_name:
                row.append(1.0)
            else:
                row_data = getter.get_column(row_feature_name, x)
                col_data = getter.get_column(col_feature_name, x)
                row_data = row_data.astype(np.float64)
                col_data = col_data.astype(np.float64)
                if np.isnan(row_data).any() > 0:
                    continue
                if np.isnan(col_data).any() > 0:
                    continue
                corr: np.ndarray = np.corrcoef(row_data, col_data)
                pearson = corr[0][1]
                # will be between -1 and 1.  A 0 means there is no relation.  A 1 or -1 means there is a perfect
                # correlation
                if np.isnan(pearson):
                    pearson = 0.0
                row.append(pearson)
        distance_matrix.append(row)
    return np.array(distance_matrix)


def print_feature_matrix(feature_matrix):
    print("feature," + ",".join(feature_names))
    row = 0
    for fn in feature_names:
        s = fn + ","
        row_data = feature_matrix[row, :]
        t = [str(round(v, 2)) for v in row_data]
        s += ",".join(t)
        print(s)
        row += 1


def to_distance_matrix(feature_matrix: np.ndarray):
    return 1.0 / (np.abs(feature_matrix) + 0.001)


def myrun():
    fm = get_feature_distance_matrix()
    dm = to_distance_matrix(fm)
    print_feature_matrix(dm)
    n_clusters = 16
    fa = AgglomerativeClustering(n_clusters=n_clusters, connectivity=dm)
    out = fa.fit_predict(dm)
    clusters = {}
    for i in range(0, len(feature_names)):
        feature_name = feature_names[i]
        cluster = out[i]
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(feature_name)

    for i in range(0, n_clusters):
        s = "cluster {}: {}".format(i, ",".join(clusters[i]))
        print(s)

    # print(out)
