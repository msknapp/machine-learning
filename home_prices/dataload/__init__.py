import pandas as pd
import regex


def load_feature_names() -> [str]:
    with open('../data/train.csv') as f:
        header = f.readline()
    parts = header.split(",")
    t = parts[1:len(parts)-1]
    return t


def load_training_data(no_shuffle: bool = False, as_numpy: bool = False, remove_indices = None) -> (pd.DataFrame, pd.DataFrame):
    data: pd.DataFrame = pd.read_csv("../data/train.csv", index_col="Id")
    if remove_indices is not None:
        data = data.drop(remove_indices)
    if not no_shuffle:
        data = data.sample(frac=1)
    values: pd.DataFrame = data.pop('SalePrice')
    values = values.astype(dtype=float)
    if not as_numpy:
        return data, values
    return data.to_numpy(), values.to_numpy()


def load_prepped_data(no_shuffle: bool = False) -> (pd.DataFrame, pd.DataFrame):
    data: pd.DataFrame = pd.read_csv("../data/prepped.csv", index_col="Id")
    if not no_shuffle:
        data = data.sample(frac=1)
    values: pd.DataFrame = data.pop('saleprice')
    values = values.astype(dtype=float)
    return data, values


def load_low_variance_feature_names() -> [str]:
    out = []
    with open("../data/analysis/low-variance-features.txt") as f:
        for line in f:
            if line != "":
                out.append(line.strip())
    return out


def load_features_with_high_missing_values(threshold: int = 5) -> [str]:
    out = []
    with open("../data/analysis/missing-values.txt") as f:
        for line in f:
            if line != "":
                parts = regex.split("\\s+", line)
                num = int(parts[1])
                if num > threshold:
                    out.append(parts[0])
    return out


def combine_as_set(a, b) -> [str]:
    out = []
    for s in a:
        out.append(s)
    for s in b:
        if s not in out:
            out.append(s)
    return out


def remove_as_set(a: [str], b: [str]) -> [str]:
    out = []
    for s in a:
        if s not in b:
            out.append(s)
    return out


def indexes_of(names: [str], subset: [str]) -> [int]:
    out = []
    for s in subset:
        i = names.index(s)
        out.append(i)
    return out


def load_test_data():
    data: pd.DataFrame = pd.read_csv("../data/test.csv", index_col="Id")
    return data
