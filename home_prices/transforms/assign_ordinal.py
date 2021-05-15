import numpy as np
import pandas as pd
from home_prices.dataload import load_training_data, load_feature_names
import pickle
from typing import Dict
from scipy import stats


class CategoricalMapper:
    def __init__(self, mapping: Dict[str, float], default_value: float = 0.0, feature_name: str = None):
        self.mapping = mapping
        self.default_value = default_value
        self.feature_name = feature_name

    def convert(self, value: str) -> float:
        if value in self.mapping:
            return self.mapping[value]
        return self.default_value

    def convert_all(self, values: np.ndarray) -> np.ndarray:
        return np.array(list(map(self.convert, values)))


class CategoricalMapping:
    def __init__(self, mapping: Dict[str, CategoricalMapper]):
        self.mapping = mapping

    def keys(self) -> [str]:
        return self.mapping.keys()

    def supports(self, category: str) -> bool:
        return category in self.mapping

    def convert(self, category: str, value: str) -> float:
        if category in self.mapping:
            converter: CategoricalMapper = self.mapping[category]
            return converter.convert(value)
        return 0.0

    def convert_all(self, category: str, x: np.ndarray) -> np.ndarray:
        if category in self.mapping:
            converter: CategoricalMapper = self.mapping[category]
            return converter.convert_all(x)
        return x


class CategoricalTransformer:
    def __init__(self, feature_names: [str], mapping: CategoricalMapping):
        if feature_names is None:
            feature_names = load_feature_names()
        self.feature_names = feature_names
        self.mapping = mapping

    def column_number(self, feature_name: str) -> int:
        return self.feature_names.index(feature_name)

    def fit(self, x, y=None):
        return self

    def transform(self, x: np.ndarray, exclude_columns: [str] = None, include_columns: [str] = None) -> np.ndarray:
        out = x.copy()
        for name in self.mapping.keys():
            if exclude_columns is not None and name in exclude_columns:
                continue
            if include_columns is not None and len(include_columns) > 0 and name not in include_columns:
                continue
            col = self.column_number(name)
            dta = out[:, col]
            dta = self.mapping.convert_all(name, dta)
            if np.isnan(dta).any():
                md = stats.mode(dta)[0][0]
                dta[np.isnan(dta)] = md
            out[:, col] = dta
        return out


def assign_ordinal(x: pd.Series, y: pd.Series):
    d = {}
    unq: pd.Series = x.unique()
    min: float = None
    max: float = None
    for s in unq:
        ys = y[x == s]
        avg = np.average(ys)
        d[s] = avg
        min = avg if min is None or avg < min else min
        max = avg if max is None or avg > max else max
    for s in d:
        v: float = (d[s] - min) / (max - min)
        d[s] = round(v, ndigits=3)
    return d


def save_categorical_mapping(o: CategoricalMapping):
    with open('../objects/categorical-mapping.pickle', 'w+b') as f:
        pickle.dump(o, f)


def load_categorical_mapping() -> CategoricalMapping:
    with open('../objects/categorical-mapping.pickle', 'rb') as f:
        return pickle.load(f)


def create_categorical_mapping(threshold: int = 30) -> CategoricalMapping:
    features: pd.DataFrame
    values: pd.DataFrame
    features, values = load_training_data()
    tmp = {}
    for k in features.keys():
        feat: pd.Series = features[k]
        if feat.dtype == float:
            continue
        if feat.unique().size > threshold:
            continue
        ord = assign_ordinal(feat, values)
        tmp[k] = CategoricalMapper(ord)
    return CategoricalMapping(tmp)


def try_it_out():
    mpng = create_categorical_mapping()
    save_categorical_mapping(mpng)
    rec = load_categorical_mapping()

    features, values = load_training_data()
    tf = CategoricalTransformer(feature_names=features.keys().to_list(), mapping=rec)
    result = tf.transform(features.to_numpy())
    print(result)
