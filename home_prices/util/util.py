import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import numpy as np
from home_prices.dataload import load_training_data
from home_prices.feature_analysis import *


def impute_missing_values(x: np.ndarray) -> np.ndarray:
    imputer = SimpleImputer(strategy='most_frequent')
    imputer.fit(x)
    x = imputer.transform(x)
    return x


def convert_categorical_features_to_numbers(x: np.ndarray) -> np.ndarray:
    enc = OrdinalEncoder()
    x = enc.fit_transform(x)
    return x


def get_categorical_features(data: pd.DataFrame):
    t = [data[k].dtype.name != 'object' for k in data.keys()]
    return data[t]


def get_continuous_features(data: pd.DataFrame):
    t = [data[k].dtype.name == 'object' for k in data.keys()]
    return data[t]


def determine_categorical_feature_names(data: pd.DataFrame) -> [str]:
    return [k for k in data.keys() if data[k].dtype == 'object']


def determine_continuous_feature_names(data: pd.DataFrame) -> [str]:
    return [k for k in data.keys() if data[k].dtype != 'object']


def determine_variance(p: float) -> float:
    return p * (1 - p)


def determine_low_variance_features(feature_names: [str], x: np.ndarray) -> ([str], [str]):
    variance = determine_variance(0.8)
    sel = VarianceThreshold(threshold=variance)
    sel.fit(x)
    features_acceptable = sel.get_support()
    good_features = [feature_names[i] for i in range(0, len(feature_names)) if features_acceptable[i]]
    bad_features = [feature_names[i] for i in range(0, len(feature_names)) if not features_acceptable[i]]
    return bad_features, good_features


def print_low_variance_features(feature_names, x):
    bad, _ = determine_low_variance_features(feature_names, x)
    for t in bad:
        print(t)


def print_most_inner_correlated_features(feature_names: [str], x: np.ndarray, threshold=0.9):
    cors: [FeatureCorrelation] = determine_correlations(feature_names, x)
    sort_feature_correlations(cors)
    for t in cors:
        if t.covariance > threshold:
            print(t)


def print_most_output_correlated_features(feature_names: [str], x: np.ndarray, y: np.ndarray):
    cors: [FeatureCorrelation] = determine_output_correlation(feature_names, x, y)
    sort_feature_correlations(cors)
    for t in cors:
        print(t)


def print_missing_values():
    features, values = load_training_data()
    mvs = count_missing_values(features)
    sort_feature_and_value(mvs)
    print_feature_values(mvs)


def index_of(a: [str], s: str) -> int:
    return a.index(s)

# chi2_scores, chi2_pvalues = chi2(categorical_data, y)
# feature_analysis: np.ndarray = np.concatenate([[categorical_feature_names], [chi2_pvalues]]).T
# mydf = pd.DataFrame(feature_analysis, columns=["name", "chipvalue"])
# mydf = mydf.sort_values("chipvalue", ascending=False)
# print(mydf)
#
#
# mi_scores = mutual_info_classif(continuous_data, y)
# feature_analysis: np.ndarray = np.concatenate([[continuous_feature_names], [mi_scores]]).T
# mydf = pd.DataFrame(feature_analysis, columns=["name", "m1pvalue"])
# mydf = mydf.sort_values("m1pvalue", ascending=False)
# print(mydf)
#
# f_scores, fp_values = f_classif(x, y)
# feature_analysis: np.ndarray = np.concatenate([[feature_names], [fp_values]]).T
# mydf = pd.DataFrame(feature_analysis, columns=["name", "fpvalue"])
# mydf = mydf.sort_values("fpvalue", ascending=False)
# print(mydf)

# s = StandardScaler()
# x: np.ndarray = s.fit_transform(x)

# good features
# best_selector = SelectKBest(chi2, k=8)
# best_selector.fit(x, y)
# support = best_selector.get_support()
# print(support)

# best_feature_names = ['LotArea' 'MasVnrArea' 'BsmtFinSF1' 'BsmtUnfSF' 'TotalBsmtSF' '1stFlrSF', '2ndFlrSF' 'GrLivArea']
