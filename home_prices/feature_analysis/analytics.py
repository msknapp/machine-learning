from home_prices.feature_analysis.feature_correlation import FeatureCorrelation
from home_prices.feature_analysis.feature_value import FeatureAndValue
import numpy as np
import pandas as pd
import math


def compute_most_correlated_features(fcs: [FeatureCorrelation]) -> [FeatureAndValue]:
    data = {}
    fc: FeatureCorrelation
    for fc in fcs:
        if fc.first not in data:
            data[fc.first] = FeatureAndValue(fc.first, 0.0)
        data[fc.first].add(fc.covariance)
        if fc.second not in data:
            data[fc.second] = FeatureAndValue(fc.second, 0.0)
        data[fc.second].add(fc.covariance)
    out = []
    for v in data.values():
        out.append(v)
    return out


def sort_feature_and_value(fvs: [FeatureAndValue]):
    fvs.sort(reverse=True, key=lambda j: j.value)


def sort_feature_correlations(x: [FeatureCorrelation]):
    x.sort(reverse=True, key=lambda z: z.covariance)


def determine_correlations(feature_names: [str], x: np.ndarray) -> [FeatureCorrelation]:
    out: [FeatureCorrelation] = []
    for i in range(0, len(feature_names) - 1):
        first = feature_names[i]
        for j in range(i + 1, len(feature_names)):
            second = feature_names[j]
            xi: np.ndarray = x[i]
            xj: np.ndarray = x[j]
            cov = np.corrcoef(xi, xj)[0, 1]
            cov = math.fabs(cov)
            fc = FeatureCorrelation(first, second, cov)
            out.append(fc)
    return out


def determine_output_correlation(feature_names: [str], x: np.ndarray, y: np.ndarray) -> [FeatureCorrelation]:
    out: [FeatureCorrelation] = []
    for i in range(0, len(feature_names)):
        first = feature_names[i]
        xi: np.ndarray = x[:, i]
        cov = np.corrcoef(xi, y)[0, 1]
        cov = math.fabs(cov)
        fc = FeatureCorrelation(first, "value", cov)
        out.append(fc)
    return out


def get_most_redundant_features(feature_names, x) -> [FeatureAndValue]:
    cm: [FeatureCorrelation] = determine_correlations(feature_names, x)
    cf: [FeatureAndValue] = compute_most_correlated_features(cm)
    sort_feature_and_value(cf)
    return cf


def print_most_redundant_features(feature_names, x):
    cf: [FeatureAndValue] = get_most_redundant_features(feature_names, x)
    print_feature_values(cf)


def print_feature_values(fvs: [FeatureAndValue]):
    for t in fvs:
        print(t)


def print_feature_correlations(fcs: [FeatureCorrelation]):
    t: FeatureCorrelation
    for t in fcs:
        print(t)


def print_inner_feature_correlation(feature_names, x):
    cm : [FeatureCorrelation] = determine_correlations(feature_names, x)
    print_feature_correlations(cm)


def get_distinct_values(feature_names: [str], x: np.ndarray) -> [FeatureAndValue]:
    out = []
    for i in range(0, len(feature_names)):
        name = feature_names[i]
        values: np.ndarray = x[:,i]
        v = np.unique(values).size
        out.append(FeatureAndValue(name, v))
    return out


def print_distinct_values(feature_names: [str], x: np.ndarray):
    dvs = get_distinct_values(feature_names, x)
    sort_feature_and_value(dvs)
    print_feature_values(dvs)


def count_missing_values(data: pd.DataFrame) -> [FeatureAndValue]:
    out = []
    for f in data.keys():
        x = data[f]
        t = x[pd.isna(x)].size
        out.append(FeatureAndValue(f, t))
    return out

