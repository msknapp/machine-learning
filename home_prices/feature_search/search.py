from home_prices.dataload import load_training_data
from sklearn.feature_selection import chi2
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import numpy as np
from home_prices.feature_analysis import FeatureAndValue, sort_feature_and_value, print_feature_values
from sklearn.decomposition import PCA


def get_chi2_scores():
    features, values = load_training_data()
    names = features.keys()
    imp = SimpleImputer(strategy='most_frequent')
    x = imp.fit_transform(features, values)
    enc = OrdinalEncoder()
    x = enc.fit_transform(x, values)
    c2, pval = chi2(x, values)
    out = []
    for i in range(0, len(names)):
        fv = FeatureAndValue(names[i], pval[i])
        out.append(fv)
    sort_feature_and_value(out)
    return out


def print_chi2_scores():
    t = get_chi2_scores()
    print_feature_values(t)


def get_pca():
    features, values = load_training_data()
    names = features.keys()
    imp = SimpleImputer(strategy='most_frequent')
    x = imp.fit_transform(features, values)
    enc = OrdinalEncoder()
    x = enc.fit_transform(x, values)
    pca = PCA(n_components=8)
    out = pca.fit_transform(x, values)
    print(pca.components_)
    print(out)


get_pca()
