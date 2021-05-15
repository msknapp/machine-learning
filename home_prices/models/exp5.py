import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from home_prices.models.cluster_features import FeatureGetter
from sklearn.impute import SimpleImputer
import math


class ExperimentalTransformerFive(TransformerMixin):

    def __init(self, getter: FeatureGetter = None):
        if getter is None:
            getter = FeatureGetter()

        # these things need to be fit.
        self.imputer = SimpleImputer(strategy='most_frequent')
        # self.categorical_mapping = load_categorical_mapping()
        # self.categorical_transform = CategoricalTransformer(feature_names=getter.feature_names,
        #                                                     mapping=self.categorical_mapping)
        self.model = None
        self.is_fit = False
        self.getter = getter

    def _preprocess(self, x, operation: str = 'fit'):
        x = self._setup(x)
        if operation == 'fit':
            x = self.imputer.fit_transform(x)
        else:
            x = self.imputer.transform(x)
        grla = self.getter.get_column(x, 'GrLivArea')
        bsma = self.getter.get_column(x, 'TotalBsmtSF')
        total_area = (grla + bsma)
        area_feature = total_area / 6000.0
        overall_quality = self.getter.get_column(x, 'OverallQual')
        quality_feature = np.exp(overall_quality / 10.0) / math.e

        yr_sold = self.getter.get_column(x, 'YrSold')
        yr_built = self.getter.get_column(x, 'YearBuilt')
        age = yr_sold - yr_built
        age_feature = np.exp(- age / 65.0)


        pass

    def fit(self, x, y):
        return self._preprocess(x, y, operation='fit')

    def transform(self, x):
        return self._preprocess(x, operation='transform')

    def fit_transform(self, x):
        return self._process(x, operation='fit_transform')

    def get_params(self, deep=True):
        return {
            "core_type": self.core_type,
            "output_transform": self.output_transform,
        }

    def _setup(self, x):
        x0 = x.copy() if not isinstance(x, pd.DataFrame) else x.to_numpy()
        x1 = self.categorical_transform.transform(x0, exclude_columns=['YrSold', 'MoSold', 'YearBuilt'])
        return x1
