import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import *
from home_prices.dataload import load_training_data, load_feature_names
from sklearn.metrics import make_scorer, mean_squared_log_error
from home_prices.transforms.assign_ordinal import load_categorical_mapping, CategoricalTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import cross_val_score, GridSearchCV
from home_prices.util import save_model, create_submission
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from home_prices.models.cluster_features import FeatureGetter
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.linear_model import SGDRegressor, RidgeCV
from sklearn.ensemble import VotingRegressor
from sklearn.neural_network import MLPRegressor
import math


class DistanceFunc:
    def __init__(self, getter: FeatureGetter, feature_weights):
        self.getter = getter
        self.feature_weights = feature_weights

    def get_distance(self, x, y, metric=None, missing_values=None):
        vector = []
        for feature in self.feature_weights:
            col = self.getter.column_number(feature)
            delta = x[col] - y[col]
            vector.append(delta)
        vector = np.array(vector)
        distance = np.linalg.norm(vector)
        return distance


class FeatureMerger:
    def __init__(self, produces: str, source_columns: [str], getter: FeatureGetter, normalize: bool = True,
                 operation: str = "pca", morph: str = "none", morph_slope: float = 1.0):
        self.source_columns = source_columns
        self.pca = PCA(n_components=1, whiten=normalize)
        self.produces = produces
        self.merge_operation = operation
        self.normalize = normalize
        self.mean = 0.0
        self.stdev = 1.0
        self.getter = getter
        self.morph = morph
        self.morph_slope = morph_slope

    def _sum(self, x):
        out = x[:, 0]
        for i in range(1, x.shape[1]):
            out = out + x[:, i]
        return out

    def _process(self, x, operation: str = 'fit'):
        columns = []

        for n in self.source_columns:
            col = self.getter.get_column(n, x)
            columns.append(col)
        features = np.column_stack(columns)
        if 'fit' in operation:
            if 'pca' == self.merge_operation:
                self.pca.fit(features)
            else:
                t = self._sum(features)
                self.mean = np.average(t)
                self.stdev = np.std(t)
        if 'transform' not in operation:
            return self
        if 'pca' == self.merge_operation:
            tmp: np.ndarray = self.pca.transform(features)
            out = tmp.reshape([tmp.size])
        elif 'sum' == self.merge_operation:
            out = self._sum(features)
            if self.normalize:
                out = (out - self.mean) / self.stdev
        if self.morph == "log":
            out = np.log(self.morph_slope * out)
        elif self.morph == "exp":
            out = np.exp(self.morph_slope * out)
        elif self.morph == "negexp":
            out = np.exp(-self.morph_slope * out)
        return out

    def fit(self, x):
        return self._process(x, operation='fit')

    def transform(self, x):
        return self._process(x, operation='transform')

    def fit_transform(self, x):
        return self._process(x, operation='fit_transform')


class MyModel(BaseEstimator, RegressorMixin):
    def __init__(self, getter: FeatureGetter = None, core_type: str = 'linear-regression',
                 output_transform: str = 'log',
                 imputation_method: str = "basic", scaling=None, age_half_life: float = 65.0):
        if getter is None:
            getter = FeatureGetter()
        self.core_type = core_type
        self.output_transform = output_transform
        self.age_half_life = 65.0

        # these things need to be fit.
        self.feature_mergers = [
            # FeatureMerger(produces='area', source_columns=['TotalBsmtSF', 'GrLivArea'],
            #               getter=getter, operation='sum'),
            FeatureMerger(produces='baths', source_columns=['BsmtFullBath', 'FullBath'],
                          getter=getter, operation='sum'),
            FeatureMerger(produces='half_baths', source_columns=['BsmtHalfBath', 'HalfBath'],
                          getter=getter, operation='sum'),
            FeatureMerger(produces='access', source_columns=['LotFrontage', 'PavedDrive'],
                          getter=getter),
            # FeatureMerger(produces='shape', source_columns=['LotShape', 'LandContour', 'LandSlope'],
            #               feature_names=self.feature_names),
            FeatureMerger(produces='utilities', source_columns=['Heating', 'HeatingQC'],
                          getter=getter),
            FeatureMerger(produces='quality',
                          source_columns=['OverallQual', 'OverallCond', 'KitchenQual'],
                          getter=getter, morph="exp", morph_slope=0.1),
            FeatureMerger(produces='exterior',
                          source_columns=['LotArea', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
                                          'MasVnrType', 'MasVnrArea', 'ExterCond'],
                          getter=getter),
            FeatureMerger(produces='porch', source_columns=['OpenPorchSF', 'EnclosedPorch', 'ScreenPorch'],
                          getter=getter, operation='sum'),
            FeatureMerger(produces='garage', source_columns=['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars',
                                                             'GarageArea', 'GarageQual', 'GarageCond'],
                          getter=getter),
            FeatureMerger(produces='fireplaces', source_columns=['Fireplaces', 'FireplaceQu'],
                          getter=getter),
            FeatureMerger(produces='basement', source_columns=['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                                                               'BsmtFinType2'],
                          getter=getter),
            # FeatureMerger(produces='conditions', source_columns=['Condition1', 'Condition2'],
            #               feature_names=self.feature_names, operation='sum')
        ]
        self.imputation_method = imputation_method
        if self.imputation_method == 'neighbors':
            df = DistanceFunc(getter=getter, feature_weights={
                "Neighborhood": 100.0,
                "GrLivArea": 0.01,
                "FullBath": 5,
                "BedroomAbvGr": 7,
                "YrSold": 10,
            })
            self.imputer = KNNImputer(metric=df.get_distance)
        else:
            self.imputer = SimpleImputer(strategy='most_frequent')
        self.categorical_mapping = load_categorical_mapping()
        self.categorical_transform = CategoricalTransformer(feature_names=getter.feature_names,
                                                            mapping=self.categorical_mapping)
        self.model = None
        self.is_fit = False
        self.getter = getter
        self.scaling = scaling
        self.scaler = None
        if self.scaling == 'robust':
            self.scaler = RobustScaler()
        elif self.scaling == 'power':
            self.scaler = PowerTransformer()

    def get_params(self, deep=True):
        return {
            "core_type": self.core_type,
            "output_transform": self.output_transform,
            "age_half_life": self.age_half_life,
        }

    def predict(self, x):
        x2 = self.transform(x)
        y2 = self.model.predict(x2)
        y = y2
        if self.output_transform == 'log':
            y = np.exp(y2)
        return y

    def _setup(self, x):
        x0 = x.copy() if not isinstance(x, pd.DataFrame) else x.to_numpy()
        x1 = self.categorical_transform.transform(x0, exclude_columns=['YrSold', 'MoSold', 'YearBuilt'])
        return x1

    def _preprocess(self, x: np.ndarray, y: np.ndarray = None, operation: str = 'fit'):
        overall_quality = self.getter.get_column('OverallQual', x)
        overall_quality = overall_quality.astype(float)
        x = self._setup(x)
        if operation == 'fit':
            x = self.imputer.fit_transform(x)
        else:
            x = self.imputer.transform(x)
        year_sold = self.getter.get_column('YrSold', x)
        month_sold = self.getter.get_column('MoSold', x)
        year_built = self.getter.get_column('YearBuilt', x)
        age = year_sold - year_built

        # TODO a column for the seasonal effect of things
        merged_columns = []
        for merger in self.feature_mergers:
            op = 'fit_transform' if 'fit' in operation else 'transform'
            merged_column = merger._process(x, operation=op)
            merged_columns.append(merged_column)
        merged_columns = np.array(merged_columns).T
        bedrooms = self.getter.get_column("BedroomAbvGr", x)

        year_built = normalize(year_built)
        age = normalize(age)
        bedrooms = normalize(bedrooms)

        # foundation seems to make it worse.
        misc = self.getter.get_column('MiscVal', x)
        misc = normalize(misc)
        neighborhood = self.getter.get_column('Neighborhood', x)
        # zoning seems to help the prediction
        zoning = self.getter.get_column('MSZoning', x)
        pool = self.getter.get_column('PoolArea', x)
        has_pool = np.array(list(map(lambda t: 1.0 if t > 0.0 else 0.0, pool)))
        building_type = self.getter.get_column('BldgType', x)
        house_style = self.getter.get_column('HouseStyle', x)
        functional = self.getter.get_column('Functional', x)
        central_air = self.getter.get_column("CentralAir", x)
        electrical = self.getter.get_column("Electrical", x)
        twoflr = self.getter.get_column("2ndFlrSF", x)
        has_two_floors = np.array(list(map(lambda t: 1.0 if t > 0.0 else 0.0, twoflr)))


        grla = self.getter.get_column('GrLivArea', x)
        bsma = self.getter.get_column('TotalBsmtSF', x)
        total_area = (grla + bsma)
        area_feature = total_area / 6000.0
        tmp = overall_quality / 10.0
        quality_feature = np.exp(tmp) / math.e

        yr_sold = self.getter.get_column('YrSold', x)
        yr_sold = yr_sold.astype(float)
        yr_built = self.getter.get_column('YearBuilt', x)
        yr_built = yr_built.astype(float)
        age = yr_sold - yr_built
        age_feature = np.exp(- age / self.age_half_life)
        combined = age_feature * quality_feature * area_feature
        combined2 = quality_feature * area_feature

        input_data = [combined, combined2, age_feature, quality_feature, area_feature, neighborhood, bedrooms,
                      merged_columns, misc, building_type, house_style]#, functional, central_air, electrical, month_sold,
                      # has_pool, zoning, has_two_floors]
        x3 = np.column_stack(input_data)

        # The robust scaler seems to be helping the accuracy.
        if self.scaler is not None:
            if 'fit' in operation:
                self.scaler.fit(x3)
            if 'transform' in operation:
                x3 = self.scaler.transform(x3)

        if operation == 'transform':
            return x3
        # if self.core_type == 'linear-regression':
        #     core = LinearRegression()
        # elif self.core_type == 'elastic-net':
        #     core = ElasticNet()
        # elif self.core_type == 'perceptron':
        #     core = MLPRegressor(max_iter=1500)

        core1 = ElasticNet()

        self.model = GradientBoostingRegressor(init=core1, n_estimators=100, loss='huber')
        y2 = y.copy()
        if self.output_transform == 'log':
            y2 = np.log(y2)
        self.model.fit(x3, y2)
        self.is_fit = True
        return self

    def fit(self, x, y):
        return self._preprocess(x, y, operation='fit')

    def transform(self, x):
        return self._preprocess(x, operation='transform')


def normalize(x: np.ndarray):
    avg: float = np.average(x)
    std: float = np.std(x)
    if std == 0.0:
        std = 1.0
    n = (x - avg) / std
    return n


def train_and_evaluate_model():
    getter = FeatureGetter()
    model = MyModel(getter=getter, core_type='elastic-net', output_transform='log', imputation_method='basic')
    x, y = load_training_data(as_numpy=True)
    scorer = make_scorer(mean_squared_log_error)
    results = cross_val_score(model, x, y, scoring=scorer, n_jobs=6, verbose=3)
    print(np.average(results))


def run_model():
    getter = FeatureGetter()
    model = MyModel(getter=getter, core_type='linear-regression', output_transform='none', imputation_method='basic',
                    scaling='none')
    x, y = load_training_data(as_numpy=True,
                              remove_indices=[5, 14, 59, 67, 191, 219, 463, 504, 569, 589, 609,
                                              633, 689, 729, 775, 1325, 1424])
    scorer = make_scorer(mean_squared_log_error)
    results = cross_val_score(model, x, y, scoring=scorer, n_jobs=6, verbose=3)
    print(np.average(results))
    model.fit(x, y)
    y2 = model.predict(x)
    yd = np.abs(y2 - y)
    print(np.average(yd))
    save_model(model, 'none')
    create_submission(model, 'none')


def grid_search_model():
    scorer = make_scorer(mean_squared_log_error)
    x, y = load_training_data(as_numpy=True)
    model = MyModel()
    grid_params = {
        "core_type": ["linear-regression", "elastic-net"],
        "output_transform": ["none", "log"]
    }
    gs = GridSearchCV(model, grid_params, scoring=scorer, n_jobs=6, verbose=4)
    gs.fit(x, y)
    print("best score: ", gs.best_score_)
    print("best parameters: ", gs.best_params_)


def identify_outlier_rows():
    getter = FeatureGetter()
    model = MyModel(getter=getter, core_type='elastic-net', output_transform='none', imputation_method='basic',
                    scaling='none')
    x, y = load_training_data(as_numpy=True, no_shuffle=True)
    model.fit(x, y)
    yp = model.predict(x)
    dev = np.abs(yp - y)
    avg_dev = np.average(dev)
    # add 1 to the index because the pandas dataframe index starts at 1, while numpy indices start at 0
    outliers = np.where(dev > 4.0 * avg_dev)
    outliers = [1 + k for k in outliers]
    print(outliers)


run_model()
