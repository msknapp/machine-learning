from home_prices.util.util import *
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer, mean_squared_log_error
from sklearn.pipeline import Pipeline
from home_prices.transforms.transforms import remove_unused_columns_transform, convert_ordinal_columns_transform, \
    derive_age_transform, extract_nearest_neighbors_prediction_transform
from home_prices.knn.weighted_knn_regressor import WeightedKNNRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from math import fabs as absolute_value
from home_prices.dataload import load_test_data, load_training_data
from sklearn.base import RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def build_nearest_neighbors_pipeline():
    feature_names = ["Neighborhood", "MSZoning", "HouseStyle", "GrLivArea", "FullBath", "BedroomAbvGr", "YearBuilt",
                     "YrSold"]
    estimator = WeightedKNNRegressor()
    pipeline_steps = [
        ('remove_unused_columns', remove_unused_columns_transform(feature_names)),
        ('convert_to_ordinal', convert_ordinal_columns_transform()),
        ('derive_age', derive_age_transform()),
        ('impute_missing', ColumnTransformer(transformers=[
            ('impute_missing', SimpleImputer(strategy='most_frequent'), [*range(0, 8)])
        ])),
        ('estimator', estimator)
    ]
    pipe = Pipeline(steps=pipeline_steps)
    return pipe


def view_nearest_neighbors_output():
    features, values = load_training_data()
    pipe = build_nearest_neighbors_pipeline()
    out = pipe.fit_transform(features, values)
    print(out)


def produce_submission(model: RegressorMixin):
    td = load_test_data()
    out: pd.DataFrame = model.predict(td)
    submission_data = pd.DataFrame([td.index, out]).T
    submission_data.columns = ['Id', 'SalePrice']
    submission_data = submission_data.astype({'Id': int, 'SalePrice': float})
    submission_data['SalePrice'].round(decimals=2)
    submission_data.to_csv('../data/submissions/nearest-neighbors.csv', header=['Id', 'SalePrice'], index=False)


def view_transformed_data():
    features, values = load_training_data()
    pipeline_steps = [
        ('remove_unused_columns', remove_unused_columns_transform()),
        ('convert_to_ordinal', convert_ordinal_columns_transform()),
        ('derive_age', derive_age_transform())
    ]
    pipe = Pipeline(steps=pipeline_steps)
    out = pipe.fit_transform(features)
    print(out)


def view_stack_input():
    features, values = load_training_data()
    pipeline_steps = [
        ('remove_unused_columns', remove_unused_columns_transform()),
        ('convert_to_ordinal', convert_ordinal_columns_transform()),
        ('derive_age', derive_age_transform()),
        ('extract_predictions', extract_nearest_neighbors_prediction_transform())
    ]
    pipe = Pipeline(steps=pipeline_steps)
    res = pipe.fit_transform(features, values)
    print(res)


def evaluate_nearest_neighbors_pipeline():
    scorer = make_scorer(mean_squared_log_error, greater_is_better=False)
    features, values = load_training_data()
    pipe = build_nearest_neighbors_pipeline()
    results: np.ndarray = cross_val_score(pipe, features, values, scoring=scorer, verbose=4)
    print(-1.0 * np.average(results))


def search_nearest_neighbors_parameters():
    features, values = load_training_data()
    scorer = make_scorer(mean_squared_log_error, greater_is_better=False)

    param_grid = {
        "estimator__area_weight": [0.005, 0.01, 0.05],
    }
    pipe = build_nearest_neighbors_pipeline()
    searcher = GridSearchCV(estimator=pipe, scoring=scorer, param_grid=param_grid, verbose=4, n_jobs=6)
    searcher.fit(features, values)
    print("best parameters: ", searcher.best_params_)
    print("best score: ", absolute_value(searcher.best_score_))
    print("done")


def build_boosted_estimator():
    # Columns to add for the linear regressor:
    # OverallQual, GarageCars, KitchenQual, LotArea, SaleType, Functional, LotShape
    linear_learner = LinearRegression()
    pipeline_steps = [
        ('remove_unused_columns', remove_unused_columns_transform()),
        ('convert_to_ordinal', convert_ordinal_columns_transform()),
        ('derive_age', derive_age_transform()),
        ('extract_predictions', extract_nearest_neighbors_prediction_transform()),
        ('estimator', linear_learner)
    ]
    pipe = Pipeline(steps=pipeline_steps)
    return pipe


def evaluate_boosted_estimator():
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    features, values = load_training_data()
    pipe = build_boosted_estimator()
    pipe.fit(features, values)
    t = pipe.predict(features)
    results: np.ndarray = cross_val_score(pipe, features, values, scoring=scorer, verbose=4, )
    print(-1.0 * np.average(results))


pipe = build_nearest_neighbors_pipeline()
features, values = load_training_data()
pipe.fit(features, values)
produce_submission(pipe)

# view_transformed_data()
# evaluate_nearest_neighbors_pipeline()
# evaluate_boosted_estimator()
