import numpy as np

from home_prices.encode import zoning_to_ordinal, house_style_ordinal, map_kitchen_quality, map_functional
from home_prices.knn import WeightedKNNRegressor
from home_prices.util.util import load_training_data
from home_prices.transforms import AgeTransformer, ColumnMapper, NeighborhoodTransform, remove_unused_columns_transform

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import *
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA


def experiment_2():
    features, values = load_training_data()
    feature_names = ["Neighborhood", "MSZoning", "HouseStyle", "GrLivArea", "FullBath", "BedroomAbvGr", "YearBuilt",
                     "YrSold", "OverallQual", "GarageCars", "KitchenQual", "LotArea", "Functional"]
    neighborhood_transform = NeighborhoodTransform()
    estimator = WeightedKNNRegressor()
    linear_learner = LinearRegression()
    pipe = Pipeline(steps=[
        ('remove_unused_columns', remove_unused_columns_transform(feature_names)),
        ("transform_neighborhoods", ColumnMapper(column_ref=0, transform_function=neighborhood_transform.
                                                 transform_neighborhood_to_ordinal)),
        ('transform_zones', ColumnMapper(column_ref=1, transform_function=zoning_to_ordinal)),
        ('transform_styles', ColumnMapper(column_ref=2, transform_function=house_style_ordinal)),
        ('transform_kitchen_quality', ColumnMapper(column_ref=10, transform_function=map_kitchen_quality)),
        ('map_functional', ColumnMapper(column_ref=12, transform_function=map_functional)),
        ('normalize_years', ColumnTransformer(transformers=[
            ('pass', 'passthrough', [*range(0, index_of(feature_names, 'YearBuilt'))]),
            ('normalize_years', FunctionTransformer(func=lambda j: j - 1900.0),
             [index_of(feature_names, 'YearBuilt'), index_of(feature_names, 'YrSold')]
             )
        ], remainder='passthrough')),
        ('derive_age', AgeTransformer(built_column=index_of(feature_names, "YearBuilt"),
                                      sold_column=index_of(feature_names, "YrSold"))),
        ('neighbors_estimate', estimator),
        ('drop_features', ColumnTransformer(transformers=[
            ('retain_useful_features', 'passthrough', [*range(7, 15)])
        ], remainder='drop')),
        ('normalize_area', ColumnTransformer(transformers=[
            ('pass', 'passthrough', [*range(0, 4)]),
            ('normalize_area', StandardScaler(), [4]),
        ], remainder='passthrough')),
        ('svd', PCA(n_components=8)),
        ('estimator', ElasticNet())
    ])
    # t = pipe[:11]
    # t.fit(features, values)
    # out = t.transform(features)
    # print(out)
    # TODO before the last estimator, remove columns that could throw it off, or that were already captured.
    # reorder the columns.
    # You might want to experiment with an Extreme Gradient Boosted Tree.
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    results: np.ndarray = cross_val_score(pipe, features, values, scoring=scorer, verbose=4, n_jobs=6)
    print(-1.0 * np.average(results))




# experiment_2()
