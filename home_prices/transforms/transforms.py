from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from home_prices.encode.house_style import house_style_array_to_ordinal
from home_prices.encode.zoning import zone_array_to_ordinal
from home_prices.transforms.neighborhood_transform import NeighborhoodTransform
from home_prices.transforms.age import derive_age_array
from home_prices.knn.weighted_knn_regressor import WeightedKNNRegressor


def remove_unused_columns_transform(feature_names: [str] = None) -> ColumnTransformer:
    if feature_names is None:
        feature_names = ["Neighborhood", "MSZoning", "HouseStyle", "GrLivArea", "FullBath", "BedroomAbvGr", "YearBuilt",
                         "YrSold", "OverallQual", "GarageCars", "KitchenQual", "LotArea", "SaleType", "Functional"]
    t0 = ('select_columns', 'passthrough', feature_names)
    # Any columns that are not on the list above will be dropped.
    return ColumnTransformer(transformers=[t0], remainder='drop')


def convert_ordinal_columns_transform() -> ColumnTransformer:
    # There is not really any way to compare neighborhoods, so we just encode them.

    # The problem here is in one pass the ordinal may not find all possible neighbrhoods.
    # Then if a second pass over the data runs (like in k-fold testing), then it might encounter a new
    # category and then crash.
    # ordinal_encoder = OrdinalEncoder()
    # t1 = ('encode_neighborhood', ordinal_encoder, [0])
    neighborhood_encoder = NeighborhoodTransform()
    neighborhood_transform = FunctionTransformer(func=neighborhood_encoder.transform_neighborhood_array_to_ordinal)
    t1 = ('encode_neighborhood', neighborhood_transform, [0])

    # Some zones are more similar than others, so we have a special function to assign numbers to them.
    zone_transform = FunctionTransformer(func=zone_array_to_ordinal)
    t2 = ('transform_zone', zone_transform, [1])

    # Some home styles are more similar than others, so we have a special function to assign numbers to them.
    style_transform = FunctionTransformer(func=house_style_array_to_ordinal)
    t3 = ('transform_style', style_transform, [2])

    return ColumnTransformer(transformers=[t1, t2, t3], remainder='passthrough')


def derive_age_transform() -> ColumnTransformer:
    # The goal here is to produce a new column that has the home age, and replace the two columns that
    # produced it.
    ft = FunctionTransformer(func=derive_age_array)
    # It seems that the columns will be re-ordered unless we explicitly include them in the transforms here.
    t0 = ('retain_columns', 'passthrough', [*range(0, 6)])
    t1 = ('derive_age', ft, [6, 7])
    return ColumnTransformer(transformers=[t0, t1], remainder='passthrough')


def extract_nearest_neighbors_prediction_transform() -> ColumnTransformer:
    nn_estimator = WeightedKNNRegressor()
    transformers = [
        ('retain_columns', 'passthrough', [*range(0, 8)]),
        ('extract_nn_estimates', nn_estimator, [*range(0, 8)])
    ]
    return ColumnTransformer(transformers=transformers, remainder='passthrough')
