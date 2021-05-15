from sklearn.pipeline import Pipeline
from home_prices.dataload import load_training_data, load_features_with_high_missing_values, \
    load_low_variance_feature_names, combine_as_set, remove_as_set, indexes_of
from home_prices.transforms import *
from home_prices.encode import *
from home_prices.util.util import index_of
from sklearn.compose import ColumnTransformer


def experiment3():

    low_variance_features = load_low_variance_feature_names()
    high_missing_value_features = load_features_with_high_missing_values()
    combined = combine_as_set(low_variance_features, high_missing_value_features)

    neighborhood_transform = NeighborhoodTransform()
    features, values = load_training_data()
    names = features.keys().tolist()
    retained_names = remove_as_set(names, combined)
    x = features.to_numpy()
    y = values.to_numpy()
    pipe = Pipeline(steps=[
        ('retain', ColumnTransformer(transformers=[
            ('retain', 'passthrough', indexes_of(names, retained_names))
        ], remainder='drop')),
        ("transform_neighborhoods", ColumnMapper(column_ref=index_of(retained_names, "Neighborhood"),
                                                 transform_function=neighborhood_transform.
                                                 transform_neighborhood_to_ordinal)),
        ('transform_zones', ColumnMapper(column_ref=index_of(retained_names, "MSZoning"), transform_function=zoning_to_ordinal)),
        ('transform_styles',
         ColumnMapper(column_ref=index_of(retained_names, "HouseStyle"), transform_function=house_style_ordinal)),
        ('transform_kitchen_quality',
         ColumnMapper(column_ref=index_of(retained_names, "KitchenQual"), transform_function=map_kitchen_quality)),
        ('map_functional', ColumnMapper(column_ref=index_of(retained_names, "Functional"), transform_function=map_functional)),
        # ('remove_low_variance_features', ColumnTransformer(transformers=[
        #     ('', preprocessing.StandardScaler(), [index_of(names, "KitchenQual")])
        # ]))
    ])
    out = pipe.fit_transform(x, y)
    print(out)


# experiment3()
