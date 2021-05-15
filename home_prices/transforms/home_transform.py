from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd
from home_prices.transforms import NeighborhoodTransform
from home_prices.encode import *
from home_prices.dataload import load_training_data
from home_prices.dataload import load_feature_names


class HomeTransformer(TransformerMixin):
    def __init__(self, column_names: [str] = None):
        if column_names is None:
            column_names = load_feature_names()
        self.column_names = column_names

    def index_of(self, name) -> int:
        return self.column_names.index(name)

    def get_column(self, x: np.ndarray, name: str) -> np.ndarray:
        return x[:, self.index_of(name)]

    def get_transformed_column(self, x: np.ndarray, name: str, tf) -> np.ndarray:
        y = self.get_column(x, name)
        y = np.array([tf(k) for k in y])
        return y

    def fit(self, x, y):
        return self

    def transform(self, x: np.ndarray):
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        nt = NeighborhoodTransform()
        neighborhoods = self.get_transformed_column(x, 'Neighborhood', nt.transform_neighborhood_to_ordinal)
        neighborhoods = neighborhoods / 25.0
        zoning = self.get_transformed_column(x, 'MSZoning', zoning_to_ordinal)
        zoning[np.isnan(zoning)] = 1.0
        zoning = zoning / 10.0
        style = self.get_transformed_column(x, 'HouseStyle', house_style_ordinal)
        style = style / 2.5
        area = self.get_column(x, 'GrLivArea')
        area_mean = 1515.46
        area_stdev = 525.3
        area_normalized = np.array([((k - area_mean) / area_stdev) for k in area])
        beds = self.get_column(x, 'BedroomAbvGr').astype(float)
        baths = self.get_column(x, 'FullBath').astype(float)
        built = self.get_column(x, 'YearBuilt').astype(float)
        sold = self.get_column(x, 'YrSold').astype(float)
        age = sold - built
        built = np.array([(j - 1910.0) / 100.0 for j in built])
        sold = np.array([(j - 2005.0) / 5.0 for j in sold])
        age = age / 30.0
        cars = self.get_column(x, 'GarageCars')
        cars = cars.astype(float) / 3.0
        kitchen = self.get_transformed_column(x, 'KitchenQual', map_kitchen_quality)
        kitchen = kitchen / 10.0
        overall = self.get_column(x, 'OverallQual')
        overall = overall.astype(float)
        overall = overall / 10.0
        lotarea = self.get_column(x, 'LotArea')
        lotarea_mean = 10516.828
        lotarea_stdev = 9977.846
        lotarea_normalized = np.array([((j - lotarea_mean) / lotarea_stdev) for j in lotarea])
        tmp = np.concatenate([[neighborhoods], [zoning], [style], [area_normalized], [beds], [baths], [built], [sold],
                              [age], [cars], [kitchen], [overall], [lotarea_normalized]],
                             axis=0, dtype=np.float64).T
        return tmp


def produce_prepped_dataset():
    output_names = ["neighborhood", "zone", "style", "area", "beds", "baths", "build", 'sold', 'age', 'cars', 'kitchen',
                    'overall', 'lotarea', 'saleprice']

    features, values = load_training_data(no_shuffle=True)
    indices = features.index
    input_names = features.keys().tolist()
    x = features.to_numpy()
    tf = HomeTransformer(input_names)
    out = tf.transform(x)
    t = np.concatenate([out.T, [values]]).T
    df = pd.DataFrame(t, columns=output_names, index=indices)
    df.round(decimals=3)
    df.to_csv('../data/prepped.csv', header=True)
