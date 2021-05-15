import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin


def derive_age_array(x):
    if isinstance(x, pd.DataFrame):
        t = x[:, 1] - x[:, 0]
        return t
    elif isinstance(x, np.ndarray):
        # The first column is the year built
        # the second column is the year sold
        year_built = x[:, 0]
        year_sold = x[:, 1]
        age_at_sale: np.ndarray = year_sold - year_built
        s: np.ndarray = np.concatenate([[age_at_sale], [year_sold]])
        s = s.transpose()
        return s


class AgeTransformer(TransformerMixin):

    def __init__(self, built_column, sold_column):
        self.built_column = built_column
        self.sold_column = sold_column

    def fit(self, x, y):
        return self

    def transform(self, x):
        if isinstance(x, np.ndarray):
            age_at_sale = x[:, self.sold_column] - x[:, self.built_column]
            out = np.concatenate([x.T, [age_at_sale]]).T
            return out
        elif isinstance(x, pd.DataFrame):
            pass
        elif isinstance(x, pd.Series):
            pass
