import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


class ColumnMapper(TransformerMixin):

    def __init__(self, column_ref, transform_function):
        self.column_ref = column_ref
        self.transform_function = transform_function

    def fit(self, x, y):
        return self

    def transform(self, x):
        if isinstance(x, np.ndarray):
            t = np.array([self.transform_function(z[self.column_ref]) for z in x])
            # t = x[:, self.column_ref].apply(self.transform_function)
            x[:, self.column_ref] = t
            return x
        elif isinstance(x, pd.DataFrame):
            pass
        elif isinstance(x, pd.Series):
            pass
