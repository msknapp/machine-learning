from sklearn.base import BaseEstimator, RegressorMixin
from home_prices.knn.weighted_distance import DistanceCalculator
from sklearn.neighbors import KNeighborsRegressor
import numpy as np


class WeightedKNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, neighborhood_weight=100., zone_weight=10., style_weight=5., area_weight=.01, bath_weight=1.,
                 bedroom_weight=0.5, age_weight=0.1, year_weight=1.0):
        self.neighborhood_weight = neighborhood_weight
        self.zone_weight = zone_weight
        self.style_weight = style_weight
        self.area_weight = area_weight
        self.bath_weight = bath_weight
        self.bedroom_weight = bedroom_weight
        self.age_weight = age_weight
        self.year_weight = year_weight
        self.weights = [self.neighborhood_weight, self.zone_weight, self.style_weight, self.area_weight,
                        self.bath_weight, self.bedroom_weight, self.age_weight, self.year_weight]
        self.distance_calculator = DistanceCalculator(self.weights)
        self.estimator = KNeighborsRegressor(weights='distance', metric=self.distance_calculator.home_distance)

    def set_params(self, **params):
        for p in params.keys():
            if p == "neighborhood_weight":
                self.neighborhood_weight = params[p]
            if p == "zone_weight":
                self.zone_weight = params[p]
            if p == "style_weight":
                self.style_weight = params[p]
            if p == "area_weight":
                self.area_weight = params[p]
            if p == "bath_weight":
                self.bath_weight = params[p]
            if p == "bedroom_weight":
                self.bedroom_weight = params[p]
        self.reset_weights()

    def reset_weights(self):
        self.weights = [self.neighborhood_weight, self.zone_weight, self.style_weight, self.area_weight,
                        self.bath_weight, self.bedroom_weight, self.age_weight, self.year_weight]
        self.distance_calculator.weights = self.weights

    def fit(self, x, y):
        self.estimator.fit(x, y)
        return self

    def predict(self, x):
        return self.estimator.predict(x)

    def transform(self, x):
        predictions = self.estimator.predict(x)
        t = np.concatenate([x.T, [predictions]]).T
        return t
