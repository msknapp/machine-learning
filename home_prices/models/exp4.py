import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import *
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from home_prices.dataload import load_training_data
from home_prices.transforms import HomeTransformer

from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from home_prices.transforms.assign_ordinal import load_categorical_mapping, CategoricalTransformer
from home_prices.util import create_submission, save_model
from sklearn.neighbors import *
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor



# Strategy:
# pick a few features you want to use, produce a clean dataset.


def build_model_of(core_estimator, no_target_transform: bool = False):
    # tf = HomeTransformer()
    tf0 = load_categorical_mapping()
    tf = CategoricalTransformer(feature_names=None, mapping=tf0)
    si = SimpleImputer(strategy='most_frequent')
    regressor = core_estimator
    if not no_target_transform:
        regressor = TransformedTargetRegressor(regressor=core_estimator, func=np.log, inverse_func=np.exp)
    pipe = Pipeline(steps=[
        ('prep', tf),
        ('remove_zeros', si),
        ('estimator', regressor)
    ])
    return pipe


def model_alpha():
    n = ElasticNet()
    return build_model_of(n)


def train_and_evaluate_model(model):
    x, y = load_training_data(as_numpy=True)
    scorer = make_scorer(mean_squared_log_error)
    results = cross_val_score(model, x, y, scoring=scorer)
    return np.average(results)


def experiment_with_core(core, name):
    m1 = build_model_of(core)
    score = train_and_evaluate_model(m1)
    print("{}: {}".format(name, score))


def grid_search_model():
    scorer = make_scorer(mean_squared_log_error)
    x, y = load_training_data(as_numpy=True)
    reg1 = LinearRegression()
    reg2 = KNeighborsRegressor()
    estimators = [
        ("linear", reg1),
        ("neighbors", reg2)
    ]
    reg3 = StackingRegressor(estimators=estimators, passthrough=True, final_estimator=RidgeCV())
    core = GradientBoostingRegressor(init=reg3)
    model = build_model_of(core)
    results = cross_val_score(model, x, y, scoring=scorer)
    t = np.average(results)
    print(t)
    model.fit(x, y)
    save_model(model, 'stacked-and-boosted')
    create_submission(model, 'stacked-and-boosted')
    # grid_params = {
    #     "estimator__regressor__max_depth": [3],
    #     "estimator__regressor__min_impurity_decrease": [0.0]
    # }
    # gs = GridSearchCV(model, grid_params, scoring=scorer, n_jobs=6, verbose=4)
    # gs.fit(x, y)
    # print("best score: ", gs.best_score_)
    # print("best parameters: ", gs.best_params_)


def experiment_with_many_models():
    experiment_with_core(GradientBoostingRegressor(loss='huber'), "gradient-boosting-regression-huber")
    # experiment_with_core(KNeighborsRegressor(), "kneighbors-regression")
    # experiment_with_core(RadiusNeighborsRegressor(), "radius-neighbors-regression")
    experiment_with_core(LinearRegression(), "linear-regression")
    experiment_with_core(Ridge(), "ridge")
    experiment_with_core(RidgeCV(), "ridge-cv")
    experiment_with_core(SGDRegressor(), "sgd")
    experiment_with_core(ElasticNet(), "elastic-net")
    experiment_with_core(ElasticNetCV(), "elastic-net-cv")
    experiment_with_core(Lars(), "lars")
    experiment_with_core(Lasso(), "lasso")
    experiment_with_core(LassoCV(), "lasso-cv")
    experiment_with_core(ARDRegression(), "ard-regression")
    experiment_with_core(BayesianRidge(), "bayes")
    experiment_with_core(OrthogonalMatchingPursuit(), "orthogonal-matching-pursuit")


def build_train_and_submit(core, name):
    model = build_model_of(core)
    score = train_and_evaluate_model(model)
    print(score)
    x, y = load_training_data(as_numpy=True)
    model.fit(x, y)
    create_submission(model, name)


grid_search_model()
