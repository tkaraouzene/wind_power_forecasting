from typing import Dict

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


def get_estimator_parameters_dict(estimator) -> Dict:
    if isinstance(estimator, LinearRegression):

        params_dict = {}

    elif isinstance(estimator, RandomForestRegressor):

        params_dict = {'bootstrap': [True, False],
                       'max_depth': [10, 20, 40, None],
                       'max_features': ['auto', 'sqrt'],
                       'min_samples_leaf': [2, 4, 6],
                       'min_samples_split': [2, 5, 10],
                       'n_estimators': [10, 20, 50, 100]}
    else:
        raise ValueError('Unexpected estimator: {}'.format(estimator))

    return params_dict


def ts_grid_search(estimator, parms_grid, n_splits=3, **kwargs):
    tscv = TimeSeriesSplit(n_splits =n_splits)

    return GridSearchCV(estimator, parms_grid, cv=tscv, **kwargs)
