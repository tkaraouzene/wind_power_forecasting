from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV

from wind_power_forecasting.model_selection.utils import get_estimator_parameters_dict


def model_autotuning(X, y, estimator=None, n_splits=3, strategy='randomized', **kwargs):
    if estimator is None:
        estimator = RandomForestRegressor(random_state=42)

    params_grid = get_estimator_parameters_dict(estimator)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    if strategy == 'grid':
        clf = GridSearchCV(estimator, params_grid, cv=tscv, **kwargs)

    elif strategy == 'randomized':
        clf = RandomizedSearchCV(estimator, params_grid, cv=tscv, **kwargs)

    else:
        raise ValueError('Unexpected SearchCV strategy: {}'.format(strategy))

    clf.fit(X, y)

    return clf.best_estimator_, clf.best_params_, clf.best_score_
