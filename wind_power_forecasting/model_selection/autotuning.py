from wind_power_forecasting.model_selection.utils import get_estimator_parameters_dict, ts_grid_search


def model_autotuning(X, y, estimator):
    parms_grid = get_estimator_parameters_dict(estimator)
    clf = ts_grid_search(estimator, parms_grid)
    clf.fit(X, y)

    return clf.best_estimator_, clf.best_params_, clf.best_score_
