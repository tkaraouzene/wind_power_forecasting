import numpy as np
import pandas as pd
from sklearn.metrics.regression import _check_reg_targets
from sklearn.utils import check_consistent_length


def cumulated_absolute_percentage_error(y_true, y_pred):
    """
    CAPE (Cumulated Absolute Percentage Error)
    Function used by CNR for the evaluation of predictions

    Parameters
    ----------
    y_true: array-like of shape = (n_samples)
        Ground truth (correct) target values.
    y_pred: array-like of shape = (n_samples)
        Estimated target values.

    Returns
    -------
    cape: float
        The metric evaluated with the two dataframes. This must not be NaN.

    References
    ----------
    """

    return 100 * np.sum(np.abs(y_pred - y_true)) / np.sum(y_true)


def theils_u1_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
    """
    Ref: http://www.forecastingprinciples.com/data/definitions/theil's%20u.html
    Theil, H. (1958), Economic Forecasts and Policy. Amsterdam: North Holland.

    This formula is implemented but not used because considered ambiguous by Bliemel.

    :param y_true: true value
    :param y_pred: predicted value
    :param sample_weight:
    :param multioutput:

    :return:
    """

    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    # return mean_absolute_error(y_true, y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError('The real values and the prediction must have the same lenght,'
                         ' {} != {}'.format(len(y_true), len(y_pred)))

    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)

    # Root Mean Squared Error
    rmse = np.sqrt(np.average((y_true - y_pred) ** 2, weights=sample_weight, axis=0))

    b = np.sqrt(np.average(y_pred ** 2, weights=sample_weight, axis=0))
    c = np.sqrt(np.average(y_true ** 2, weights=sample_weight, axis=0))

    output_error = rmse / (b + c)

    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            return output_error
        elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
            multioutput = None
    return np.average(output_error, weights=multioutput)


def theils_u2_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
    """
    Ref: http://www.forecastingprinciples.com/data/definitions/theil's%20u.html

    Thiel, H. (1966), Applied Economic Forecasting. Chicago: Rand McNally.

    This formula is implemented because considered non-ambiguous by Bliemel.

    :param y_true: true value
    :param y_pred: predicted value
    :param sample_weight:
    :param multioutput:

    :return:
    """

    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    # return mean_absolute_error(y_true, y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError('The real values and the prediction must have the same lenght,'
                         ' {} != {}'.format(len(y_true), len(y_pred)))

    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)

    # Root Squared Error
    rse = np.sqrt(((y_true - y_pred) ** 2).sum())
    c = np.sqrt((y_true ** 2).sum())

    output_error = rse / c

    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            return output_error
        elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
            multioutput = None
    return np.average(output_error, weights=multioutput)
