import numpy as np
import pandas as pd
from sklearn.metrics.regression import _check_reg_targets
from sklearn.utils import check_consistent_length


def theils_u1_error(y_true, y_hat, sample_weight=None, multioutput='uniform_average'):
    """
    Ref: http://www.forecastingprinciples.com/data/definitions/theil's%20u.html
    Theil, H. (1958), Economic Forecasts and Policy. Amsterdam: North Holland.

    This formula is implemented but not used because considered ambiguous by Bliemel.

    :param y_true: true value
    :param y_hat: predicted value
    :param sample_weight:
    :param multioutput:

    :return:
    """

    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    # return mean_absolute_error(y_true, y_hat)
    if len(y_true) != len(y_hat):
        raise ValueError('The real values and the prediction must have the same lenght,'
                         ' {} != {}'.format(len(y_true), len(y_hat)))

    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_hat, multioutput)
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


def theils_u2_error(y_true, y_hat, sample_weight=None, multioutput='uniform_average'):
    """
    Ref: http://www.forecastingprinciples.com/data/definitions/theil's%20u.html

    Thiel, H. (1966), Applied Economic Forecasting. Chicago: Rand McNally.

    This formula is implemented because considered non-ambiguous by Bliemel.

    :param y_true: true value
    :param y_hat: predicted value
    :param sample_weight:
    :param multioutput:

    :return:
    """

    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    # return mean_absolute_error(y_true, y_hat)
    if len(y_true) != len(y_hat):
        raise ValueError('The real values and the prediction must have the same lenght,'
                         ' {} != {}'.format(len(y_true), len(y_hat)))

    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_hat, multioutput)
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
