from itertools import product

import pandas as pd

from wind_power_forecasting.utils.dataframe import copy_or_not_copy


def add_lags(df: pd.DataFrame, lag_range, to_lag_labels, copy=True):
    df = copy_or_not_copy(df, copy)


    if isinstance(to_lag_labels, str):
        to_lag_labels = [to_lag_labels]

    for lag, label in [*product(lag_range, to_lag_labels)]:
        added_label = label + '_-' + str(lag)
        df[added_label] = df[label].shift(lag)

    return df
