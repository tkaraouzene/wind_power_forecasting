import pandas as pd
from itertools import product

from wind_power_forecasting.utils.dataframe import copy_or_not_copy


def add_lags(df: pd.DataFrame, lag_range, to_lag_labels=None, to_not_lag_labels=None, copy=True):
    df = copy_or_not_copy(df, copy)

    if to_not_lag_labels is None:
        to_not_lag_labels = []

    if isinstance(to_not_lag_labels, str):
        to_not_lag_labels = [to_not_lag_labels]

    if to_lag_labels is None:
        to_lag_labels = list(df)

    if isinstance(to_lag_labels, str):
        to_lag_labels = [to_lag_labels]

    to_lag_labels = [l for l in to_lag_labels if l not in to_not_lag_labels]

    for lag, label in [*product(lag_range, to_lag_labels)]:
        added_label = label + '_-' + str(lag)
        df[added_label] = df[label].shift(lag)

    return df
