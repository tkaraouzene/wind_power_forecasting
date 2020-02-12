from itertools import product

import pandas as pd

from wind_power_forecasting.utils.dataframe import copy_or_not_copy


def add_lags(df: pd.DataFrame, lag_range, to_lag_labels=None, to_not_lag_labels=None, copy=True):
    df = copy_or_not_copy(df, copy)

    to_lag_labels = _get_labels_to_process(df, to_lag_labels, to_not_lag_labels)

    for lag, label in [*product(lag_range, to_lag_labels)]:
        added_label = label + '_-' + str(lag)
        df[added_label] = df[label].shift(lag)

    return df


def add_rollmeans(df: pd.DataFrame, periods, to_roll_labels=None, to_not_roll_labels=None, copy=True):
    df = copy_or_not_copy(df, copy)

    to_roll_labels = _get_labels_to_process(df, to_roll_labels, to_not_roll_labels)

    for period, label in [*product(periods, to_roll_labels)]:
        added_label = label + '_rollmean_' + str(period)
        df[added_label] = df[label].rolling(period).mean()


def _get_labels_to_process(df, to_process_labels=None, to_not_process_labels=None):
    if to_not_process_labels is None:
        to_not_process_labels = []

    if isinstance(to_not_process_labels, str):
        to_not_process_labels = [to_not_process_labels]

    if to_process_labels is None:
        to_process_labels = list(df)

    if isinstance(to_process_labels, str):
        to_process_labels = [to_process_labels]

    to_process_labels = [l for l in to_process_labels if l not in to_not_process_labels]

    return to_process_labels
