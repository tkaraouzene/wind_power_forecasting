import pandas as pd

from wind_power_forecasting.utils.dataframe import copy_or_not_copy


def add_linear_time_descriptor(df: pd.DataFrame,
                               time_descriptor=None,
                               index_attribute: str = None,
                               time_descriptor_label: str = None,
                               copy=True):
    df = copy_or_not_copy(df, copy)

    time_descriptor = compute_time_descriptor(df, time_descriptor, index_attribute)

    if time_descriptor_label is None:
        if index_attribute is not None:
            time_descriptor_label = index_attribute
        else:
            raise ValueError('label has to be provided when index_attribute is None.')

    df[time_descriptor_label] = time_descriptor

    return df


def compute_time_descriptor(df: pd.DataFrame, time_descriptor=None, index_attribute: str = None):

    if time_descriptor is None and index_attribute is None:
        raise ValueError('time_descriptor and index_attribute cannot be both None')

    if time_descriptor is not None and index_attribute is not None:
        raise ValueError('time_descriptor and index_attribute cannot be both defined')

    if index_attribute is not None:
        time_descriptor = getattr(df.index, index_attribute)

    return time_descriptor


def add_day_of_week(df, added_label='day_of_week', copy=False):
    df = add_linear_time_descriptor(df, index_attribute='dayofweek', time_descriptor_label=added_label, copy=copy)

    return df


def compute_minute_of_day(dt_idx: pd.DatetimeIndex):
    return (dt_idx.hour * 60) + dt_idx.minute


def compute_second_of_minute(dt_idx: pd.DatetimeIndex):
    second = dt_idx.second
    micro_second = dt_idx.microsecond / 1000000

    return second + micro_second
