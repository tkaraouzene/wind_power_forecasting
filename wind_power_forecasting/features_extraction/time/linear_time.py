import numpy as np
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


def compute_is_weekend(dt_idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(dt_idx.weekday >= 5)


def compute_nb_days_in_year(dt_idx: pd.DatetimeIndex):
    return 365 + dt_idx.is_leap_year


def compute_is_afternoon(dt_idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(map(bool, np.floor(dt_idx.hour / 12)))


def compute_nb_weeks_in_year(dt_idx: pd.DatetimeIndex):
    """
    from: https://www.timeanddate.com/date/week-numbers.html
    The weeks of the year are numbered from week 1 to 52 or 53 depending on several factors.
    Most years have 52 weeks but if the year starts on a Thursday or is a leap year that starts on a Wednesday, that particular year will have 53 numbered weeks. These week numbers are commonly used in some European and Asian countries; but not so much in the United States.


    Parameters
    ----------
    dt_idx

    Returns
    -------

    """

    major_week = 0
    # TODO(TK): implement the condition
    # if ...:
    #     major_week = 1
    #

    return 52 + major_week
