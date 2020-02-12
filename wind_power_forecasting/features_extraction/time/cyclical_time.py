from typing import List

import numpy as np
import pandas as pd

from wind_power_forecasting.features_extraction.time.linear_time import compute_minute_of_day, compute_second_of_minute, \
    compute_time_descriptor
from wind_power_forecasting.utils.dataframe import copy_or_not_copy


class UnexpectedTrigFunc(ValueError):
    """
    This error is raised when a column is found twice in a dataframe
    """

    def __init__(self, given_func, valid_func):
        self.given_func = str(given_func)
        self.valid_func = list(map(str, valid_func))
        super(UnexpectedTrigFunc, self).__init__()

    def __str__(self):
        return 'Unexpected function: {}. Should be one of: {}'.format(self.given_func, self.valid_func)


def add_cyclical_time_feature(df,
                              hour_of_day=False,
                              half_hour_of_day=False,
                              week_of_year=False,
                              month_of_year=False,
                              day_of_week=False,
                              day_of_month=False,
                              min_of_hour=False,
                              min_of_day=False,
                              sec_of_min=False,
                              copy=False):
    if hour_of_day:
        df = add_cyclical_hour_of_day(df, copy=copy)

    if half_hour_of_day:
        df = add_cyclical_half_hour_of_day(df, copy=copy)

    if week_of_year:
        df = add_cyclical_week_of_year(df, copy=copy)

    if month_of_year:
        df = add_cyclical_month_of_year(df, copy=copy)

    if day_of_week:
        df = add_cyclical_day_of_week(df, copy=copy)

    if day_of_month:
        df = add_cyclical_day_of_month(df, copy=copy)

    if min_of_hour:
        df = add_cyclical_minute_of_hour(df, copy=copy)

    if min_of_day:
        df = add_cyclical_minute_of_day(df, copy=copy)

    if sec_of_min:
        df = add_cyclical_second_of_minute(df, copy=copy)

    return df


def add_cycle_time_descriptor(df: pd.DataFrame,
                              max_value: int,
                              time_descriptor=None,
                              index_attribute: str = None,
                              trig_func_list: List[np.ufunc] = None,
                              cyclical_time_descriptor_label: str = None,
                              label_prefix: str = 'cyclical_',
                              copy=True):
    df = copy_or_not_copy(df, copy)

    time_descriptor = compute_time_descriptor(df, time_descriptor, index_attribute)

    if trig_func_list is None:
        trig_func_list = [np.sin, np.cos]

    if cyclical_time_descriptor_label is None:
        if index_attribute is not None:
            cyclical_time_descriptor_label = label_prefix + index_attribute
        else:
            raise ValueError('label has to be provided when index_attribute is None.')

    for trig_func in trig_func_list:
        trig_func_str = trig_func.__name__
        final_label = label_prefix + cyclical_time_descriptor_label + '_' + trig_func_str
        df[final_label] = cycle_transformation(time_descriptor, max_value, trig_func)

    return df


def add_cyclical_hour_of_day(df, added_label='hour_of_day', copy=False):
    df = add_cycle_time_descriptor(df, max_value=24, index_attribute='hour', cyclical_time_descriptor_label=added_label,
                                   copy=copy)

    return df


def add_cyclical_half_hour_of_day(df, added_label='half_hour_of_day', copy=False):
    df = add_cycle_time_descriptor(df, max_value=12, time_descriptor=df.index.hour,
                                   cyclical_time_descriptor_label=added_label, copy=copy)

    return df


def add_cyclical_week_of_year(df, added_label='week_of_year', copy=False):
    # TODO manage case when nb weeks == 53
    # from: https://www.timeanddate.com/date/week-numbers.html
    # The weeks of the year are numbered from week 1 to 52 or 53 depending on several factors.
    # Most years have 52 weeks but if the year starts on a Thursday or is a leap year that starts on a Wednesday,
    # that particular year will have 53 numbered weeks.
    # These week numbers are commonly used in some European and Asian countries; but not so much in the United States.
    nb_week = 52
    df = add_cycle_time_descriptor(df, max_value=nb_week, index_attribute='week',
                                   cyclical_time_descriptor_label=added_label, copy=copy)

    return df


def add_cyclical_month_of_year(df, added_label='month_of_year', copy=False):
    df = add_cycle_time_descriptor(df, max_value=12, index_attribute='month',
                                   cyclical_time_descriptor_label=added_label, copy=copy)

    return df


def add_cyclical_day_of_week(df, added_label='day_of_week', copy=False):
    df = add_cycle_time_descriptor(df, max_value=7, index_attribute='dayofweek',
                                   cyclical_time_descriptor_label=added_label, copy=copy)

    return df


def add_cyclical_day_of_month(df, added_label='day_of_month', copy=False):
    df = add_cycle_time_descriptor(df, max_value=df.index.daysinmonth, index_attribute='dayofweek',
                                   cyclical_time_descriptor_label=added_label,
                                   copy=copy)

    return df


def add_cyclical_minute_of_hour(df, added_label='minute_of_hour', copy=False):
    df = add_cycle_time_descriptor(df, max_value=60, index_attribute='minute',
                                   cyclical_time_descriptor_label=added_label, copy=copy)

    return df


def add_cyclical_minute_of_day(df, added_label='minute_of_day', copy=False):
    min_of_day = compute_minute_of_day(df.index)
    df = add_cycle_time_descriptor(df, max_value=1440, time_descriptor=min_of_day,
                                   cyclical_time_descriptor_label=added_label, copy=copy)

    return df


def add_cyclical_second_of_minute(df, added_label='second_of_minute', copy=False):
    sec_min = compute_second_of_minute(df.index)
    df = add_cycle_time_descriptor(df, max_value=60, time_descriptor=sec_min,
                                   cyclical_time_descriptor_label=added_label, copy=copy)

    return df


def cycle_transformation(num, denom, trig_func):
    check_trigonometric_function(trig_func)

    return trig_func(2 * np.pi * num / denom)


def check_trigonometric_function(trig_func):
    valid_func = [np.sin, np.cos]
    if trig_func not in valid_func:
        raise UnexpectedTrigFunc(trig_func, valid_func)
