from typing import List

import numpy as np
import pandas as pd

from wind_power_forecasting.features_extraction.time.linear_time import compute_minute_of_day, compute_second_of_minute


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
                              label: str = None,
                              label_prefix: str = 'cyclical_',
                              copy=True):
    if time_descriptor is None and index_attribute is None:
        raise ValueError('time_descriptor and index_attribute cannot be both None')

    if time_descriptor is not None and index_attribute is not None:
        raise ValueError('time_descriptor and index_attribute cannot be both defined')

    if index_attribute is not None:
        time_descriptor = getattr(df.index, index_attribute)

    if trig_func_list is None:
        trig_func_list = [np.sin, np.cos]

    if copy:
        df = df.copy()

    if label is None:
        if index_attribute is not None:
            label = label_prefix + index_attribute
        else:
            raise ValueError('label has to be provided when index_attribute is None.')

    for trig_func in trig_func_list:
        trig_func_str = trig_func.__name__
        final_label = label_prefix + label + '_' + trig_func_str
        df[final_label] = cycle_transformation(time_descriptor, max_value, trig_func)

    return df


def add_cyclical_hour_of_day(df, added_label='hour_of_day', copy=False):
    df = add_cycle_time_descriptor(df, max_value=24, index_attribute='hour', label=added_label, copy=copy)

    return df


def add_cyclical_half_hour_of_day(df, added_label='half_hour_of_day', copy=False):
    df = add_cycle_time_descriptor(df, max_value=12, time_descriptor=df.index.hour, label=added_label, copy=copy)

    return df


def add_cyclical_week_of_year(df, added_label='week_of_year', copy=False):
    # TODO manage case when nb weeks == 53
    # from: https://www.timeanddate.com/date/week-numbers.html
    # The weeks of the year are numbered from week 1 to 52 or 53 depending on several factors.
    # Most years have 52 weeks but if the year starts on a Thursday or is a leap year that starts on a Wednesday,
    # that particular year will have 53 numbered weeks.
    # These week numbers are commonly used in some European and Asian countries; but not so much in the United States.
    nb_week = 52
    df = add_cycle_time_descriptor(df, max_value=nb_week, index_attribute='week', label=added_label, copy=copy)

    return df


def add_cyclical_month_of_year(df, added_label='month_of_year', copy=False):
    df = add_cycle_time_descriptor(df, max_value=12, index_attribute='month', label=added_label, copy=copy)

    return df


def add_cyclical_day_of_week(df, added_label='day_of_week', copy=False):
    df = add_cycle_time_descriptor(df, max_value=7, index_attribute='dayofweek', label=added_label, copy=copy)

    return df


def add_cyclical_day_of_month(df, added_label='day_of_month', copy=False):
    df = add_cycle_time_descriptor(df, max_value=df.index.daysinmonth, index_attribute='dayofweek', label=added_label,
                                   copy=copy)

    return df


def add_cyclical_minute_of_hour(df, added_label='minute_of_hour', copy=False):
    df = add_cycle_time_descriptor(df, max_value=60, index_attribute='minute', label=added_label, copy=copy)

    return df


def add_cyclical_minute_of_day(df, added_label='minute_of_day', copy=False):
    min_of_day = compute_minute_of_day(df.index)
    df = add_cycle_time_descriptor(df, max_value=1440, time_descriptor=min_of_day, label=added_label, copy=copy)

    return df


def add_cyclical_second_of_minute(df, added_label='second_of_minute', copy=False):
    sec_min = compute_second_of_minute(df.index)
    df = add_cycle_time_descriptor(df, max_value=60, time_descriptor=sec_min, label=added_label, copy=copy)

    return df


def cycle_transformation(num, denom, trig_func):
    valid_func = [np.sin, np.cos]

    if trig_func not in valid_func:
        raise ValueError(trig_func, valid_func)

    return trig_func(2 * np.pi * num / denom)
