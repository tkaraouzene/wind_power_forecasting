import numpy as np
from sklearn.utils import check_consistent_length, check_array

from wind_power_forecasting.utils.dataframe import copy_or_not_copy


def add_wind_speed(df, u_label, v_label, wind_speed_label='wind_speed', copy=True):
    df = copy_or_not_copy(df, copy)

    df[wind_speed_label] = compute_wind_speed(df[u_label], df[v_label])

    return df


def add_wind_vector_azimuth(df, u_label, v_label, wind_speed_label='wind_vector_azimuth', copy=True):
    df = copy_or_not_copy(df, copy)

    df[wind_speed_label] = compute_wind_vector_azimuth(df[u_label], df[v_label])

    return df


def add_meteorological_wind_direction(df, u_label, v_label,
                                      meteorological_wind_direction_label='meteorological_wind_direction', copy=True):
    df = copy_or_not_copy(df, copy)

    df[meteorological_wind_direction_label] = compute_meteorological_wind_direction(df[u_label], df[v_label])

    return df


def compute_wind_speed(u, v):
    u = check_array(u, ensure_2d=False, force_all_finite=False)
    v = check_array(v, ensure_2d=False, force_all_finite=False)
    check_consistent_length(u, v)

    return np.sqrt(np.square(u) + np.square(v))


def compute_wind_vector_azimuth(u, v):
    u = check_array(u, ensure_2d=False, force_all_finite=False)
    v = check_array(v, ensure_2d=False, force_all_finite=False)
    check_consistent_length(u, v)
    return np.degrees(np.arctan2(u, v))


def compute_meteorological_wind_direction(u, v):
    u = check_array(u, ensure_2d=False, force_all_finite=False)
    v = check_array(v, ensure_2d=False, force_all_finite=False)
    check_consistent_length(u, v)
    return np.degrees(np.arctan2(-u, -v))


def wind_vector_azimuth_to_meteorological_wind_direction(wind_vector_azimuth):
    return wind_vector_azimuth + 180


def meteorological_wind_direction_to_wind_vector_azimuth(meteorological_wind_direction):
    return meteorological_wind_direction - 180
