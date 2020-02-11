import numpy as np
from sklearn.utils import check_consistent_length, check_array

from wind_power_forecasting.utils.dataframe import copy_or_not_copy


def add_wind_speed(df, meridional_velocity_label, zonal_velocity_label, wind_speed_label='wind_speed', copy=True):
    df = copy_or_not_copy(df, copy)

    df[wind_speed_label] = compute_wind_speed(df[meridional_velocity_label], df[zonal_velocity_label])

    return df


def compute_wind_speed(meridional_velocity, zonal_velocity):
    meridional_velocity = check_array(meridional_velocity, ensure_2d=False, force_all_finite=False)
    zonal_velocity = check_array(zonal_velocity, ensure_2d=False, force_all_finite=False)
    check_consistent_length(meridional_velocity, zonal_velocity)

    return np.sqrt(np.square(meridional_velocity) + np.square(zonal_velocity))
