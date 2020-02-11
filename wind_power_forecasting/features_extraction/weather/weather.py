import pandas as pd

from wind_power_forecasting import TIME_LABEL
from wind_power_forecasting.preprocessing.numerical_weather_prediction import format_nwp
from wind_power_forecasting.utils.dataframe import copy_or_not_copy


def add_numerical_weather_prediction_median(df, wp_label, copy=True):
    df = copy_or_not_copy(df, copy)

    wp_number_label = 'wp_number'
    wp_hour_label = 'wp_hour'
    wp_day_offset_label = 'wp_day_offset'
    w_feature_label = 'w_feature'

    nwp_df = format_nwp(df, wp_number_label, wp_hour_label, wp_day_offset_label, w_feature_label, wp_label, copy=True)

    nwp_df = nwp_df. \
        groupby([TIME_LABEL, w_feature_label]). \
        wp_value. \
        median(). \
        reset_index(). \
        set_index(TIME_LABEL). \
        pivot(columns=w_feature_label, values=wp_label)

    return df.join(nwp_df, how='left')
