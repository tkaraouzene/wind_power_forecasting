import pandas as pd

from wind_power_forecasting import TIME_LABEL, NWP_PREFIX
from wind_power_forecasting.utils.dataframe import extract_columns


def format_nwp(df, wp_number_label, wp_hour_label, wp_day_offset_label, w_feature_label, wp_label, copy=True):
    if copy:
        df = df.copy()

    # 1. extract NWP columns
    df = extract_columns(df, prefix=NWP_PREFIX)

    # 2. dataframe melting by NWP
    df = melt_nwp(df, wp_label)

    # 3. split by day offset and variable observed (day speed etc...)
    df = split_nwp(df, [wp_number_label, wp_hour_label, wp_day_offset_label, w_feature_label])

    # 4. format day offset as int instead off str
    df = format_day_offset(df, wp_day_offset_label)

    return df


def melt_nwp(df, wp_label):
    return pd.melt(df.reset_index(), id_vars=TIME_LABEL, value_name=wp_label).dropna()


def split_nwp(df, col_labels):
    df[col_labels] = df['variable'].str.split('_', expand=True)
    return df.drop('variable', axis=1)


def format_day_offset(df, day_offset_label):
    df[day_offset_label] = df[day_offset_label].str.replace('D-', '-')
    df[day_offset_label] = df[day_offset_label].str.replace('D', '0')
    df[day_offset_label] = df[day_offset_label].astype(int)

    return df
