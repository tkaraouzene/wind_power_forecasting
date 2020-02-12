import pandas as pd

from wind_power_forecasting.preprocessing.dataframe import sort_df_index_if_needed, \
    convert_df_index_to_datetime_if_needed
from wind_power_forecasting.utils.dataframe import copy_or_not_copy


def input_preprocessing(X_df: pd.DataFrame, datetime_label) -> pd.DataFrame:

    # Convert dataframe into time series:
    #   1. Set datetime column as index
    #   2. Convert it into datetime
    #   3. Sort it
    #   4. Force the frequency (fill with missing values).
    X_df = df_to_ts(X_df, datetime_label, freq='H')

    return X_df


def df_to_ts(df: pd.DataFrame, datetime_label: str, freq, copy=False) -> pd.DataFrame:
    # Copy the dataframe to avoid border effects (if needed).
    df = copy_or_not_copy(df, copy)

    # Set the datetime column as dataframe index and check they are no duplicated.
    df.set_index(datetime_label, inplace=True, verify_integrity=True)

    # Convert the index into a datetime.
    convert_df_index_to_datetime_if_needed(df, copy=False)

    # Sort the index
    sort_df_index_if_needed(df, copy=False)

    # Set the dataframe frequency.
    df = df.asfreq(freq)

    return df


def remove_na(df, copy=False, **kwargs):
    # Protect against automatic frequency change from pandas !!!
    # when the new dataset with dropped rows has another sampling period, pandas automatically changes it.
    sampling_freq = df.index.freq
    df.dropna(**kwargs)
    df.index.freq = sampling_freq

    return df
