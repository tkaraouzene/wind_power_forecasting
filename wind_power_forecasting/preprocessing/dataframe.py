import pandas as pd
from pandas.core.dtypes.common import is_datetime64_dtype

from wind_power_forecasting.utils.dataframe import copy_or_not_copy


def sort_df_index_if_needed(df, copy=True) -> pd.DataFrame:
    if not df.index.is_monotonic_increasing:
        df = copy_or_not_copy(df, copy)
        df.sort_index(inplace=True)

    return df


def convert_df_index_to_datetime_if_needed(df: pd.DataFrame, copy=True) -> pd.DataFrame:
    if not is_datetime64_dtype(df.index):

        try:
            df = copy_or_not_copy(df, copy)
            df.index = pd.to_datetime(df.index)

        except:
            raise ValueError('Cannot convert the index into a datetime')

    return df
