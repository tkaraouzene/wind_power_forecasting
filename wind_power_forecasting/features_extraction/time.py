import pandas as pd

def compute_minute_of_day(dt_idx: pd.DatetimeIndex):
    return (dt_idx.hour * 60) + dt_idx.minute


def compute_second_of_minute(dt_idx: pd.DatetimeIndex):
    second = dt_idx.second
    micro_second = dt_idx.microsecond / 1000000

    return second + micro_second
