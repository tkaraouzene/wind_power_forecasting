import pandas as pd

def add_weekday(df:pd.DataFrame, week_day_label:str, copy=True):

    # Copy dataframe if needed
    if copy:
        df = df.copy()

    if hasattr(df.index, 'weekday'):
        df[week_day_label] = df.index.weekday

    else:
        raise AttributeError('DataFrame index object has no  \'weekday\' attribute')

    return df