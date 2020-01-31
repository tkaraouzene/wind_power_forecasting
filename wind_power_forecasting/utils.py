import pandas as pd


def get_sub_df(df: pd.DataFrame, column_label: str, kept_values, keep_column: bool = False,
               reset_index: bool = False) -> pd.DataFrame:
    if isinstance(kept_values, str):
        # if not isinstance(kept_values, Iterable) or isinstance(kept_values, str):
        kept_values = [kept_values]

    sub_df = df.loc[df[column_label].isin(kept_values)].copy()

    if not keep_column:
        sub_df.drop(column_label, inplace=True, axis=1)

    if reset_index:
        sub_df.reset_index(drop=True, inplace=True)

    return sub_df