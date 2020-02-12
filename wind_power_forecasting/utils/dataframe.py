import numpy as np
import pandas as pd
from nose.tools import assert_is_none, assert_is_not_none, assert_equal
from sklearn.utils import check_consistent_length, check_array


def get_sub_df(df: pd.DataFrame, column_label: str, kept_values, keep_column: bool = False,
               reset_index: bool = False) -> pd.DataFrame:
    """
    Subset a dataframe selecting values within a specific column.

    Parameters
    ----------
    {df}
    column_label: str
    kept_values: array-like or str
        Values to keep
    keep_column: bool, optional
        If `False` (**default**) `column_label` is dropped from the output dataframe.
    reset_index: bool, optional
        If `False` (**default**) the output dataframe index is NOT reset.

    Returns
    -------
    sub_df: pandas DataFrame object
        Subset of the input dataframe `df`.

    """

    if isinstance(kept_values, str):
        # if not isinstance(kept_values, Iterable) or isinstance(kept_values, str):
        kept_values = [kept_values]

    sub_df = df.loc[df[column_label].isin(kept_values)].copy()

    if not keep_column:
        sub_df.drop(column_label, inplace=True, axis=1)

    if reset_index:
        sub_df.reset_index(drop=True, inplace=True)

    return sub_df


def get_X_y_df(df, target_label=None, rm_na=False):
    """
    Split a single dataframe in two. One containing features and one containing the target

    Parameters
    ----------
    {df}
    {target_label}
    {rm_na}

    Returns
    -------
    {X_df}
    {y_df}
    """

    if rm_na:
        df = df.dropna()

    if target_label is None:
        X_df = df
        y_df = None

    else:
        X_df, y_df, _ = df_to_X_y(df, target_label=target_label, output_type='dataframe')

    return X_df, y_df


def df_to_X_y(X_y_df: pd.DataFrame = None, X_df: pd.DataFrame = None, y_df: pd.DataFrame = None, target_label=None,
              output_type='array'):
    """
    Split features and target from a dataframe containing both of them and return it.

    Parameters
    ----------
    X_y_df: pandas DataFrame object, optional
        Dataframe containing both target and features.
        Each column of this dataframe represents a variable (feature or target).
        Each row represents an observation. If not specified, neither  `X` or `y`can be `None`.
    {X_opt}
    {y_opt}
    {target_label} Mandatory if `X_y_df` is not `None`
    output_type: str
        Type of the output.
        If `array`, the output will be a numpy array, if `dataframe` the output will be a pandas DataFrame object.

    Returns
    -------
    {X}
    {y}
    X_labels: list or None
        If the input was a dataframe, return the features as a list

    Raises
    ------
    ValueError
        If `X_y_df` is not `None` and either `X` or `y` is not `None`.
        If `X_y_df` is not `None` and target_label is `None`.
        If `X_y_df` is `None` and either `X` or `y` is `None`.
        If `output_type` is not one of: `array` or `dataframe`.
    DataFrameNbColError
        If `y` is not in 1D.
    EmptyDataFrameError
        If `y` is empty.

    Examples
    --------
    >>>import pandas as pd
    >>>X_y_df = pd.DataFrame({{'target':[1,2,3,4], 'feature1': [5,6,7,8], 'feature2': [9,10,11,12]}})
    >>>X, y, X_labels = df_to_X_y(X_y_df, target_label='target')
    >>>X_labels
    >>>X
    >>>y

    output as dataframe

    >>>X_y_df = pd.DataFrame({{'target':[1,2,3,4], 'feature1': [5,6,7,8], 'feature2': [9,10,11,12]}})
    >>>X, y, X_labels = df_to_X_y(X_y_df, target_label='target', output_type='dataframe')
    >>>X_labels
    >>>X
    >>>y
    """

    if X_y_df is not None:
        assert_is_none(X_df, 'X_df')
        assert_is_none(y_df, 'y_df')
        assert_is_not_none(target_label, 'target_label')

        X_df = X_y_df.drop(target_label, axis=1)
        y_df = X_y_df[[target_label]]

        assert_equal(y_df.shape[1], 1)

    assert_is_not_none(X_df, 'X_df')
    assert_is_not_none(y_df, 'y_df')
    check_consistent_length(X_df, y_df)

    X_labels = list(X_df)

    if output_type == 'array':
        X_out = check_array(X_df, accept_sparse='csr')
        y_out = np.ravel(check_array(y_df, ensure_2d=False))

    elif output_type == 'dataframe':
        X_out = X_df
        y_out = y_df

    else:
        raise ValueError('Unexpected output_type: should be one of \'array\' or \'dataframe\'.')

    return X_out, y_out, X_labels


def split_features_to_process_df(df, force_keeping=None):
    """

    Parameters
    ----------
    {df}
    force_keeping: str or list

    Returns
    -------
    {df}
    kept_features_df: pandas DataFrame object or None
    """
    if force_keeping is not None:

        # To ensure that kept_features_df will be a df and not a series
        if isinstance(force_keeping, str):
            force_keeping = [force_keeping]

        kept_features_df = df.loc[:, force_keeping]
        df = df.drop(force_keeping, axis=1)
    else:
        kept_features_df = None

    return df, kept_features_df


def extract_columns(df: pd.DataFrame, prefix: str = None, suffix: str = None, contained: str = None):
    subset_columns = list(df)

    if prefix is not None:
        subset_columns = [c for c in subset_columns if c.startswith(prefix)]

    if suffix is not None:
        subset_columns = [c for c in subset_columns if c.endswith(suffix)]

    if contained is not None:
        subset_columns = [c for c in subset_columns if contained in c]

    return df.loc[:, subset_columns]


def copy_or_not_copy(df: pd.DataFrame, copy=True) -> pd.DataFrame:
    if copy:
        df = df.copy()

    return df
