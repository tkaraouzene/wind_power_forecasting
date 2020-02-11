import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from wind_power_forecasting.utils import split_features_to_process_df, get_X_y_df


def remove_variance_threshold(df: pd.DataFrame, target_label=None, threshold=0., force_keeping=None):
    """
    This module remove all columns of a dataframe presenting high low variance.

    Parameters
    ----------
    {df}
    {target_label_opt}
    threshold : float, optional
            Features with a training-set variance lower than this threshold will
            be removed. The default is to keep all features with non-zero variance,
            i.e. remove the features that have the same value in all samples.
    force_keeping: array-like, optional
        List of features which will be kept whatever their variance

    Returns
    -------
    {df} Same as the input from which features with a low variance where removed.
    """

    # TODO add a copy parameter
    df, kept_features_df = split_features_to_process_df(df, force_keeping)

    n_kept = kept_features_df.shape[1] if kept_features_df is not None else 0
    X_df, y_df = get_X_y_df(df, target_label, rm_na=True)

    if X_df.shape[1] > 0 and X_df.shape[1] + n_kept >= 1:
        # Var[X] = p * (1 - p)
        threshold = threshold * (1 - threshold)

        is_selected = VarianceThreshold(threshold).fit(X_df).get_support()
        X_df = X_df.loc[:, is_selected]

    return pd.concat([y_df, X_df, kept_features_df], axis=1)
