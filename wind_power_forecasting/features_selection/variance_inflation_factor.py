import numpy as np
import pandas as pd
import statsmodels.stats.outliers_influence as oi
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from statsmodels.tools import add_constant

from wind_power_forecasting.utils import get_X_y_df, split_features_to_process_df


class VarianceInflationFactorThreshold(BaseEstimator, SelectorMixin):
    """
    Feature selector that removes all low variance-inflation-factor (vif) features.

    This feature selection algorithm looks only at the features (X), not the
    desired outputs (y), and can thus be used for unsupervised learning.

    Parameters
    ----------
    threshold : float, optional
        Features with a training-set variance lower than this threshold will
        be removed. The default is to keep all features with non-zero vif,
        i.e. remove the features that have the same value in all samples.

    Attributes
    ----------
    is_removed:
    kept_labels: list or None
        In case where input data (`X`) where a pandas DataFrame object, store the conserved columns labels.

    Examples
    --------
    The following data set has integer features, the second one is a multiple of the first one.
    It is removed with the default setting for threshold:

        >>> X = [[1, 2, 0, 3], [2, 4, 7, 3], [3, 6, 1, 2]]
        >>> selector = VarianceInflationFactorThreshold()
        >>> # Check for every features if they are kept or not.
        >>> selector.fit(X).get_support()
        array([False,  True,  True,  True], dtype=bool)
        >>> # Remove filtered features from data set.
        >>> selector.fit_transform(X)
        array([[2, 0, 3],
               [4, 7, 3],
               [6, 1, 2]])
    """

    def __init__(self, threshold=0.):
        self.threshold = threshold

    def fit(self, X, y=None):
        """
        Apply the variance inflation threshold

        Parameters
        ----------
        {X}
        y : any
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        self: returns an instance of self.
        """

        if isinstance(X, pd.DataFrame):
            labels = list(X)
        else:
            labels = None

        X = check_array(X, ('csr', 'csc'), dtype=np.float64)
        is_removed = np.repeat(False, X.shape[1])

        for _ in range(X.shape[1] - 1):

            vif = compute_variance_inflation_factor(X)
            # Retrieve the column index of the column presenting the highest collinearity value
            max_vif_idx = vif.argmax()

            # If the highest collinearity value is smaller than threshold
            # no need to continue, exist of the loop
            if vif[max_vif_idx] <= self.threshold:
                continue
            else:
                is_removed[max_vif_idx] = True
                X = np.delete(X, max_vif_idx, axis=1)

                if labels is not None:
                    labels.remove(labels[max_vif_idx])

        self.is_removed = is_removed
        self.kept_labels = labels

        return self

    def _get_support_mask(self):
        check_is_fitted(self, ['is_removed'])

        return np.array([not self.is_removed[idx] for idx in range(self.is_removed.size)])


def compute_variance_inflation_factor(X: np.array) -> np.array:
    """
    Compute the variance inflation factor for each features of a matrix

    Parameters
    ----------
    {X}

    Returns
    -------
    vif: np.array of shape = (n_features)
        variance inflation factor of each features of the input matrix
    """

    # Add a constant columns as suggest here:
    # https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python
    X = add_constant(X, prepend=True)
    vif = np.array([oi.variance_inflation_factor(X, j) for j in range(X.shape[1])])
    # remove the firs element corresponding to the constant col
    vif = np.delete(vif, 0)
    vif[np.isnan(vif)] = np.inf

    return vif


def remove_collinear_drivers(df: pd.DataFrame, target_label=None, threshold=0., force_keeping=None):
    """
    This module remove all columns of a dataframe presenting high collinearity.
    Collinearity is estimated using the variance inflation factor (VIF):
    https://en.wikipedia.org/wiki/Variance_inflation_factor

    VIF is compute iteratively for each feature. The one with the highest VIF is removed if
    this value is higher than the threshold parameter.
    This process is repeated until the highest VIF value is lower than threshold parameter.

    It is possible to force the keeping of some features. VIF wont be calculated on them and they will be automatically
    returned.

    The dataframe devoid of columns presenting high collinearity is returned.

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
    {df} Same as the input from which features with a low variance inflation factor where removed.
    """

    df, kept_features_df = split_features_to_process_df(df, force_keeping)
    n_kept = kept_features_df.shape[1] if kept_features_df is not None else 0

    X_df, y_df = get_X_y_df(df, target_label, rm_na=True)

    if X_df.shape[1] > 0 and X_df.shape[1] + n_kept >= 1:
        is_selected = VarianceInflationFactorThreshold(threshold).fit(X_df).get_support()
        X_df = X_df.loc[:, is_selected]

    return pd.concat([y_df, X_df, kept_features_df], axis=1)
