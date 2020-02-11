import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import check_array

from wind_power_forecasting import NWP_PREFIX, ID_LABEL
from wind_power_forecasting.features_extraction.time.cyclical_time import add_cyclical_time_feature
from wind_power_forecasting.features_extraction.time.time_shift import add_lags
from wind_power_forecasting.features_extraction.weather.weather import add_numerical_weather_prediction_median
from wind_power_forecasting.features_extraction.weather.wind import add_wind_speed, add_wind_vector_azimuth, \
    add_meteorological_wind_direction
from wind_power_forecasting.features_selection.numerical_weather_prediction import remove_numerical_weather_features
from wind_power_forecasting.features_selection.variance_inflation_factor import remove_collinear_drivers
from wind_power_forecasting.features_selection.variance_threshold import remove_variance_threshold
from wind_power_forecasting.preprocessing.inputs import df_to_ts
from wind_power_forecasting.utils.dataframe import copy_or_not_copy, get_sub_df, df_to_X_y


class WindPowerForecaster(BaseEstimator, RegressorMixin):

    def __init__(self, target_label: str, datetime_label: str):
        self.target_label = target_label
        self.datetime_label = datetime_label

    def fit(self, X_df, y_df):
        """Fit model."""
        X_df = copy_or_not_copy(X_df, True)

        X_df = self._preprocess_data(X_df, copy=False)
        X_df = self._features_extraction(X_df, copy=False)
        X_df = self._features_selection(X_df, wp_prefix=NWP_PREFIX)
        X_df, y_df = self._clean_data(X_df, y_df)
        X, y, X_labels = df_to_X_y(X_df=X_df, y_df=y_df)

        self.X = X
        self.y = y
        self.y_median = np.nanmedian(y)
        self.X_y_idx = X_df.index
        self.X_labels = list(X_df)
        self.y_label = list(y_df)
        self.estimator = RandomForestRegressor(random_state=42).fit(X, y)

        return self

    def add_prediction(self, X_df, prediction_label='prediction', preprocess=True, na_rm=False):

        pred_df = self.predict(X_df, preprocess=preprocess, output_type='dataframe', prediction_label=prediction_label)
        how = 'right' if na_rm else 'left'
        return X_df.join(pred_df, how=how)

    def predict(self, X_df, preprocess=True, output_type='array', prediction_label='prediction'):
        """Apply the model
        """
        X_df = copy_or_not_copy(X_df)

        if preprocess:
            X_df = self._preprocess_data(X_df, copy=False)
            X_df = self._features_extraction(X_df, copy=False)
            X_df = self._features_selection(X_df, from_fit=False)
            X_df, _ = self._clean_data(X_df, from_fit=False)

        X = check_array(X_df, accept_sparse='csr')
        y_pred = self.estimator.predict(X)
        if output_type == 'array':
            out = y_pred

        elif output_type == 'dataframe':
            out = pd.DataFrame(index=X_df.index, data=y_pred, columns=[prediction_label])

        return out

    def _preprocess_data(self, X_df, copy=True):
        # 1. Convert dataframe into time series
        X_df = df_to_ts(X_df, self.datetime_label, freq='H', copy=copy)

        return X_df

    def _features_extraction(self, X_df, copy=True):
        X_df = copy_or_not_copy(X_df, copy)

        # 1. Cyclical time encoding
        X_df = add_cyclical_time_feature(X_df, hour_of_day=True, half_hour_of_day=True, week_of_year=True,
                                         month_of_year=True, day_of_week=True, copy=False)

        # 2. Meteorological features
        wp_label = 'wp_value'
        X_df = add_numerical_weather_prediction_median(X_df, wp_label=wp_label, copy=False)
        X_df = add_wind_speed(X_df, 'U', 'V', 'wind_speed', copy=False)
        X_df = add_wind_vector_azimuth(X_df, 'U', 'V', copy=False)
        X_df = add_meteorological_wind_direction(X_df, 'U', 'V', copy=False)

        # 3. Meteorological features lags
        X_df = add_lags(X_df, lag_range=[1, 2], to_not_lag_labels=ID_LABEL, copy=False)

        return X_df

    def _features_selection(self, X_df, wp_prefix=None, from_fit=True):

        if from_fit:
            X_df = remove_numerical_weather_features(X_df, wp_prefix)
            X_df = remove_variance_threshold(X_df, force_keeping=ID_LABEL, threshold=0.8)
            X_df = remove_collinear_drivers(X_df, force_keeping=ID_LABEL, threshold=5)

        else:
            X_df = X_df.loc[:, self.X_labels + [ID_LABEL]]

        return X_df

    def _clean_data(self, X_df, y_df=None, from_fit=True, copy=True):

        X_df = copy_or_not_copy(X_df, copy)

        if from_fit:
            X_df = X_df.dropna()
        else:
            X_df = X_df.fillna(method='ffill')
            X_df = X_df.fillna(self.y_median)

        if y_df is not None:
            y_df = copy_or_not_copy(y_df, copy)
            y_df = get_sub_df(y_df, ID_LABEL, X_df[ID_LABEL])

        if from_fit:
            X_df = X_df.drop(ID_LABEL, axis=1)
        else:
            X_df = X_df.set_index(ID_LABEL)
            X_df.index = X_df.index.astype(int)

        return X_df, y_df
