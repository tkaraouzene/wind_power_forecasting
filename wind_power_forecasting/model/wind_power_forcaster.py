from sklearn.base import BaseEstimator, RegressorMixin

from wind_power_forecasting import NWP_PREFIX, ID_LABEL
from wind_power_forecasting.features_extraction.time.cyclical_time import add_cyclical_time_feature
from wind_power_forecasting.features_extraction.time.time_shift import add_lags
from wind_power_forecasting.features_extraction.weather.weather import add_numerical_weather_prediction_median
from wind_power_forecasting.features_extraction.weather.wind import add_wind_speed
from wind_power_forecasting.features_selection.numerical_weather_prediction import remove_numerical_weather_features
from wind_power_forecasting.features_selection.variance_inflation_factor import remove_collinear_drivers
from wind_power_forecasting.preprocessing.inputs import df_to_ts


class WindPowerForecaster(BaseEstimator, RegressorMixin):

    def __init__(self, target_label: str, datetime_label: str):
        self.target_label = target_label
        self.datetime_label = datetime_label

    def fit(self, X_df, y_df):
        """Fit model."""

        X_df = self._preprocess_data(X_df, copy=False)
        X_df = self._features_extraction(X_df, copy=False)
        X_df = self._features_selection(X_df, wp_prefix=NWP_PREFIX)

        self.X_df = X_df
        self.y_df = y_df
        self.X_labels = list(X_df)

    def predict(self, X_df):
        """Apply the model
        """
        pass

    def _preprocess_data(self, X_df, copy=True):
        # 1. Convert dataframe into time series
        X_df = df_to_ts(X_df, self.datetime_label, freq='H', copy=copy)

        return X_df

    def _features_extraction(self, X_df, copy=True):
        # 1. Cyclical time encoding
        X_df = add_cyclical_time_feature(X_df, hour_of_day=True, half_hour_of_day=True, week_of_year=True,
                                         month_of_year=True, day_of_week=True, copy=copy)

        # 2. Meteorological features merging
        wp_label = 'wp_value'
        X_df = add_numerical_weather_prediction_median(X_df, wp_label=wp_label, copy=copy)
        X_df = add_wind_speed(X_df, 'U', 'V', 'wind_speed', copy)

        # 3. Meteorological features lags
        X_df = add_lags(X_df, lag_range=[1, 2], to_lag_labels=['U', 'V', 'T', 'CLCT', 'wind_speed'])

        return X_df

    def _features_selection(self, X_df, wp_prefix):
        X_df = remove_numerical_weather_features(X_df, wp_prefix)
        X_df = remove_collinear_drivers(X_df, force_keeping=ID_LABEL, threshold=10)

        return X_df

    def _clean_data(self, X_df):
        pass
