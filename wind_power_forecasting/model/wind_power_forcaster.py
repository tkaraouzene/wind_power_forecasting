from sklearn.base import BaseEstimator, RegressorMixin

from wind_power_forecasting import NWP_PREFIX, ID_LABEL
from wind_power_forecasting.features_extraction.cyclical_time import add_cyclical_time_feature
from wind_power_forecasting.features_extraction.weather import add_numerical_weather_prediction_median, \
    add_numerical_weather_prediction_shift
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

        # 3. Meteorological features lags
        X_df = add_numerical_weather_prediction_shift(X_df, shift_range=[1, 2], wp_label=wp_label, copy=copy)

        return X_df

    def _features_selection(self, X_df, wp_prefix):
        X_df = remove_numerical_weather_features(X_df, wp_prefix)
        kept_features = [ID_LABEL]
        X_df = remove_collinear_drivers(X_df, force_keeping=kept_features, threshold=10)

        return X_df

    def _clean_data(self, X_df):
        pass
