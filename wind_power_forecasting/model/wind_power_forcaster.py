from sklearn.base import BaseEstimator, RegressorMixin

from wind_power_forecasting.features_extraction.cyclical_time import add_cyclical_hour_of_day, \
    add_cyclical_half_hour_of_day, add_cyclical_week_of_year, add_cyclical_month_of_year, add_cyclical_day_of_week
from wind_power_forecasting.features_selection.variance_inflation_factor import remove_collinear_drivers
from wind_power_forecasting.preprocessing.inputs import df_to_ts
from wind_power_forecasting.utils import df_to_X_y


class WindPowerForecaster(BaseEstimator, RegressorMixin):

    def __init__(self, target_label: str, datetime_label: str):

        self.target_label = target_label
        self.datetime_label = datetime_label

    def fit(self, X_df, y_df):
        """Fit model."""

        X_df = self._preprocess_data(X_df, copy=False)
        X_df = self._features_extraction(X_df, copy=False)



        # remove NAs...
        X_df = X_df.dropna(axis=1)
        # X_df = self._clean_data()

        # For now features selection cannot be done since data contains NAs
        # X_df = self._features_selection(X_df, copy=False)

        # X, y, X_labels = df_to_X_y(X_y_df=X_df.dropna(), target_label=self.target_label)
        self.X_df = X_df
        # self.y = y
        self.X_labels = list(X_df)

    def predict(self, X_df):
        """Apply the model
        """
        pass

    def _preprocess_data(self, X_df, copy=True):
        # Convert dataframe into time series:
        #   1. Set datetime column as index
        #   2. Convert it into datetime
        #   3. Sort it
        #   4. Force the frequency (fill with missing values).
        X_df = df_to_ts(X_df, self.datetime_label, freq='H')

        return X_df

    def _features_extraction(self, X_df, copy=True):
        # --- Feature extraction --- #
        X_df = add_cyclical_hour_of_day(X_df, copy=copy)
        X_df = add_cyclical_half_hour_of_day(X_df, copy=copy)
        X_df = add_cyclical_week_of_year(X_df, copy=copy)
        X_df = add_cyclical_month_of_year(X_df, copy=copy)
        X_df = add_cyclical_day_of_week(X_df, copy=copy)

        return X_df

    def _features_selection(self, X_df, copy=True):
        # --- Feature selection --- #
        X_df = remove_collinear_drivers(X_df, threshold=10)

        return X_df

    def _clean_data(self, X_df):
        pass
