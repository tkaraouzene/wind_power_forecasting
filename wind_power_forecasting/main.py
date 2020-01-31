from sklearn.base import RegressorMixin, BaseEstimator


class WindPowerForecaster(BaseEstimator, RegressorMixin):

    def __init__(self, target_label, feature_labels):
        self.target = target_label
        self.features = feature_labels

        pass

    def fit(self, X_df, y_df):
        pass

    def predict(self, X_df):
        pass
