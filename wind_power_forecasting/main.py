import pandas as pd

from wind_power_forecasting import X_TRAIN_FILE, TIME_LABEL, WF_LABEL
from wind_power_forecasting.features_extraction.feature_extraction import feature_extraction
from wind_power_forecasting.preprocessing.inputs import input_preprocessing
from wind_power_forecasting.utils import get_sub_df

if __name__ == '__main__':

    X_all_df = pd.read_csv(X_TRAIN_FILE)

    for wf in X_all_df[WF_LABEL].unique():
        X_wf_df = get_sub_df(X_all_df, WF_LABEL, wf)

        X_df = input_preprocessing(X_wf_df, TIME_LABEL)

        feature_extraction(X_df, hour_of_day=True, half_hour_of_day=True, week_of_year=True, month_of_year=True,
                           day_of_week=True)

        pass
    pass
