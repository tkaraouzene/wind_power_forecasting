import pandas as pd

from wind_power_forecasting import X_TRAIN_FILE, TIME_LABEL, WF_LABEL, Y_TRAIN_FILE, ID_LABEL, TARGET_LABEL
from wind_power_forecasting.model.wind_power_forcaster import WindPowerForecaster
from wind_power_forecasting.utils import get_sub_df

if __name__ == '__main__':

    X_all_df = pd.read_csv(X_TRAIN_FILE)
    y_all_df = pd.read_csv(Y_TRAIN_FILE)

    for wf in X_all_df[WF_LABEL].unique():
        X_wf_df = get_sub_df(X_all_df, WF_LABEL, wf)
        y_wf_df = get_sub_df(y_all_df, ID_LABEL, X_wf_df[ID_LABEL], keep_column=True)

        wpf = WindPowerForecaster(target_label=TARGET_LABEL, datetime_label=TIME_LABEL)

        wpf.fit(X_wf_df, y_wf_df)

        predict_df = wpf.add_prediction(X_wf_df, na_rm=True)
        pass
    pass
