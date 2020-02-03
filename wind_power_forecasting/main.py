import pandas as pd

from wind_power_forecasting import X_TRAIN_FILE, TIME_LABEL, WF_LABEL
from wind_power_forecasting.model.wind_power_forcaster import WindPowerForecaster
from wind_power_forecasting.utils import get_sub_df

if __name__ == '__main__':

    X_all_df = pd.read_csv(X_TRAIN_FILE)

    for wf in X_all_df[WF_LABEL].unique():
        X_wf_df = get_sub_df(X_all_df, WF_LABEL, wf)

        wpf = WindPowerForecaster(target_label = 'ID', datetime_label=TIME_LABEL)

        wpf.fit(X_wf_df, None)

        pass
    pass
