import pandas as pd

from wind_power_forecasting import X_TRAIN_FILE, TIME_LABEL, WF_LABEL, Y_TRAIN_FILE, ID_LABEL, TARGET_LABEL, \
    X_TEST_FILE, SUBMISSION_FILE
from wind_power_forecasting.models.wind_power_forcaster import WindPowerForecaster
from wind_power_forecasting.utils import get_sub_df

if __name__ == '__main__':

    X_train_all_df = pd.read_csv(X_TRAIN_FILE)
    X_test_all_df = pd.read_csv(X_TEST_FILE)
    y_train_all_df = pd.read_csv(Y_TRAIN_FILE)
    predict_dfs = []
    all_wf = X_train_all_df[WF_LABEL].unique()

    for i, wf in enumerate(all_wf):
        print('Wind farm: {}: {}/{}'.format(wf, i + 1, len(all_wf)))

        X_train_wf_df = get_sub_df(X_train_all_df, WF_LABEL, wf)
        X_test_wf_df = get_sub_df(X_test_all_df, WF_LABEL, wf)
        y_train_wf_df = get_sub_df(y_train_all_df, ID_LABEL, X_train_wf_df[ID_LABEL], keep_column=True)

        wpf = WindPowerForecaster(target_label=TARGET_LABEL, datetime_label=TIME_LABEL)

        wpf.fit(X_train_wf_df, y_train_wf_df)
        predict_df = wpf.predict(X_test_wf_df, output_type='dataframe', prediction_label='Production')
        predict_dfs.append(predict_df)

    final_predict_df = pd.concat(predict_dfs).reset_index()

    final_predict_df.to_csv(SUBMISSION_FILE, index=False)
    pass
