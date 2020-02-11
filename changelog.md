## Changelog


## 1.0.0 -- First try

### Preprocessing

1. `df_to_ts`


### Feature extraction

1. cyclical time encoding:

   1. ``hour_of_day``
   2. ``half_hour_of_day``
   3. ``week_of_year``
   4. ``month_of_year``
   5. ``day_of_week``

2. numerical weather prediction merging by median
3. wind speed
4. wind vector azimuth
5. meteorological wind direction
6. lags:

   + **Features:** all
   + **range:** ``[1, 2]``

### Feature selection

1. remove all ``NWP`` features
2. variance threshold: ``threshold==0.8``
3. variance inflation factor: ``threshold==5``

### Data cleaning

1. remove remaining ``NAs``
2. Get ``X`` and ``y`` from ``X_df`` and ``y_df``

### Fitting

+ Estimator = ``RandomForestRegressor``
+ Parameters = Default
+ Fill ``NAs`` strategy:

  1. With last valid observation (done within ``_clean_data`` method)
  2. By 0 (if first n lines are ``NAs`` (done within ``_clean_data`` method))

