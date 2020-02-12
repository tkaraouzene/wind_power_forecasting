## Changelog

## 1.1.0

## Cross validation

+ **params distribution:** ``RandomizedSearchCV``
+ **scorer:** CAPE (from Anasse library)

## 1.0.3

### Feature extraction

1. cyclical time encoding:

   2. remove ``half_hour_of_day``
   3. remove ``day_of_week``

2. lags:
   + **Features:** wind features only
   + **range:** ``[1]``

3. rollmean:
   + **Features:** ``[wind_speed, wind_vector_azimuth, meteorological_wind_direction]``
   + **period:** ``[3H]``

## 1.0.2

### Refactoring

+ ``model`` to ``models``

### Fitting

+ Estimator = ``RandomForestRegressor``
+ Parameters:

  + set ``random_state=42`` instead of ``None``
+ Fill ``NAs`` strategy:

  2. **Second fill:** By ``self.y_median``, i.e the median Production median (instead of 0)

## 1.0.1

+ add the submission file [submission.csv](results/submission.csv) to the repo

## 1.0.0 -- First submission

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

