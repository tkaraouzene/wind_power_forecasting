from wind_power_forecasting.features_extraction.cyclical_time import add_cyclical_hour_of_day, \
    add_cyclical_half_hour_of_day, add_cyclical_week_of_year, add_cyclical_month_of_year, add_cyclical_day_of_week, \
    add_cyclical_day_of_month, add_cyclical_minute_of_hour, add_cyclical_minute_of_day, add_cyclical_second_of_minute


def feature_extraction(df,
                       hour_of_day=False,
                       half_hour_of_day=False,
                       week_of_year=False,
                       month_of_year=False,
                       day_of_week=False,
                       day_of_month=False,
                       min_of_hour=False,
                       min_of_day=False,
                       sec_of_min=False,
                       copy=False):
    if hour_of_day:
        df = add_cyclical_hour_of_day(df, copy=copy)

    if half_hour_of_day:
        df = add_cyclical_half_hour_of_day(df, copy=copy)

    if week_of_year:
        df = add_cyclical_week_of_year(df, copy=copy)

    if month_of_year:
        df = add_cyclical_month_of_year(df, copy=copy)

    if day_of_week:
        df = add_cyclical_day_of_week(df, copy=copy)

    if day_of_month:
        df = add_cyclical_day_of_month(df, copy=copy)

    if min_of_hour:
        df = add_cyclical_minute_of_hour(df, copy=copy)

    if min_of_day:
        df = add_cyclical_minute_of_day(df, copy=copy)

    if sec_of_min:
        df = add_cyclical_second_of_minute(df, copy=copy)

    return df
