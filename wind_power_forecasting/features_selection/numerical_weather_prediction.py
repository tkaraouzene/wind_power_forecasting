def remove_numerical_weather_features(df, wp_prefix):

    wp_labels = [l for l in df if l.startswith(wp_prefix)]
    df.drop(wp_labels, axis=1, inplace=True)
    return df