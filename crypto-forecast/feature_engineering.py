import numpy as np
import pandas as pd


def add_features(df):
    """
    Adds engineered features for Bitcoin direction prediction.

    Features include:
    - price returns
    - log returns
    - OHLC ratios
    - volume changes
    - Google Trends changes
    - lag features
    - moving averages
    - volatility features
    """

    df = df.copy()

    # Basic price features
    df["price_return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    df["open_close_change"] = (df["close"] - df["open"]) / df["open"]
    df["high_low_range"] = (df["high"] - df["low"]) / df["low"]

    df["close_open_ratio"] = df["close"] / df["open"]
    df["high_close_ratio"] = df["high"] / df["close"]
    df["low_close_ratio"] = df["low"] / df["close"]

    # Volume features
    df["volume_change"] = df["volume"].pct_change()
    df["volume_log"] = np.log1p(df["volume"])

    if "number_of_trades" in df.columns:
        df["trades_change"] = df["number_of_trades"].pct_change()
    else:
        df["number_of_trades"] = 0
        df["trades_change"] = 0

    if "taker_buy_base_volume" not in df.columns:
        df["taker_buy_base_volume"] = 0

    df["taker_buy_ratio"] = np.where(
        df["volume"] != 0,
        df["taker_buy_base_volume"] / df["volume"],
        0
    )

    # Google Trends features
    if "trend" in df.columns:
        df["trend_change"] = df["trend"].diff()
        df["trend_pct_change"] = df["trend"].pct_change()
        df["trend_pct_change"] = df["trend_pct_change"].replace([np.inf, -np.inf], np.nan)
    else:
        df["trend"] = 0
        df["trend_change"] = 0
        df["trend_pct_change"] = 0

    # Lag features
    lag_values = [1, 2, 3, 5, 7]

    for lag in lag_values:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        df[f"price_return_lag_{lag}"] = df["price_return"].shift(lag)
        df[f"volume_lag_{lag}"] = df["volume"].shift(lag)
        df[f"trend_lag_{lag}"] = df["trend"].shift(lag)
        df[f"trend_change_lag_{lag}"] = df["trend_change"].shift(lag)

    # Moving average features
    moving_windows = [3, 7, 14]

    for window in moving_windows:
        df[f"close_ma_{window}"] = df["close"].rolling(window).mean()
        df[f"volume_ma_{window}"] = df["volume"].rolling(window).mean()
        df[f"trend_ma_{window}"] = df["trend"].rolling(window).mean()

        df[f"close_ma_ratio_{window}"] = df["close"] / df[f"close_ma_{window}"]
        df[f"volume_ma_ratio_{window}"] = df["volume"] / df[f"volume_ma_{window}"]
        df[f"trend_ma_ratio_{window}"] = df["trend"] / df[f"trend_ma_{window}"]

    # Volatility features
    df["volatility_7"] = df["price_return"].rolling(7).std()
    df["volatility_14"] = df["price_return"].rolling(14).std()

    df["trend_volatility_7"] = df["trend_change"].rolling(7).std()
    df["trend_volatility_14"] = df["trend_change"].rolling(14).std()

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def add_target(df):
    """
    Adds binary target variable.

    target = 1 means next day's close price is higher than current close.
    target = 0 means next day's close price is lower or equal.
    """

    df = df.copy()

    df["next_close"] = df["close"].shift(-1)
    df["target"] = (df["next_close"] > df["close"]).astype(int)

    df = df.drop(columns=["next_close"])

    return df


def get_feature_columns(df, target_column="target"):
    """
    Returns the feature columns used for model training.
    Excludes target, date and timestamp-related columns.
    """

    excluded_columns = {
        target_column,
        "date",
        "timestamp",
        "open_time",
        "close_time"
    }

    feature_columns = [
        col for col in df.columns
        if col not in excluded_columns
    ]

    return feature_columns