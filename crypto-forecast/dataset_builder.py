from pathlib import Path

import numpy as np
import pandas as pd

from services.bitcoin_client import BitcoinClient
from services.google_trends import GoogleTrendsClient


def build_bitcoin_dataset(
        bitcoin_limit=365,
        trends_timeframe="today 12-m",
        keyword="Bitcoin",
        save=True,
        output_file="data/bitcoin_dataset.csv"
):
    """
    Builds the final dataset used for Bitcoin next-day direction prediction.

    The dataset combines:
    - Bitcoin market data
    - Google Trends data
    - engineered features
    - binary target variable

    Target:
    1 = UP, if next day's close is greater than current close
    0 = DOWN, otherwise
    """

    bitcoin_client = BitcoinClient()
    trends_client = GoogleTrendsClient()

    bitcoin_df = bitcoin_client.get_bitcoin_ohlcv(limit=bitcoin_limit)
    trends_df = trends_client.get_interest_over_time(
        keyword=keyword,
        timeframe=trends_timeframe
    )

    bitcoin_df = prepare_bitcoin_dataframe(bitcoin_df)
    trends_df = prepare_trends_dataframe(trends_df)

    dataset = merge_market_and_trends(bitcoin_df, trends_df)

    dataset = add_features(dataset)

    dataset = add_target(dataset)

    dataset = dataset.dropna().reset_index(drop=True)

    if save:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(output_path, index=False)

    target_distribution = dataset["target"].value_counts().to_dict()

    return {
        "message": "Bitcoin dataset created successfully.",
        "dataset_file": output_file,
        "samples": int(len(dataset)),
        "columns": int(len(dataset.columns)),
        "feature_count": int(len(dataset.columns) - 3),
        "target_column": "target",
        "target_distribution": {
            "DOWN": int(target_distribution.get(0, 0)),
            "UP": int(target_distribution.get(1, 0))
        },
        "bitcoin": {
            "samples": int(len(bitcoin_df)),
            "start_date": str(bitcoin_df["date"].min()),
            "end_date": str(bitcoin_df["date"].max())
        },
        "google_trends": {
            "keyword": keyword,
            "timeframe": trends_timeframe,
            "samples": int(len(trends_df))
        }
    }


def prepare_bitcoin_dataframe(df):
    """
    Normalizes the Bitcoin market dataframe.
    Expected columns may come either from Binance/Kraken-style client or custom client.
    """

    df = df.copy()

    if "date" not in df.columns:
        if "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce").dt.date
        elif "open_time" in df.columns:
            df["date"] = pd.to_datetime(df["open_time"], unit="ms", errors="coerce").dt.date
        else:
            raise ValueError("Bitcoin dataframe must contain date, timestamp or open_time column.")

    df["date"] = pd.to_datetime(df["date"]).dt.date

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required_columns = ["date", "open", "high", "low", "close", "volume"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required Bitcoin column: {col}")

    optional_defaults = {
        "quote_asset_volume": 0.0,
        "number_of_trades": 0.0,
        "taker_buy_base_volume": 0.0,
        "taker_buy_quote_volume": 0.0
    }

    for col, default_value in optional_defaults.items():
        if col not in df.columns:
            df[col] = default_value

    df = df.sort_values("date").reset_index(drop=True)

    return df


def prepare_trends_dataframe(df):
    """
    Normalizes the Google Trends dataframe.
    Expected final columns:
    - date
    - trend
    """

    df = df.copy()

    if "date" not in df.columns:
        if df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        else:
            raise ValueError("Google Trends dataframe must contain a date column or DatetimeIndex.")

    df["date"] = pd.to_datetime(df["date"]).dt.date

    if "trend" not in df.columns:
        possible_trend_columns = [
            col for col in df.columns
            if col not in ["date", "isPartial"]
        ]

        if len(possible_trend_columns) == 0:
            raise ValueError("Google Trends dataframe does not contain a trend column.")

        df = df.rename(columns={possible_trend_columns[0]: "trend"})

    df["trend"] = pd.to_numeric(df["trend"], errors="coerce")

    df = df[["date", "trend"]]
    df = df.sort_values("date").reset_index(drop=True)

    return df


def merge_market_and_trends(bitcoin_df, trends_df):
    """
    Merges Bitcoin market data with Google Trends data using date.
    """

    dataset = pd.merge(
        bitcoin_df,
        trends_df,
        on="date",
        how="inner"
    )

    dataset = dataset.sort_values("date").reset_index(drop=True)

    return dataset


def add_features(df):
    """
    Adds market, Google Trends, lag, moving average and volatility features.
    """

    df = df.copy()

    df["price_return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    df["open_close_change"] = (df["close"] - df["open"]) / df["open"]
    df["high_low_range"] = (df["high"] - df["low"]) / df["low"]

    df["close_open_ratio"] = df["close"] / df["open"]
    df["high_close_ratio"] = df["high"] / df["close"]
    df["low_close_ratio"] = df["low"] / df["close"]

    df["volume_change"] = df["volume"].pct_change()
    df["volume_log"] = np.log1p(df["volume"])

    df["trades_change"] = df["number_of_trades"].pct_change()

    df["taker_buy_ratio"] = np.where(
        df["volume"] != 0,
        df["taker_buy_base_volume"] / df["volume"],
        0
    )

    df["trend_change"] = df["trend"].diff()
    df["trend_pct_change"] = df["trend"].pct_change()
    df["trend_pct_change"] = df["trend_pct_change"].replace([np.inf, -np.inf], np.nan)

    lag_values = [1, 2, 3, 5, 7]

    for lag in lag_values:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        df[f"price_return_lag_{lag}"] = df["price_return"].shift(lag)
        df[f"volume_lag_{lag}"] = df["volume"].shift(lag)
        df[f"trend_lag_{lag}"] = df["trend"].shift(lag)
        df[f"trend_change_lag_{lag}"] = df["trend_change"].shift(lag)

    moving_windows = [3, 7, 14]

    for window in moving_windows:
        df[f"close_ma_{window}"] = df["close"].rolling(window).mean()
        df[f"volume_ma_{window}"] = df["volume"].rolling(window).mean()
        df[f"trend_ma_{window}"] = df["trend"].rolling(window).mean()

        df[f"close_ma_ratio_{window}"] = df["close"] / df[f"close_ma_{window}"]
        df[f"volume_ma_ratio_{window}"] = df["volume"] / df[f"volume_ma_{window}"]
        df[f"trend_ma_ratio_{window}"] = df["trend"] / df[f"trend_ma_{window}"]

    df["volatility_7"] = df["price_return"].rolling(7).std()
    df["volatility_14"] = df["price_return"].rolling(14).std()

    df["trend_volatility_7"] = df["trend_change"].rolling(7).std()
    df["trend_volatility_14"] = df["trend_change"].rolling(14).std()

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def add_target(df):
    """
    Adds binary target:
    1 = next close > current close
    0 = otherwise
    """

    df = df.copy()

    df["next_close"] = df["close"].shift(-1)
    df["target"] = (df["next_close"] > df["close"]).astype(int)

    df = df.drop(columns=["next_close"])

    return df


if __name__ == "__main__":
    result = build_bitcoin_dataset()
    print(result)