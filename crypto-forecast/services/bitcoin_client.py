import requests
import pandas as pd


class BitcoinClient:
    """
    Client for downloading Bitcoin daily market data.

    It uses Binance public API klines endpoint.
    No API key is required.
    """

    def __init__(self):
        self.base_url = "https://api.binance.com"
        self.symbol = "BTCUSDT"
        self.interval = "1d"

    def get_bitcoin_ohlcv(self, limit=365):
        """
        Main method used by dataset_builder.py.

        Returns daily Bitcoin OHLCV data with columns:
        date, open, high, low, close, volume,
        quote_asset_volume, number_of_trades,
        taker_buy_base_volume, taker_buy_quote_volume
        """

        url = f"{self.base_url}/api/v3/klines"

        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": limit
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore"
        ]

        df = pd.DataFrame(data, columns=columns)

        df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.date

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
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[
            [
                "date",
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
        ]

        df = df.sort_values("date").reset_index(drop=True)

        return df

    def get_historical_data(self, limit=365):
        """
        Compatibility method.
        """

        return self.get_bitcoin_ohlcv(limit=limit)

    def get_market_data(self, limit=365):
        """
        Compatibility method.
        """

        return self.get_bitcoin_ohlcv(limit=limit)