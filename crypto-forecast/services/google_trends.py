from pathlib import Path

import pandas as pd
from pytrends.request import TrendReq


class GoogleTrendsClient:
    """
    Client for downloading or loading cached Google Trends interest-over-time data.

    The final dataframe always has:
    - date
    - trend
    """

    def __init__(
            self,
            keyword="Bitcoin",
            timeframe="today 12-m",
            geo="",
            hl="en-US",
            tz=0
    ):
        self.keyword = keyword
        self.timeframe = timeframe
        self.geo = geo
        self.hl = hl
        self.tz = tz
        self.pytrends = self._create_client()

    def _create_client(self):
        """
        Creates the PyTrends client.
        """

        return TrendReq(
            hl=self.hl,
            tz=self.tz
        )

    def _build_payload(
            self,
            keyword=None,
            timeframe=None
    ):
        """
        Builds the Google Trends request payload.
        """

        selected_keyword = keyword if keyword is not None else self.keyword
        selected_timeframe = timeframe if timeframe is not None else self.timeframe

        self.pytrends.build_payload(
            kw_list=[selected_keyword],
            cat=0,
            timeframe=selected_timeframe,
            geo=self.geo,
            gprop=""
        )

    def fetch_raw_interest(
            self,
            keyword=None,
            timeframe=None
    ):
        """
        Fetches raw Google Trends interest-over-time data from Google.
        """

        self._build_payload(
            keyword=keyword,
            timeframe=timeframe
        )

        dataframe = self.pytrends.interest_over_time()

        if dataframe is None or dataframe.empty:
            raise ValueError("Google Trends returned empty data.")

        return dataframe

    def raw_interest_to_dataframe(
            self,
            dataframe,
            keyword=None
    ):
        """
        Converts raw Google Trends dataframe to clean format:
        date, trend
        """

        selected_keyword = keyword if keyword is not None else self.keyword

        df = dataframe.copy()

        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])

        if "date" not in df.columns:
            df = df.reset_index()

        if selected_keyword in df.columns:
            df = df.rename(columns={selected_keyword: "trend"})
        elif "Bitcoin" in df.columns:
            df = df.rename(columns={"Bitcoin": "trend"})
        elif "trend" not in df.columns:
            possible_columns = [
                column for column in df.columns
                if column not in ["date", "isPartial"]
            ]

            if len(possible_columns) == 0:
                raise ValueError("No Google Trends value column found.")

            df = df.rename(columns={possible_columns[0]: "trend"})

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df["trend"] = pd.to_numeric(df["trend"], errors="coerce")

        df = df[["date", "trend"]]
        df = df.dropna()
        df = df.sort_values("date").reset_index(drop=True)

        self._validate_dataframe(df)

        return df

    def convert_to_daily(self, dataframe):
        """
        Converts Google Trends data to daily format.

        If Google Trends returns weekly data, this method fills the weekly value
        across the corresponding daily dates.
        """

        df = dataframe.copy()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "trend"])
        df = df.sort_values("date").reset_index(drop=True)

        if len(df) < 2:
            df["date"] = df["date"].dt.date
            return df[["date", "trend"]]

        date_differences = df["date"].diff().dropna().dt.days
        median_difference = date_differences.median()

        if median_difference <= 1:
            df["date"] = df["date"].dt.date
            return df[["date", "trend"]]

        start_date = df["date"].min()
        end_date = df["date"].max()

        daily_dates = pd.date_range(
            start=start_date,
            end=end_date,
            freq="D"
        )

        daily_df = pd.DataFrame({
            "date": daily_dates
        })

        df = df.sort_values("date")

        daily_df = pd.merge_asof(
            daily_df,
            df,
            on="date",
            direction="backward"
        )

        daily_df["trend"] = daily_df["trend"].ffill()
        daily_df["date"] = daily_df["date"].dt.date

        daily_df = daily_df[["date", "trend"]]
        daily_df = daily_df.dropna()
        daily_df = daily_df.reset_index(drop=True)

        self._validate_dataframe(daily_df)

        return daily_df

    def load_cached_dataframe(
            self,
            cache_file="data/google_trends.csv"
    ):
        """
        Loads cached Google Trends data if it exists.

        This is useful because Google Trends may return HTTP 429
        when too many requests are sent.
        """

        cache_path = Path(cache_file)

        if not cache_path.exists():
            return None

        df = pd.read_csv(cache_path)

        if df.empty:
            return None

        if "date" not in df.columns:
            return None

        if "trend" not in df.columns:
            possible_columns = [
                column for column in df.columns
                if column not in ["date", "isPartial"]
            ]

            if len(possible_columns) == 0:
                return None

            df = df.rename(columns={possible_columns[0]: "trend"})

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df["trend"] = pd.to_numeric(df["trend"], errors="coerce")

        df = df[["date", "trend"]]
        df = df.dropna()
        df = df.sort_values("date").reset_index(drop=True)

        if df.empty:
            return None

        self._validate_dataframe(df)

        return df

    def fetch_dataframe(
            self,
            keyword=None,
            timeframe=None,
            daily=True
    ):
        """
        Fetches Google Trends data from Google and returns clean dataframe.
        """

        raw_dataframe = self.fetch_raw_interest(
            keyword=keyword,
            timeframe=timeframe
        )

        clean_dataframe = self.raw_interest_to_dataframe(
            dataframe=raw_dataframe,
            keyword=keyword
        )

        if daily:
            clean_dataframe = self.convert_to_daily(clean_dataframe)

        self._validate_dataframe(clean_dataframe)

        return clean_dataframe

    def get_interest_over_time(
            self,
            keyword="Bitcoin",
            timeframe="today 12-m",
            cache_file="data/google_trends.csv"
    ):
        """
        Main method used by dataset_builder.py.

        First tries to load cached data from data/google_trends.csv.
        If no valid cached file exists, it fetches data from Google Trends.
        """

        cached_dataframe = self.load_cached_dataframe(
            cache_file=cache_file
        )

        if cached_dataframe is not None:
            return cached_dataframe

        dataframe = self.fetch_dataframe(
            keyword=keyword,
            timeframe=timeframe,
            daily=True
        )

        cache_path = Path(cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        dataframe.to_csv(
            cache_path,
            index=False
        )

        return dataframe

    def get_trends_data(
            self,
            keyword="Bitcoin",
            timeframe="today 12-m"
    ):
        """
        Compatibility method.
        """

        return self.get_interest_over_time(
            keyword=keyword,
            timeframe=timeframe
        )

    def fetch_trend_data(
            self,
            keyword="Bitcoin",
            timeframe="today 12-m"
    ):
        """
        Compatibility method.
        """

        return self.get_interest_over_time(
            keyword=keyword,
            timeframe=timeframe
        )

    def save_dataframe(
            self,
            dataframe,
            output_file="data/google_trends.csv"
    ):
        """
        Saves Google Trends dataframe to CSV.
        """

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dataframe.to_csv(
            output_path,
            index=False
        )

        return str(output_path)

    def fetch_and_save(
            self,
            keyword="Bitcoin",
            timeframe="today 12-m",
            output_file="data/google_trends.csv"
    ):
        """
        Fetches Google Trends data and saves it to CSV.
        """

        dataframe = self.get_interest_over_time(
            keyword=keyword,
            timeframe=timeframe,
            cache_file=output_file
        )

        self.save_dataframe(
            dataframe=dataframe,
            output_file=output_file
        )

        return {
            "message": "Google Trends data saved successfully.",
            "keyword": keyword,
            "timeframe": timeframe,
            "samples": int(len(dataframe)),
            "output_file": output_file
        }

    @staticmethod
    def _validate_dataframe(dataframe):
        """
        Validates final Google Trends dataframe.
        """

        required_columns = ["date", "trend"]

        for column in required_columns:
            if column not in dataframe.columns:
                raise ValueError(f"Missing required Google Trends column: {column}")

        if dataframe.empty:
            raise ValueError("Google Trends dataframe is empty.")


def main():
    client = GoogleTrendsClient()

    dataframe = client.get_interest_over_time(
        keyword="Bitcoin",
        timeframe="today 12-m"
    )

    print(dataframe.tail())


if __name__ == "__main__":
    main()