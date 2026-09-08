from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import (
    BITCOIN_DATA_FILE,
    DATASET_FILE,
    MOVING_AVERAGE_WINDOWS,
    PRICE_LAGS,
    TARGET_COLUMN,
    TRENDS_DATA_FILE,
    TREND_LAGS,
    VOLATILITY_WINDOWS,
)


class DatasetBuilderError(Exception):
    """
    Ειδική εξαίρεση για σφάλματα που σχετίζονται με τη δημιουργία,
    επεξεργασία και αποθήκευση του τελικού συνόλου δεδομένων.
    """


class DatasetBuilder:
    """
    Κλάση δημιουργίας του τελικού dataset της εφαρμογής.

    Η κλάση πραγματοποιεί τα ακόλουθα στάδια:

    1. Φορτώνει τα ιστορικά δεδομένα Bitcoin.
    2. Φορτώνει τα δεδομένα Google Trends.
    3. Μετατρέπει τις ημερομηνίες σε κοινή μορφή.
    4. Συγχρονίζει τις δύο χρονοσειρές ανά ημερομηνία.
    5. Δημιουργεί χαρακτηριστικά τιμών και όγκου.
    6. Δημιουργεί χαρακτηριστικά Google Trends.
    7. Δημιουργεί lag features.
    8. Δημιουργεί κινητούς μέσους όρους.
    9. Δημιουργεί χαρακτηριστικά μεταβλητότητας.
    10. Δημιουργεί τη μεταβλητή στόχου UP/DOWN.
    11. Αφαιρεί γραμμές με ελλιπείς τιμές.
    12. Αποθηκεύει το τελικό dataset σε αρχείο CSV.

    Η μεταβλητή στόχου ορίζεται ως:

        target = 1, όταν close(t+1) > close(t)
        target = 0, διαφορετικά

    Επομένως, το μοντέλο προβλέπει την κατεύθυνση της τιμής
    κλεισίματος του Bitcoin για την επόμενη ημέρα.
    """

    REQUIRED_BITCOIN_COLUMNS = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    REQUIRED_TRENDS_COLUMNS = {
        "date",
        "trend",
    }

    def __init__(
        self,
        price_lags: Iterable[int] = PRICE_LAGS,
        trend_lags: Iterable[int] = TREND_LAGS,
        moving_average_windows: Iterable[int] = MOVING_AVERAGE_WINDOWS,
        volatility_windows: Iterable[int] = VOLATILITY_WINDOWS,
        target_column: str = TARGET_COLUMN,
    ) -> None:
        """
        Αρχικοποιεί το DatasetBuilder.

        Parameters
        ----------
        price_lags:
            Χρονικές υστερήσεις που θα δημιουργηθούν για τα δεδομένα
            τιμών Bitcoin.

        trend_lags:
            Χρονικές υστερήσεις που θα δημιουργηθούν για τα δεδομένα
            Google Trends.

        moving_average_windows:
            Παράθυρα κινητού μέσου όρου.

        volatility_windows:
            Παράθυρα υπολογισμού μεταβλητότητας.

        target_column:
            Το όνομα της μεταβλητής στόχου.
        """

        self.price_lags = self._validate_windows(
            values=price_lags,
            parameter_name="price_lags",
        )

        self.trend_lags = self._validate_windows(
            values=trend_lags,
            parameter_name="trend_lags",
        )

        self.moving_average_windows = self._validate_windows(
            values=moving_average_windows,
            parameter_name="moving_average_windows",
        )

        self.volatility_windows = self._validate_windows(
            values=volatility_windows,
            parameter_name="volatility_windows",
        )

        if not isinstance(target_column, str) or not target_column.strip():
            raise ValueError(
                "Το target_column πρέπει να είναι μη κενή συμβολοσειρά."
            )

        self.target_column = target_column.strip()

    @staticmethod
    def _validate_windows(
        values: Iterable[int],
        parameter_name: str,
    ) -> list[int]:
        """
        Ελέγχει και καθαρίζει λίστα χρονικών παραθύρων.
        """

        try:
            validated_values = sorted(
                set(int(value) for value in values)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Η παράμετρος {parameter_name} πρέπει να περιέχει "
                "μόνο ακεραίους."
            ) from exc

        if not validated_values:
            raise ValueError(
                f"Η παράμετρος {parameter_name} δεν μπορεί να είναι κενή."
            )

        if any(value <= 0 for value in validated_values):
            raise ValueError(
                f"Όλες οι τιμές της παραμέτρου {parameter_name} "
                "πρέπει να είναι θετικές."
            )

        return validated_values

    @staticmethod
    def _load_csv(
        file_path: str | Path,
        dataset_name: str,
    ) -> pd.DataFrame:
        """
        Φορτώνει αρχείο CSV και επιστρέφει pandas DataFrame.
        """

        path = Path(file_path)

        if not path.exists():
            raise DatasetBuilderError(
                f"Δεν βρέθηκε το αρχείο {dataset_name}: {path}"
            )

        if not path.is_file():
            raise DatasetBuilderError(
                f"Η διαδρομή δεν αντιστοιχεί σε αρχείο: {path}"
            )

        try:
            dataframe = pd.read_csv(path)
        except pd.errors.EmptyDataError as exc:
            raise DatasetBuilderError(
                f"Το αρχείο {dataset_name} είναι κενό."
            ) from exc
        except pd.errors.ParserError as exc:
            raise DatasetBuilderError(
                f"Το αρχείο {dataset_name} δεν είναι έγκυρο CSV."
            ) from exc
        except (OSError, PermissionError) as exc:
            raise DatasetBuilderError(
                f"Δεν ήταν δυνατή η ανάγνωση του αρχείου: {path}"
            ) from exc

        if dataframe.empty:
            raise DatasetBuilderError(
                f"Το αρχείο {dataset_name} δεν περιέχει εγγραφές."
            )

        return dataframe

    @staticmethod
    def _normalize_date_column(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Μετατρέπει τη στήλη date σε κοινή ημερήσια μορφή UTC.

        Η ώρα αφαιρείται, ώστε οι χρονοσειρές να συγχρονίζονται
        αποκλειστικά με βάση την ημερομηνία.
        """

        if "date" not in dataframe.columns:
            raise DatasetBuilderError(
                "Το DataFrame δεν περιέχει τη στήλη date."
            )

        normalized_dataframe = dataframe.copy()

        normalized_dataframe["date"] = pd.to_datetime(
            normalized_dataframe["date"],
            utc=True,
            errors="coerce",
        )

        normalized_dataframe = normalized_dataframe.dropna(
            subset=["date"]
        )

        normalized_dataframe["date"] = (
            normalized_dataframe["date"]
            .dt.normalize()
        )

        normalized_dataframe = normalized_dataframe.drop_duplicates(
            subset=["date"],
            keep="last",
        )

        normalized_dataframe = normalized_dataframe.sort_values(
            by="date",
            ascending=True,
        )

        normalized_dataframe = normalized_dataframe.reset_index(
            drop=True
        )

        if normalized_dataframe.empty:
            raise DatasetBuilderError(
                "Δεν απέμειναν έγκυρες ημερομηνίες μετά την επεξεργασία."
            )

        return normalized_dataframe

    def load_bitcoin_data(
        self,
        file_path: str | Path = BITCOIN_DATA_FILE,
    ) -> pd.DataFrame:
        """
        Φορτώνει και ελέγχει τα δεδομένα Bitcoin.
        """

        dataframe = self._load_csv(
            file_path=file_path,
            dataset_name="Bitcoin",
        )

        missing_columns = self.REQUIRED_BITCOIN_COLUMNS.difference(
            dataframe.columns
        )

        if missing_columns:
            raise DatasetBuilderError(
                "Λείπουν υποχρεωτικές στήλες από τα δεδομένα Bitcoin: "
                f"{sorted(missing_columns)}"
            )

        dataframe = self._normalize_date_column(dataframe)

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        optional_numeric_columns = [
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        ]

        for column in numeric_columns + optional_numeric_columns:
            if column in dataframe.columns:
                dataframe[column] = pd.to_numeric(
                    dataframe[column],
                    errors="coerce",
                )

        dataframe = dataframe.dropna(
            subset=numeric_columns
        )

        if dataframe.empty:
            raise DatasetBuilderError(
                "Δεν υπάρχουν έγκυρες εγγραφές Bitcoin."
            )

        if (dataframe[["open", "high", "low", "close"]] <= 0).any().any():
            raise DatasetBuilderError(
                "Εντοπίστηκαν μη θετικές τιμές στα δεδομένα Bitcoin."
            )

        if (dataframe["volume"] < 0).any():
            raise DatasetBuilderError(
                "Εντοπίστηκε αρνητικός όγκος συναλλαγών."
            )

        return dataframe

    def load_trends_data(
        self,
        file_path: str | Path = TRENDS_DATA_FILE,
    ) -> pd.DataFrame:
        """
        Φορτώνει και ελέγχει τα δεδομένα Google Trends.
        """

        dataframe = self._load_csv(
            file_path=file_path,
            dataset_name="Google Trends",
        )

        missing_columns = self.REQUIRED_TRENDS_COLUMNS.difference(
            dataframe.columns
        )

        if missing_columns:
            raise DatasetBuilderError(
                "Λείπουν υποχρεωτικές στήλες από τα δεδομένα "
                f"Google Trends: {sorted(missing_columns)}"
            )

        dataframe = self._normalize_date_column(dataframe)

        dataframe["trend"] = pd.to_numeric(
            dataframe["trend"],
            errors="coerce",
        )

        dataframe = dataframe.dropna(
            subset=["trend"]
        )

        if dataframe.empty:
            raise DatasetBuilderError(
                "Δεν υπάρχουν έγκυρες εγγραφές Google Trends."
            )

        invalid_trend_values = (
            dataframe["trend"] < 0
        ) | (
            dataframe["trend"] > 100
        )

        if invalid_trend_values.any():
            raise DatasetBuilderError(
                "Οι τιμές Google Trends πρέπει να βρίσκονται "
                "στο διάστημα 0–100."
            )

        return dataframe

    @staticmethod
    def merge_data(
        bitcoin_dataframe: pd.DataFrame,
        trends_dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Συγχρονίζει τα δεδομένα Bitcoin και Google Trends ανά ημερομηνία.

        Χρησιμοποιείται inner join, ώστε να διατηρούνται μόνο οι
        ημερομηνίες που υπάρχουν και στις δύο πηγές.
        """

        if not isinstance(bitcoin_dataframe, pd.DataFrame):
            raise TypeError(
                "Το bitcoin_dataframe πρέπει να είναι pandas DataFrame."
            )

        if not isinstance(trends_dataframe, pd.DataFrame):
            raise TypeError(
                "Το trends_dataframe πρέπει να είναι pandas DataFrame."
            )

        if bitcoin_dataframe.empty:
            raise DatasetBuilderError(
                "Το DataFrame Bitcoin είναι κενό."
            )

        if trends_dataframe.empty:
            raise DatasetBuilderError(
                "Το DataFrame Google Trends είναι κενό."
            )

        merged_dataframe = pd.merge(
            bitcoin_dataframe,
            trends_dataframe[["date", "trend"]],
            on="date",
            how="inner",
            validate="one_to_one",
        )

        merged_dataframe = merged_dataframe.sort_values(
            by="date",
            ascending=True,
        )

        merged_dataframe = merged_dataframe.reset_index(
            drop=True
        )

        if merged_dataframe.empty:
            raise DatasetBuilderError(
                "Δεν βρέθηκαν κοινές ημερομηνίες μεταξύ Bitcoin "
                "και Google Trends."
            )

        if len(merged_dataframe) < 30:
            raise DatasetBuilderError(
                "Οι κοινές εγγραφές είναι λιγότερες από 30. "
                "Απαιτείται μεγαλύτερο κοινό χρονικό διάστημα."
            )

        return merged_dataframe

    @staticmethod
    def add_price_features(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Δημιουργεί βασικά χαρακτηριστικά τιμών Bitcoin.
        """

        result = dataframe.copy()

        result["price_return"] = result["close"].pct_change()

        result["log_return"] = np.log(
            result["close"] / result["close"].shift(1)
        )

        result["open_close_change"] = (
            result["close"] - result["open"]
        ) / result["open"]

        result["high_low_range"] = (
            result["high"] - result["low"]
        ) / result["close"]

        result["close_open_ratio"] = (
            result["close"] / result["open"]
        )

        result["high_close_ratio"] = (
            result["high"] / result["close"]
        )

        result["low_close_ratio"] = (
            result["low"] / result["close"]
        )

        return result

    @staticmethod
    def add_volume_features(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Δημιουργεί χαρακτηριστικά όγκου συναλλαγών.
        """

        result = dataframe.copy()

        result["volume_change"] = result["volume"].pct_change()

        result["volume_log"] = np.log1p(
            result["volume"]
        )

        if "number_of_trades" in result.columns:
            result["trades_change"] = (
                result["number_of_trades"].pct_change()
            )

        if "taker_buy_base_volume" in result.columns:
            result["taker_buy_ratio"] = (
                result["taker_buy_base_volume"]
                / result["volume"].replace(0, np.nan)
            )

        return result

    @staticmethod
    def add_trend_features(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Δημιουργεί βασικά χαρακτηριστικά Google Trends.
        """

        result = dataframe.copy()

        result["trend_change"] = result["trend"].diff()

        result["trend_pct_change"] = (
            result["trend"]
            .replace(0, np.nan)
            .pct_change()
        )

        return result

    def add_lag_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Δημιουργεί χρονικές υστερήσεις για τιμές και Google Trends.
        """

        result = dataframe.copy()

        price_lag_columns = [
            "close",
            "price_return",
            "volume",
        ]

        for lag in self.price_lags:
            for column in price_lag_columns:
                if column in result.columns:
                    result[f"{column}_lag_{lag}"] = (
                        result[column].shift(lag)
                    )

        trend_lag_columns = [
            "trend",
            "trend_change",
        ]

        for lag in self.trend_lags:
            for column in trend_lag_columns:
                if column in result.columns:
                    result[f"{column}_lag_{lag}"] = (
                        result[column].shift(lag)
                    )

        return result

    def add_moving_average_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Δημιουργεί κινητούς μέσους όρους για τιμές, όγκο και trends.
        """

        result = dataframe.copy()

        for window in self.moving_average_windows:
            result[f"close_ma_{window}"] = (
                result["close"]
                .rolling(window=window)
                .mean()
            )

            result[f"volume_ma_{window}"] = (
                result["volume"]
                .rolling(window=window)
                .mean()
            )

            result[f"trend_ma_{window}"] = (
                result["trend"]
                .rolling(window=window)
                .mean()
            )

            result[f"close_to_ma_{window}"] = (
                result["close"]
                / result[f"close_ma_{window}"]
            )

            result[f"trend_to_ma_{window}"] = (
                result["trend"]
                / result[f"trend_ma_{window}"].replace(0, np.nan)
            )

        return result

    def add_volatility_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Δημιουργεί χαρακτηριστικά ιστορικής μεταβλητότητας.
        """

        result = dataframe.copy()

        for window in self.volatility_windows:
            result[f"volatility_{window}"] = (
                result["price_return"]
                .rolling(window=window)
                .std()
            )

            result[f"trend_volatility_{window}"] = (
                result["trend_change"]
                .rolling(window=window)
                .std()
            )

        return result

    def add_target(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Δημιουργεί τον στόχο πρόβλεψης της επόμενης ημέρας.

        target = 1:
            Η επόμενη τιμή κλεισίματος είναι μεγαλύτερη.

        target = 0:
            Η επόμενη τιμή κλεισίματος είναι μικρότερη ή ίση.
        """

        result = dataframe.copy()

        result["next_close"] = result["close"].shift(-1)

        result[self.target_column] = np.where(
            result["next_close"] > result["close"],
            1,
            0,
        )

        # Η τελευταία γραμμή δεν έχει πραγματική τιμή next_close.
        result.loc[
            result["next_close"].isna(),
            self.target_column,
        ] = np.nan

        return result

    @staticmethod
    def replace_infinite_values(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Αντικαθιστά θετικές και αρνητικές άπειρες τιμές με NaN.
        """

        return dataframe.replace(
            [np.inf, -np.inf],
            np.nan,
        )

    def clean_final_dataset(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Καθαρίζει το τελικό dataset μετά το feature engineering.
        """

        result = self.replace_infinite_values(
            dataframe=dataframe.copy()
        )

        result = result.dropna().copy()

        if result.empty:
            raise DatasetBuilderError(
                "Το τελικό dataset είναι κενό μετά την αφαίρεση "
                "των ελλιπών τιμών."
            )

        result[self.target_column] = result[
            self.target_column
        ].astype(int)

        unique_targets = sorted(
            result[self.target_column].unique().tolist()
        )

        if not set(unique_targets).issubset({0, 1}):
            raise DatasetBuilderError(
                "Η μεταβλητή στόχου περιέχει μη έγκυρες τιμές: "
                f"{unique_targets}"
            )

        if result[self.target_column].nunique() < 2:
            raise DatasetBuilderError(
                "Το dataset περιέχει μόνο μία κατηγορία στόχου. "
                "Δεν είναι δυνατή η εκπαίδευση ταξινομητή."
            )

        result = result.sort_values(
            by="date",
            ascending=True,
        )

        result = result.reset_index(drop=True)

        return result

    def build_dataset_from_dataframes(
        self,
        bitcoin_dataframe: pd.DataFrame,
        trends_dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Δημιουργεί το τελικό dataset από δύο DataFrames.
        """

        bitcoin_dataframe = self._normalize_date_column(
            bitcoin_dataframe
        )

        trends_dataframe = self._normalize_date_column(
            trends_dataframe
        )

        dataframe = self.merge_data(
            bitcoin_dataframe=bitcoin_dataframe,
            trends_dataframe=trends_dataframe,
        )

        dataframe = self.add_price_features(dataframe)
        dataframe = self.add_volume_features(dataframe)
        dataframe = self.add_trend_features(dataframe)
        dataframe = self.add_lag_features(dataframe)
        dataframe = self.add_moving_average_features(dataframe)
        dataframe = self.add_volatility_features(dataframe)
        dataframe = self.add_target(dataframe)
        dataframe = self.clean_final_dataset(dataframe)

        return dataframe

    def build_dataset(
        self,
        bitcoin_file: str | Path = BITCOIN_DATA_FILE,
        trends_file: str | Path = TRENDS_DATA_FILE,
    ) -> pd.DataFrame:
        """
        Φορτώνει τα αρχεία εισόδου και δημιουργεί το τελικό dataset.
        """

        bitcoin_dataframe = self.load_bitcoin_data(
            file_path=bitcoin_file
        )

        trends_dataframe = self.load_trends_data(
            file_path=trends_file
        )

        return self.build_dataset_from_dataframes(
            bitcoin_dataframe=bitcoin_dataframe,
            trends_dataframe=trends_dataframe,
        )

    @staticmethod
    def save_dataset(
        dataframe: pd.DataFrame,
        output_file: str | Path = DATASET_FILE,
    ) -> Path:
        """
        Αποθηκεύει το τελικό dataset σε αρχείο CSV.
        """

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "Το dataframe πρέπει να είναι pandas DataFrame."
            )

        if dataframe.empty:
            raise DatasetBuilderError(
                "Δεν είναι δυνατή η αποθήκευση κενού dataset."
            )

        output_path = Path(output_file)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            dataframe.to_csv(
                output_path,
                index=False,
                encoding="utf-8",
            )
        except (OSError, PermissionError) as exc:
            raise DatasetBuilderError(
                f"Αποτυχία αποθήκευσης dataset: {output_path}"
            ) from exc

        return output_path

    def build_and_save(
        self,
        bitcoin_file: str | Path = BITCOIN_DATA_FILE,
        trends_file: str | Path = TRENDS_DATA_FILE,
        output_file: str | Path = DATASET_FILE,
    ) -> pd.DataFrame:
        """
        Δημιουργεί και αποθηκεύει το τελικό dataset.
        """

        dataframe = self.build_dataset(
            bitcoin_file=bitcoin_file,
            trends_file=trends_file,
        )

        self.save_dataset(
            dataframe=dataframe,
            output_file=output_file,
        )

        return dataframe

    def get_feature_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> list[str]:
        """
        Επιστρέφει τις στήλες που μπορούν να χρησιμοποιηθούν
        ως χαρακτηριστικά του μοντέλου.

        Εξαιρούνται:
        - date,
        - target,
        - next_close.
        """

        excluded_columns = {
            "date",
            self.target_column,
            "next_close",
        }

        feature_columns = [
            column
            for column in dataframe.columns
            if column not in excluded_columns
            and pd.api.types.is_numeric_dtype(dataframe[column])
        ]

        if not feature_columns:
            raise DatasetBuilderError(
                "Δεν βρέθηκαν αριθμητικά χαρακτηριστικά."
            )

        return feature_columns


def main() -> None:
    """
    Χειροκίνητη δοκιμή του DatasetBuilder.

    Εκτελείται με:

        python -m services.dataset_builder
    """

    print("==============================================")
    print("Έναρξη δημιουργίας τελικού dataset")
    print("==============================================")

    try:
        builder = DatasetBuilder()

        dataset = builder.build_and_save()

        feature_columns = builder.get_feature_columns(dataset)

        print()
        print("Η δημιουργία του dataset ολοκληρώθηκε.")
        print(f"Πλήθος εγγραφών: {len(dataset)}")
        print(f"Πλήθος συνολικών στηλών: {len(dataset.columns)}")
        print(f"Πλήθος χαρακτηριστικών: {len(feature_columns)}")
        print(f"Αρχείο αποθήκευσης: {DATASET_FILE}")

        print()
        print("Πρώτη ημερομηνία:")
        print(dataset["date"].min())

        print()
        print("Τελευταία ημερομηνία:")
        print(dataset["date"].max())

        print()
        print("Κατανομή μεταβλητής στόχου:")
        print(
            dataset[TARGET_COLUMN]
            .value_counts()
            .sort_index()
            .rename(
                index={
                    0: "DOWN",
                    1: "UP",
                }
            )
            .to_string()
        )

        print()
        print("Χαρακτηριστικά μοντέλου:")
        for index, column in enumerate(
            feature_columns,
            start=1,
        ):
            print(f"{index:02d}. {column}")

        print()
        print("Τελευταίες πέντε εγγραφές:")
        print(
            dataset.tail().to_string(
                index=False
            )
        )

    except DatasetBuilderError as exc:
        print()
        print("Σφάλμα DatasetBuilder:")
        print(exc)

    except (ValueError, TypeError) as exc:
        print()
        print("Σφάλμα παραμέτρων:")
        print(exc)

    except Exception as exc:
        print()
        print("Μη αναμενόμενο σφάλμα:")
        print(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()