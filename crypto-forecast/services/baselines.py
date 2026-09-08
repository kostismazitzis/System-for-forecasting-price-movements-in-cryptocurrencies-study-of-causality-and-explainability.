from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class BaselineError(Exception):
    """
    Ειδική εξαίρεση για σφάλματα που σχετίζονται με
    τις baseline μεθόδους πρόβλεψης.
    """


@dataclass(frozen=True)
class BaselinePrediction:
    """
    Αναπαριστά μία πρόβλεψη baseline.

    Attributes
    ----------
    prediction:
        Αριθμητική πρόβλεψη:
        0 = DOWN
        1 = UP

    direction:
        Λεκτική περιγραφή της πρόβλεψης.

    method:
        Η μέθοδος baseline που χρησιμοποιήθηκε.

    details:
        Πρόσθετες πληροφορίες για τον κανόνα πρόβλεψης.
    """

    prediction: int
    direction: str
    method: str
    details: dict[str, Any]


class BaselineModels:
    """
    Υλοποιεί απλές μεθόδους αναφοράς για την πρόβλεψη
    της κατεύθυνσης της τιμής του Bitcoin.

    Οι baseline μέθοδοι χρησιμοποιούνται ως σημεία αναφοράς
    για τη σύγκριση με το μοντέλο μηχανικής μάθησης.

    Περιλαμβάνονται:

    1. Persistence baseline
       Υποθέτει ότι η επόμενη κατεύθυνση θα είναι ίδια
       με την πιο πρόσφατη παρατηρούμενη κατεύθυνση.

    2. Moving Average baseline
       Προβλέπει UP όταν η τρέχουσα τιμή κλεισίματος βρίσκεται
       πάνω από τον κινητό μέσο όρο και DOWN διαφορετικά.
    """

    CLASS_LABELS = {
        0: "DOWN",
        1: "UP",
    }

    REQUIRED_PRICE_COLUMNS = {
        "close",
    }

    def __init__(
        self,
        moving_average_window: int = 7,
    ) -> None:
        """
        Αρχικοποιεί τις baseline μεθόδους.

        Parameters
        ----------
        moving_average_window:
            Το παράθυρο του κινητού μέσου όρου.
            Η προεπιλεγμένη τιμή είναι 7 ημέρες.
        """

        if not isinstance(moving_average_window, int):
            raise TypeError(
                "Το moving_average_window πρέπει να είναι ακέραιος."
            )

        if moving_average_window < 2:
            raise ValueError(
                "Το moving_average_window πρέπει να είναι τουλάχιστον 2."
            )

        self.moving_average_window = moving_average_window

    @staticmethod
    def _validate_dataframe(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Ελέγχει και καθαρίζει το DataFrame εισόδου.

        Returns
        -------
        pandas.DataFrame
            Καθαρισμένο και χρονολογικά ταξινομημένο DataFrame.
        """

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "Το dataframe πρέπει να είναι pandas DataFrame."
            )

        if dataframe.empty:
            raise BaselineError(
                "Το DataFrame εισόδου είναι κενό."
            )

        missing_columns = (
            BaselineModels.REQUIRED_PRICE_COLUMNS
            .difference(dataframe.columns)
        )

        if missing_columns:
            raise BaselineError(
                "Λείπουν υποχρεωτικές στήλες: "
                f"{sorted(missing_columns)}"
            )

        result = dataframe.copy()

        result["close"] = pd.to_numeric(
            result["close"],
            errors="coerce",
        )

        if "date" in result.columns:
            result["date"] = pd.to_datetime(
                result["date"],
                utc=True,
                errors="coerce",
            )

            result = result.dropna(
                subset=["date"]
            )

            result = result.sort_values(
                by="date",
                ascending=True,
            )

        result = result.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        result = result.dropna(
            subset=["close"]
        )

        result = result.reset_index(drop=True)

        if result.empty:
            raise BaselineError(
                "Δεν απέμειναν έγκυρες εγγραφές."
            )

        if (result["close"] <= 0).any():
            raise BaselineError(
                "Οι τιμές κλεισίματος πρέπει να είναι θετικές."
            )

        return result

    @classmethod
    def _direction_label(
        cls,
        prediction: int,
    ) -> str:
        """
        Μετατρέπει αριθμητική πρόβλεψη σε λεκτική ετικέτα.
        """

        if prediction not in cls.CLASS_LABELS:
            raise BaselineError(
                f"Μη έγκυρη κατηγορία πρόβλεψης: {prediction}"
            )

        return cls.CLASS_LABELS[prediction]

    @staticmethod
    def calculate_observed_direction(
        previous_close: float,
        current_close: float,
    ) -> int:
        """
        Υπολογίζει την παρατηρούμενη κατεύθυνση μεταξύ δύο τιμών.

        Returns
        -------
        int
            1 όταν η τιμή αυξήθηκε.
            0 όταν η τιμή μειώθηκε ή παρέμεινε ίση.
        """

        if previous_close <= 0 or current_close <= 0:
            raise ValueError(
                "Οι τιμές κλεισίματος πρέπει να είναι θετικές."
            )

        return int(current_close > previous_close)

    def persistence_predict_one(
        self,
        previous_close: float,
        current_close: float,
    ) -> BaselinePrediction:
        """
        Παράγει μία πρόβλεψη με τον κανόνα persistence.

        Η προβλεπόμενη κατεύθυνση της επόμενης περιόδου
        θεωρείται ίδια με την τελευταία παρατηρούμενη κίνηση.

        Παράδειγμα:
        - αν close(t) > close(t-1), προβλέπει UP,
        - διαφορετικά προβλέπει DOWN.
        """

        prediction = self.calculate_observed_direction(
            previous_close=previous_close,
            current_close=current_close,
        )

        return BaselinePrediction(
            prediction=prediction,
            direction=self._direction_label(prediction),
            method="persistence",
            details={
                "previous_close": float(previous_close),
                "current_close": float(current_close),
                "rule": (
                    "Η επόμενη κατεύθυνση θεωρείται ίδια "
                    "με την τελευταία παρατηρούμενη κίνηση."
                ),
            },
        )

    def moving_average_predict_one(
        self,
        current_close: float,
        moving_average: float,
    ) -> BaselinePrediction:
        """
        Παράγει μία πρόβλεψη με βάση κινητό μέσο όρο.

        Κανόνας:
        - UP όταν current_close > moving_average,
        - DOWN διαφορετικά.
        """

        if current_close <= 0:
            raise ValueError(
                "Η τρέχουσα τιμή κλεισίματος πρέπει να είναι θετική."
            )

        if not np.isfinite(moving_average) or moving_average <= 0:
            raise ValueError(
                "Ο κινητός μέσος όρος πρέπει να είναι έγκυρος "
                "και θετικός."
            )

        prediction = int(
            current_close > moving_average
        )

        return BaselinePrediction(
            prediction=prediction,
            direction=self._direction_label(prediction),
            method=f"moving_average_{self.moving_average_window}",
            details={
                "current_close": float(current_close),
                "moving_average": float(moving_average),
                "window": self.moving_average_window,
                "rule": (
                    "UP όταν η τιμή κλεισίματος βρίσκεται "
                    "πάνω από τον κινητό μέσο όρο."
                ),
            },
        )

    def generate_persistence_predictions(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.Series:
        """
        Δημιουργεί persistence προβλέψεις για ολόκληρο DataFrame.

        Για κάθε χρονική στιγμή t, χρησιμοποιείται η κίνηση
        μεταξύ t-1 και t για την πρόβλεψη της κατεύθυνσης t+1.

        Returns
        -------
        pandas.Series
            Σειρά προβλέψεων με τιμές 0 και 1.
        """

        result = self._validate_dataframe(dataframe)

        predictions = (
            result["close"]
            .diff()
            .gt(0)
            .astype(float)
        )

        predictions.iloc[0] = np.nan

        predictions.name = "persistence_prediction"

        return predictions

    def generate_moving_average_predictions(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.Series:
        """
        Δημιουργεί προβλέψεις βάσει κινητού μέσου όρου.

        Για κάθε χρονική στιγμή:
        - UP όταν close(t) > MA(t),
        - DOWN διαφορετικά.

        Returns
        -------
        pandas.Series
            Σειρά προβλέψεων με τιμές 0 και 1.
        """

        result = self._validate_dataframe(dataframe)

        moving_average = (
            result["close"]
            .rolling(
                window=self.moving_average_window,
                min_periods=self.moving_average_window,
            )
            .mean()
        )

        predictions = (
            result["close"]
            .gt(moving_average)
            .astype(float)
        )

        predictions.loc[
            moving_average.isna()
        ] = np.nan

        predictions.name = (
            f"moving_average_{self.moving_average_window}_prediction"
        )

        return predictions

    def add_baseline_predictions(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Προσθέτει στο DataFrame τις προβλέψεις και των δύο baselines.

        Οι νέες στήλες είναι:

        - persistence_prediction
        - moving_average_X
        """

        result = self._validate_dataframe(dataframe)

        result["persistence_prediction"] = (
            self.generate_persistence_predictions(result)
        )

        moving_average_column = (
            f"close_ma_{self.moving_average_window}"
        )

        if moving_average_column in result.columns:
            moving_average_values = pd.to_numeric(
                result[moving_average_column],
                errors="coerce",
            )
        else:
            moving_average_values = (
                result["close"]
                .rolling(
                    window=self.moving_average_window,
                    min_periods=self.moving_average_window,
                )
                .mean()
            )

            result[moving_average_column] = moving_average_values

        moving_average_prediction_column = (
            f"moving_average_{self.moving_average_window}_prediction"
        )

        result[moving_average_prediction_column] = np.where(
            moving_average_values.notna(),
            (
                result["close"] > moving_average_values
            ).astype(int),
            np.nan,
        )

        return result

    def predict_latest_persistence(
        self,
        dataframe: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Παράγει την τελευταία διαθέσιμη persistence πρόβλεψη.
        """

        result = self._validate_dataframe(dataframe)

        if len(result) < 2:
            raise BaselineError(
                "Απαιτούνται τουλάχιστον δύο εγγραφές "
                "για persistence πρόβλεψη."
            )

        previous_close = float(
            result.iloc[-2]["close"]
        )

        current_close = float(
            result.iloc[-1]["close"]
        )

        prediction = self.persistence_predict_one(
            previous_close=previous_close,
            current_close=current_close,
        )

        output = {
            "prediction": prediction.prediction,
            "direction": prediction.direction,
            "method": prediction.method,
            "details": prediction.details,
        }

        if "date" in result.columns:
            output["date"] = (
                result.iloc[-1]["date"].isoformat()
            )

        return output

    def predict_latest_moving_average(
        self,
        dataframe: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Παράγει την τελευταία διαθέσιμη πρόβλεψη moving average.
        """

        result = self._validate_dataframe(dataframe)

        if len(result) < self.moving_average_window:
            raise BaselineError(
                "Δεν υπάρχουν αρκετές εγγραφές για υπολογισμό "
                f"κινητού μέσου {self.moving_average_window} ημερών."
            )

        current_close = float(
            result.iloc[-1]["close"]
        )

        moving_average = float(
            result["close"]
            .tail(self.moving_average_window)
            .mean()
        )

        prediction = self.moving_average_predict_one(
            current_close=current_close,
            moving_average=moving_average,
        )

        output = {
            "prediction": prediction.prediction,
            "direction": prediction.direction,
            "method": prediction.method,
            "details": prediction.details,
        }

        if "date" in result.columns:
            output["date"] = (
                result.iloc[-1]["date"].isoformat()
            )

        return output


def main() -> None:
    """
    Χειροκίνητη δοκιμή των baseline μοντέλων.

    Εκτελείται με:

        python -m services.baselines
    """

    print("==============================================")
    print("Δοκιμή baseline μοντέλων")
    print("==============================================")

    sample_dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                start="2026-01-01",
                periods=10,
                freq="D",
                tz="UTC",
            ),
            "close": [
                90000,
                91000,
                90500,
                92000,
                93000,
                92500,
                94000,
                94500,
                93500,
                95000,
            ],
        }
    )

    try:
        baselines = BaselineModels(
            moving_average_window=7
        )

        result = baselines.add_baseline_predictions(
            sample_dataframe
        )

        print()
        print("Προβλέψεις baseline:")
        print(
            result[
                [
                    "date",
                    "close",
                    "persistence_prediction",
                    "moving_average_7_prediction",
                ]
            ].to_string(index=False)
        )

        print()
        print("Τελευταία persistence πρόβλεψη:")
        print(
            baselines.predict_latest_persistence(
                sample_dataframe
            )
        )

        print()
        print("Τελευταία moving-average πρόβλεψη:")
        print(
            baselines.predict_latest_moving_average(
                sample_dataframe
            )
        )

    except BaselineError as exc:
        print()
        print("Σφάλμα BaselineModels:")
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