from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class MetricsError(Exception):
    """
    Ειδική εξαίρεση για σφάλματα που σχετίζονται με
    τον υπολογισμό και την παρουσίαση μετρικών αξιολόγησης.
    """


@dataclass(frozen=True)
class ConfusionMatrixResult:
    """
    Αναπαριστά τα στοιχεία ενός πίνακα σύγχυσης δυαδικής ταξινόμησης.

    Η θετική κατηγορία είναι:

        1 = UP

    Η αρνητική κατηγορία είναι:

        0 = DOWN

    Attributes
    ----------
    true_negative:
        Πραγματική κατηγορία DOWN και πρόβλεψη DOWN.

    false_positive:
        Πραγματική κατηγορία DOWN και πρόβλεψη UP.

    false_negative:
        Πραγματική κατηγορία UP και πρόβλεψη DOWN.

    true_positive:
        Πραγματική κατηγορία UP και πρόβλεψη UP.
    """

    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    @property
    def total(self) -> int:
        """
        Επιστρέφει το συνολικό πλήθος προβλέψεων.
        """

        return (
            self.true_negative
            + self.false_positive
            + self.false_negative
            + self.true_positive
        )

    @property
    def correct_predictions(self) -> int:
        """
        Επιστρέφει το πλήθος σωστών προβλέψεων.
        """

        return self.true_negative + self.true_positive

    @property
    def incorrect_predictions(self) -> int:
        """
        Επιστρέφει το πλήθος λανθασμένων προβλέψεων.
        """

        return self.false_positive + self.false_negative

    def to_matrix(self) -> list[list[int]]:
        """
        Επιστρέφει τον πίνακα σύγχυσης σε μορφή 2×2.

        Η διάταξη είναι:

            [[TN, FP],
             [FN, TP]]
        """

        return [
            [
                self.true_negative,
                self.false_positive,
            ],
            [
                self.false_negative,
                self.true_positive,
            ],
        ]

    def to_dict(self) -> dict[str, Any]:
        """
        Επιστρέφει αναλυτική αναπαράσταση σε dictionary.
        """

        result = asdict(self)

        result.update(
            {
                "matrix": self.to_matrix(),
                "total": self.total,
                "correct_predictions": self.correct_predictions,
                "incorrect_predictions": self.incorrect_predictions,
            }
        )

        return result


@dataclass(frozen=True)
class ClassificationMetricsResult:
    """
    Αναπαριστά τα αποτελέσματα αξιολόγησης ενός ταξινομητή.

    Attributes
    ----------
    accuracy:
        Ποσοστό συνολικά σωστών προβλέψεων.

    precision:
        Ποσοστό σωστών προβλέψεων UP μεταξύ όλων των προβλέψεων UP.

    recall:
        Ποσοστό πραγματικών UP που εντοπίστηκαν σωστά.

    f1_score:
        Αρμονικός μέσος Precision και Recall.

    sample_count:
        Συνολικό πλήθος προβλέψεων.

    confusion_matrix:
        Αναλυτικός πίνακας σύγχυσης.
    """

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sample_count: int
    confusion_matrix: ConfusionMatrixResult
    actual_down_count: int
    actual_up_count: int
    predicted_down_count: int
    predicted_up_count: int

    def to_dict(self) -> dict[str, Any]:
        """
        Επιστρέφει τις μετρικές σε dictionary κατάλληλο για JSON.
        """

        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "sample_count": self.sample_count,
            "actual_distribution": {
                "down": self.actual_down_count,
                "up": self.actual_up_count,
            },
            "predicted_distribution": {
                "down": self.predicted_down_count,
                "up": self.predicted_up_count,
            },
            "confusion_matrix": self.confusion_matrix.to_dict(),
        }


class ClassificationMetrics:
    """
    Υπηρεσία υπολογισμού μετρικών δυαδικής ταξινόμησης.

    Οι υποστηριζόμενες κατηγορίες είναι:

        0 = DOWN
        1 = UP

    Η κλάση μπορεί να χρησιμοποιηθεί για την αξιολόγηση:

    - του μοντέλου μηχανικής μάθησης,
    - του Persistence baseline,
    - του Moving Average baseline,
    - οποιασδήποτε άλλης μεθόδου δυαδικής πρόβλεψης.
    """

    VALID_LABELS = {0, 1}

    CLASS_LABELS = {
        0: "DOWN",
        1: "UP",
    }

    def __init__(
        self,
        decimal_places: int = 6,
        positive_label: int = 1,
    ) -> None:
        """
        Αρχικοποιεί την υπηρεσία μετρικών.

        Parameters
        ----------
        decimal_places:
            Πλήθος δεκαδικών ψηφίων που θα διατηρούνται.

        positive_label:
            Η κατηγορία που θεωρείται θετική.
            Στην εργασία χρησιμοποιείται η κατηγορία 1 = UP.
        """

        if not isinstance(decimal_places, int):
            raise TypeError(
                "Το decimal_places πρέπει να είναι ακέραιος αριθμός."
            )

        if decimal_places < 0 or decimal_places > 12:
            raise ValueError(
                "Το decimal_places πρέπει να βρίσκεται μεταξύ 0 και 12."
            )

        if positive_label not in self.VALID_LABELS:
            raise ValueError(
                "Το positive_label πρέπει να είναι 0 ή 1."
            )

        self.decimal_places = decimal_places
        self.positive_label = positive_label

    @staticmethod
    def _to_one_dimensional_array(
        values: Iterable[Any] | pd.Series | np.ndarray,
        parameter_name: str,
    ) -> np.ndarray:
        """
        Μετατρέπει τις τιμές εισόδου σε μονοδιάστατο NumPy array.
        """

        if values is None:
            raise TypeError(
                f"Η παράμετρος {parameter_name} δεν μπορεί να είναι None."
            )

        if isinstance(values, pd.Series):
            array = values.to_numpy()

        elif isinstance(values, np.ndarray):
            array = values.copy()

        else:
            try:
                array = np.asarray(list(values))
            except TypeError as exc:
                raise TypeError(
                    f"Η παράμετρος {parameter_name} πρέπει να είναι iterable."
                ) from exc

        if array.ndim == 0:
            array = array.reshape(1)

        if array.ndim != 1:
            raise MetricsError(
                f"Η παράμετρος {parameter_name} πρέπει να είναι "
                "μονοδιάστατη."
            )

        return array

    @classmethod
    def _validate_binary_values(
        cls,
        values: np.ndarray,
        parameter_name: str,
    ) -> np.ndarray:
        """
        Ελέγχει και μετατρέπει τις τιμές σε δυαδικούς ακεραίους.
        """

        if values.size == 0:
            raise MetricsError(
                f"Η παράμετρος {parameter_name} είναι κενή."
            )

        numeric_values = pd.to_numeric(
            pd.Series(values),
            errors="coerce",
        )

        if numeric_values.isna().any():
            raise MetricsError(
                f"Η παράμετρος {parameter_name} περιέχει "
                "κενές ή μη αριθμητικές τιμές."
            )

        numeric_array = numeric_values.to_numpy(dtype=float)

        if not np.isfinite(numeric_array).all():
            raise MetricsError(
                f"Η παράμετρος {parameter_name} περιέχει "
                "άπειρες ή μη έγκυρες τιμές."
            )

        integer_array = numeric_array.astype(int)

        if not np.array_equal(
            numeric_array,
            integer_array.astype(float),
        ):
            raise MetricsError(
                f"Η παράμετρος {parameter_name} πρέπει να περιέχει "
                "μόνο ακέραιες τιμές 0 και 1."
            )

        unique_values = set(integer_array.tolist())

        if not unique_values.issubset(cls.VALID_LABELS):
            raise MetricsError(
                f"Η παράμετρος {parameter_name} περιέχει "
                f"μη έγκυρες κατηγορίες: {sorted(unique_values)}"
            )

        return integer_array

    def validate_inputs(
        self,
        actual_values: Iterable[Any] | pd.Series | np.ndarray,
        predicted_values: Iterable[Any] | pd.Series | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Ελέγχει ότι οι πραγματικές και προβλεπόμενες τιμές
        είναι κατάλληλες για αξιολόγηση.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Καθαρισμένοι πίνακες πραγματικών και προβλεπόμενων τιμών.
        """

        actual_array = self._to_one_dimensional_array(
            values=actual_values,
            parameter_name="actual_values",
        )

        predicted_array = self._to_one_dimensional_array(
            values=predicted_values,
            parameter_name="predicted_values",
        )

        if len(actual_array) != len(predicted_array):
            raise MetricsError(
                "Οι πραγματικές και οι προβλεπόμενες τιμές πρέπει "
                "να έχουν το ίδιο πλήθος εγγραφών."
            )

        actual_array = self._validate_binary_values(
            values=actual_array,
            parameter_name="actual_values",
        )

        predicted_array = self._validate_binary_values(
            values=predicted_array,
            parameter_name="predicted_values",
        )

        return actual_array, predicted_array

    @staticmethod
    def calculate_confusion_matrix(
        actual_values: np.ndarray,
        predicted_values: np.ndarray,
    ) -> ConfusionMatrixResult:
        """
        Υπολογίζει τον πίνακα σύγχυσης με σταθερή σειρά κατηγοριών.

        Η διάταξη είναι:

            [[TN, FP],
             [FN, TP]]
        """

        matrix = confusion_matrix(
            actual_values,
            predicted_values,
            labels=[0, 1],
        )

        if matrix.shape != (2, 2):
            raise MetricsError(
                "Ο πίνακας σύγχυσης δεν έχει την αναμενόμενη "
                "διάσταση 2×2."
            )

        true_negative = int(matrix[0, 0])
        false_positive = int(matrix[0, 1])
        false_negative = int(matrix[1, 0])
        true_positive = int(matrix[1, 1])

        return ConfusionMatrixResult(
            true_negative=true_negative,
            false_positive=false_positive,
            false_negative=false_negative,
            true_positive=true_positive,
        )

    def calculate(
        self,
        actual_values: Iterable[Any] | pd.Series | np.ndarray,
        predicted_values: Iterable[Any] | pd.Series | np.ndarray,
    ) -> ClassificationMetricsResult:
        """
        Υπολογίζει όλες τις μετρικές δυαδικής ταξινόμησης.

        Parameters
        ----------
        actual_values:
            Οι πραγματικές κατηγορίες.

        predicted_values:
            Οι κατηγορίες που προβλέφθηκαν.

        Returns
        -------
        ClassificationMetricsResult
            Όλες οι μετρικές και ο πίνακας σύγχυσης.
        """

        actual_array, predicted_array = self.validate_inputs(
            actual_values=actual_values,
            predicted_values=predicted_values,
        )

        confusion_result = self.calculate_confusion_matrix(
            actual_values=actual_array,
            predicted_values=predicted_array,
        )

        accuracy = accuracy_score(
            actual_array,
            predicted_array,
        )

        precision = precision_score(
            actual_array,
            predicted_array,
            pos_label=self.positive_label,
            zero_division=0,
        )

        recall = recall_score(
            actual_array,
            predicted_array,
            pos_label=self.positive_label,
            zero_division=0,
        )

        f1 = f1_score(
            actual_array,
            predicted_array,
            pos_label=self.positive_label,
            zero_division=0,
        )

        actual_down_count = int(
            np.sum(actual_array == 0)
        )

        actual_up_count = int(
            np.sum(actual_array == 1)
        )

        predicted_down_count = int(
            np.sum(predicted_array == 0)
        )

        predicted_up_count = int(
            np.sum(predicted_array == 1)
        )

        return ClassificationMetricsResult(
            accuracy=round(
                float(accuracy),
                self.decimal_places,
            ),
            precision=round(
                float(precision),
                self.decimal_places,
            ),
            recall=round(
                float(recall),
                self.decimal_places,
            ),
            f1_score=round(
                float(f1),
                self.decimal_places,
            ),
            sample_count=int(len(actual_array)),
            confusion_matrix=confusion_result,
            actual_down_count=actual_down_count,
            actual_up_count=actual_up_count,
            predicted_down_count=predicted_down_count,
            predicted_up_count=predicted_up_count,
        )

    def calculate_as_dict(
        self,
        actual_values: Iterable[Any] | pd.Series | np.ndarray,
        predicted_values: Iterable[Any] | pd.Series | np.ndarray,
    ) -> dict[str, Any]:
        """
        Υπολογίζει και επιστρέφει τις μετρικές σε dictionary.

        Η μορφή αυτή χρησιμοποιείται από το REST API και τα JSON reports.
        """

        result = self.calculate(
            actual_values=actual_values,
            predicted_values=predicted_values,
        )

        return result.to_dict()

    def compare_methods(
        self,
        actual_values: Iterable[Any] | pd.Series | np.ndarray,
        predictions_by_method: dict[
            str,
            Iterable[Any] | pd.Series | np.ndarray,
        ],
    ) -> dict[str, dict[str, Any]]:
        """
        Υπολογίζει μετρικές για πολλές μεθόδους πρόβλεψης.

        Parameters
        ----------
        actual_values:
            Οι πραγματικές κατηγορίες.

        predictions_by_method:
            Dictionary στο οποίο:

            - το κλειδί είναι το όνομα της μεθόδου,
            - η τιμή είναι η σειρά προβλέψεων.

        Παράδειγμα
        ----------
        {
            "model": model_predictions,
            "persistence": persistence_predictions,
            "moving_average_7": moving_average_predictions
        }

        Returns
        -------
        dict
            Μετρικές για κάθε μέθοδο.
        """

        if not isinstance(predictions_by_method, dict):
            raise TypeError(
                "Το predictions_by_method πρέπει να είναι dictionary."
            )

        if not predictions_by_method:
            raise MetricsError(
                "Δεν δόθηκαν μέθοδοι για σύγκριση."
            )

        comparison_results: dict[str, dict[str, Any]] = {}

        for method_name, predicted_values in predictions_by_method.items():
            if not isinstance(method_name, str) or not method_name.strip():
                raise MetricsError(
                    "Κάθε μέθοδος πρέπει να έχει μη κενό όνομα."
                )

            comparison_results[method_name.strip()] = (
                self.calculate_as_dict(
                    actual_values=actual_values,
                    predicted_values=predicted_values,
                )
            )

        return comparison_results

    @staticmethod
    def find_best_method(
        comparison_results: dict[str, dict[str, Any]],
        metric_name: str = "f1_score",
    ) -> dict[str, Any]:
        """
        Εντοπίζει τη μέθοδο με την υψηλότερη τιμή συγκεκριμένης μετρικής.

        Parameters
        ----------
        comparison_results:
            Τα αποτελέσματα που επιστρέφει η compare_methods.

        metric_name:
            Η μετρική βάσει της οποίας γίνεται η επιλογή.

            Επιτρεπτές τιμές:
            - accuracy
            - precision
            - recall
            - f1_score
        """

        allowed_metrics = {
            "accuracy",
            "precision",
            "recall",
            "f1_score",
        }

        if metric_name not in allowed_metrics:
            raise ValueError(
                "Η metric_name πρέπει να είναι μία από τις: "
                f"{sorted(allowed_metrics)}"
            )

        if not isinstance(comparison_results, dict):
            raise TypeError(
                "Το comparison_results πρέπει να είναι dictionary."
            )

        if not comparison_results:
            raise MetricsError(
                "Δεν υπάρχουν αποτελέσματα για σύγκριση."
            )

        best_method: str | None = None
        best_value = float("-inf")

        for method_name, method_metrics in comparison_results.items():
            if metric_name not in method_metrics:
                raise MetricsError(
                    f"Η μέθοδος '{method_name}' δεν περιέχει "
                    f"τη μετρική '{metric_name}'."
                )

            metric_value = method_metrics[metric_name]

            if not isinstance(metric_value, (int, float)):
                raise MetricsError(
                    f"Η μετρική '{metric_name}' της μεθόδου "
                    f"'{method_name}' δεν είναι αριθμητική."
                )

            if float(metric_value) > best_value:
                best_value = float(metric_value)
                best_method = method_name

        if best_method is None:
            raise MetricsError(
                "Δεν ήταν δυνατός ο εντοπισμός καλύτερης μεθόδου."
            )

        return {
            "metric": metric_name,
            "best_method": best_method,
            "best_value": round(best_value, 6),
        }

    @staticmethod
    def build_summary_table(
        comparison_results: dict[str, dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Δημιουργεί συνοπτικό DataFrame σύγκρισης των μεθόδων.

        Οι στήλες είναι:

        - method
        - accuracy
        - precision
        - recall
        - f1_score
        - sample_count
        """

        if not isinstance(comparison_results, dict):
            raise TypeError(
                "Το comparison_results πρέπει να είναι dictionary."
            )

        if not comparison_results:
            raise MetricsError(
                "Δεν υπάρχουν αποτελέσματα για δημιουργία πίνακα."
            )

        rows: list[dict[str, Any]] = []

        for method_name, metrics in comparison_results.items():
            required_metrics = {
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "sample_count",
            }

            missing_metrics = required_metrics.difference(
                metrics.keys()
            )

            if missing_metrics:
                raise MetricsError(
                    f"Λείπουν μετρικές από τη μέθοδο '{method_name}': "
                    f"{sorted(missing_metrics)}"
                )

            rows.append(
                {
                    "method": method_name,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "sample_count": metrics["sample_count"],
                }
            )

        summary_dataframe = pd.DataFrame(rows)

        summary_dataframe = summary_dataframe.sort_values(
            by=[
                "f1_score",
                "accuracy",
            ],
            ascending=False,
        )

        summary_dataframe = summary_dataframe.reset_index(
            drop=True
        )

        return summary_dataframe


def main() -> None:
    """
    Χειροκίνητη δοκιμή της υπηρεσίας μετρικών.

    Εκτελείται με:

        python -m services.metrics
    """

    print("==============================================")
    print("Δοκιμή μετρικών αξιολόγησης")
    print("==============================================")

    actual_values = [
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        1,
    ]

    model_predictions = [
        1,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        1,
    ]

    persistence_predictions = [
        1,
        1,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
    ]

    moving_average_predictions = [
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        1,
    ]

    try:
        metrics_service = ClassificationMetrics(
            decimal_places=6,
            positive_label=1,
        )

        model_metrics = metrics_service.calculate_as_dict(
            actual_values=actual_values,
            predicted_values=model_predictions,
        )

        print()
        print("Μετρικές μοντέλου:")
        print(f"Accuracy:  {model_metrics['accuracy']:.6f}")
        print(f"Precision: {model_metrics['precision']:.6f}")
        print(f"Recall:    {model_metrics['recall']:.6f}")
        print(f"F1-score:  {model_metrics['f1_score']:.6f}")

        print()
        print("Confusion Matrix:")
        print(
            np.array(
                model_metrics["confusion_matrix"]["matrix"]
            )
        )

        comparison = metrics_service.compare_methods(
            actual_values=actual_values,
            predictions_by_method={
                "model": model_predictions,
                "persistence": persistence_predictions,
                "moving_average_7": moving_average_predictions,
            },
        )

        print()
        print("Συνοπτικός πίνακας σύγκρισης:")
        print(
            metrics_service.build_summary_table(
                comparison
            ).to_string(index=False)
        )

        print()
        print("Καλύτερη μέθοδος βάσει F1-score:")
        print(
            metrics_service.find_best_method(
                comparison_results=comparison,
                metric_name="f1_score",
            )
        )

    except MetricsError as exc:
        print()
        print("Σφάλμα ClassificationMetrics:")
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