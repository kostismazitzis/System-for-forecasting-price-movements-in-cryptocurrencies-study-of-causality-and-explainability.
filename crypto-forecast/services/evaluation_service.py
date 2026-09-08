from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    DATASET_FILE,
    DEFAULT_MIN_TRAIN_SIZE,
    METRICS_FILE,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from services.baselines import BaselineModels
from services.metrics import ClassificationMetrics, MetricsError
from services.model_service import ModelService


class EvaluationServiceError(Exception):
    """
    Ειδική εξαίρεση για σφάλματα που σχετίζονται με:

    - τη φόρτωση του dataset,
    - το walk-forward validation,
    - την εκπαίδευση του μοντέλου,
    - τις baseline προβλέψεις,
    - τον υπολογισμό μετρικών,
    - την αποθήκευση της αναφοράς.
    """


class EvaluationService:
    """
    Υπηρεσία πειραματικής αξιολόγησης του συστήματος.

    Η αξιολόγηση πραγματοποιείται με Walk-Forward Validation.

    Σε κάθε χρονικό βήμα:

    1. Χρησιμοποιούνται μόνο οι προηγούμενες εγγραφές για εκπαίδευση.
    2. Εκπαιδεύεται νέο μοντέλο Logistic Regression.
    3. Προβλέπεται η αμέσως επόμενη εγγραφή.
    4. Παράγονται προβλέψεις από δύο baseline μεθόδους.
    5. Αποθηκεύονται η πραγματική και οι προβλεπόμενες κατηγορίες.

    Οι μέθοδοι που συγκρίνονται είναι:

    - Machine Learning Model
    - Persistence Baseline
    - Moving Average Baseline

    Η κατηγορία στόχου είναι:

        0 = DOWN
        1 = UP
    """

    CLASS_LABELS = {
        0: "DOWN",
        1: "UP",
    }

    def __init__(
        self,
        min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
        moving_average_window: int = 7,
        random_state: int = RANDOM_STATE,
        max_iterations: int = 2000,
        decimal_places: int = 6,
        target_column: str = TARGET_COLUMN,
    ) -> None:
        """
        Αρχικοποιεί την υπηρεσία αξιολόγησης.

        Parameters
        ----------
        min_train_size:
            Ελάχιστο πλήθος εγγραφών για την πρώτη εκπαίδευση.

        moving_average_window:
            Παράθυρο του Moving Average baseline.

        random_state:
            Σταθερά αναπαραγωγιμότητας.

        max_iterations:
            Μέγιστες επαναλήψεις Logistic Regression.

        decimal_places:
            Πλήθος δεκαδικών ψηφίων στις μετρικές.

        target_column:
            Όνομα της μεταβλητής στόχου.
        """

        if not isinstance(min_train_size, int):
            raise TypeError(
                "Το min_train_size πρέπει να είναι ακέραιος."
            )

        if min_train_size < 20:
            raise ValueError(
                "Το min_train_size πρέπει να είναι τουλάχιστον 20."
            )

        if not isinstance(moving_average_window, int):
            raise TypeError(
                "Το moving_average_window πρέπει να είναι ακέραιος."
            )

        if moving_average_window < 2:
            raise ValueError(
                "Το moving_average_window πρέπει να είναι τουλάχιστον 2."
            )

        if not isinstance(random_state, int):
            raise TypeError(
                "Το random_state πρέπει να είναι ακέραιος."
            )

        if not isinstance(max_iterations, int) or max_iterations < 100:
            raise ValueError(
                "Το max_iterations πρέπει να είναι τουλάχιστον 100."
            )

        if not isinstance(target_column, str) or not target_column.strip():
            raise ValueError(
                "Το target_column δεν μπορεί να είναι κενό."
            )

        self.min_train_size = min_train_size
        self.moving_average_window = moving_average_window
        self.random_state = random_state
        self.max_iterations = max_iterations
        self.target_column = target_column.strip()

        self.metrics_service = ClassificationMetrics(
            decimal_places=decimal_places,
            positive_label=1,
        )

        self.baseline_service = BaselineModels(
            moving_average_window=moving_average_window,
        )

        self.model_service = ModelService(
            target_column=self.target_column,
            random_state=random_state,
            max_iterations=max_iterations,
        )

    @staticmethod
    def load_dataset(
        dataset_file: str | Path = DATASET_FILE,
    ) -> pd.DataFrame:
        """
        Φορτώνει το τελικό dataset από CSV.
        """

        try:
            dataframe = ModelService.load_dataset(
                dataset_file=dataset_file
            )
        except Exception as exc:
            raise EvaluationServiceError(
                f"Αποτυχία φόρτωσης dataset: {exc}"
            ) from exc

        return dataframe

    def validate_dataset(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Ελέγχει ότι το dataset είναι κατάλληλο για walk-forward.
        """

        try:
            validated_dataframe = (
                self.model_service.validate_dataset(dataframe)
            )
        except Exception as exc:
            raise EvaluationServiceError(
                f"Το dataset δεν είναι έγκυρο: {exc}"
            ) from exc

        minimum_required_rows = self.min_train_size + 1

        if len(validated_dataframe) < minimum_required_rows:
            raise EvaluationServiceError(
                "Δεν υπάρχουν αρκετές εγγραφές για walk-forward "
                f"validation. Απαιτούνται τουλάχιστον "
                f"{minimum_required_rows}, αλλά βρέθηκαν "
                f"{len(validated_dataframe)}."
            )

        if "close" not in validated_dataframe.columns:
            raise EvaluationServiceError(
                "Το dataset δεν περιέχει τη στήλη close."
            )

        return validated_dataframe

    def prepare_dataset(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Καθαρίζει το dataset και εντοπίζει τα χαρακτηριστικά.
        """

        validated_dataframe = self.validate_dataset(dataframe)

        try:
            feature_columns = self.model_service.get_feature_columns(
                validated_dataframe
            )
        except Exception as exc:
            raise EvaluationServiceError(
                f"Αποτυχία επιλογής χαρακτηριστικών: {exc}"
            ) from exc

        required_columns = (
            ["date", "close"]
            + feature_columns
            + [self.target_column]
        )

        required_columns = list(dict.fromkeys(required_columns))

        working_dataframe = validated_dataframe[
            required_columns
        ].copy()

        for column in feature_columns:
            working_dataframe[column] = pd.to_numeric(
                working_dataframe[column],
                errors="coerce",
            )

        working_dataframe["close"] = pd.to_numeric(
            working_dataframe["close"],
            errors="coerce",
        )

        working_dataframe[self.target_column] = pd.to_numeric(
            working_dataframe[self.target_column],
            errors="coerce",
        )

        working_dataframe = working_dataframe.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        working_dataframe = working_dataframe.dropna(
            subset=feature_columns
            + ["close", self.target_column, "date"]
        )

        working_dataframe[self.target_column] = (
            working_dataframe[self.target_column].astype(int)
        )

        working_dataframe = working_dataframe.sort_values(
            by="date",
            ascending=True,
        )

        working_dataframe = working_dataframe.reset_index(drop=True)

        if len(working_dataframe) < self.min_train_size + 1:
            raise EvaluationServiceError(
                "Μετά τον καθαρισμό δεν υπάρχουν αρκετές εγγραφές "
                "για walk-forward validation."
            )

        return working_dataframe, feature_columns

    def create_pipeline(self) -> Pipeline:
        """
        Δημιουργεί νέο Pipeline για κάθε walk-forward βήμα.
        """

        return Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        penalty="l2",
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=self.random_state,
                        max_iter=self.max_iterations,
                    ),
                ),
            ]
        )

    def _calculate_persistence_prediction(
        self,
        dataframe: pd.DataFrame,
        prediction_index: int,
    ) -> int:
        """
        Υπολογίζει Persistence prediction για συγκεκριμένο βήμα.

        Για την πρόβλεψη της κατεύθυνσης στη γραμμή t χρησιμοποιείται
        η πιο πρόσφατη παρατηρούμενη κίνηση:

            close(t-1) έναντι close(t-2)
        """

        if prediction_index < 2:
            raise EvaluationServiceError(
                "Δεν υπάρχουν αρκετές προηγούμενες τιμές "
                "για Persistence prediction."
            )

        previous_close = float(
            dataframe.iloc[prediction_index - 2]["close"]
        )

        current_close = float(
            dataframe.iloc[prediction_index - 1]["close"]
        )

        prediction = (
            self.baseline_service.persistence_predict_one(
                previous_close=previous_close,
                current_close=current_close,
            )
        )

        return prediction.prediction

    def _calculate_moving_average_prediction(
        self,
        dataframe: pd.DataFrame,
        prediction_index: int,
    ) -> int:
        """
        Υπολογίζει Moving Average prediction για συγκεκριμένο βήμα.

        Χρησιμοποιούνται αποκλειστικά εγγραφές πριν από τη γραμμή
        που πρόκειται να προβλεφθεί.
        """

        available_history = dataframe.iloc[:prediction_index]

        if len(available_history) < self.moving_average_window:
            raise EvaluationServiceError(
                "Δεν υπάρχουν αρκετές προηγούμενες εγγραφές "
                "για Moving Average prediction."
            )

        current_close = float(
            available_history.iloc[-1]["close"]
        )

        moving_average = float(
            available_history["close"]
            .tail(self.moving_average_window)
            .mean()
        )

        prediction = (
            self.baseline_service.moving_average_predict_one(
                current_close=current_close,
                moving_average=moving_average,
            )
        )

        return prediction.prediction

    def walk_forward_validate(
        self,
        dataframe: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Εκτελεί expanding-window Walk-Forward Validation.

        Σε κάθε βήμα:

        - train = εγγραφές 0 έως t-1,
        - test = εγγραφή t,
        - το μοντέλο επανεκπαιδεύεται,
        - παράγεται μία πρόβλεψη.
        """

        working_dataframe, feature_columns = self.prepare_dataset(
            dataframe
        )

        actual_values: list[int] = []
        model_predictions: list[int] = []
        persistence_predictions: list[int] = []
        moving_average_predictions: list[int] = []
        prediction_records: list[dict[str, Any]] = []

        skipped_steps = 0

        for prediction_index in range(
            self.min_train_size,
            len(working_dataframe),
        ):
            training_dataframe = working_dataframe.iloc[
                :prediction_index
            ].copy()

            test_dataframe = working_dataframe.iloc[
                prediction_index:prediction_index + 1
            ].copy()

            training_target = training_dataframe[
                self.target_column
            ]

            if training_target.nunique() < 2:
                skipped_steps += 1
                continue

            train_features = training_dataframe[
                feature_columns
            ].astype(float)

            train_target = training_target.astype(int)

            test_features = test_dataframe[
                feature_columns
            ].astype(float)

            actual_value = int(
                test_dataframe.iloc[0][self.target_column]
            )

            pipeline = self.create_pipeline()

            try:
                pipeline.fit(
                    train_features,
                    train_target,
                )

                model_prediction = int(
                    pipeline.predict(test_features)[0]
                )

                model_probabilities = (
                    pipeline.predict_proba(test_features)[0]
                )

            except Exception as exc:
                raise EvaluationServiceError(
                    "Αποτυχία μοντέλου στο walk-forward βήμα "
                    f"{prediction_index}: {exc}"
                ) from exc

            classifier = pipeline.named_steps["classifier"]

            class_order = [
                int(value)
                for value in classifier.classes_
            ]

            try:
                down_index = class_order.index(0)
                up_index = class_order.index(1)
            except ValueError as exc:
                raise EvaluationServiceError(
                    "Το μοντέλο δεν περιέχει και τις δύο "
                    "κατηγορίες 0 και 1."
                ) from exc

            persistence_prediction = (
                self._calculate_persistence_prediction(
                    dataframe=working_dataframe,
                    prediction_index=prediction_index,
                )
            )

            moving_average_prediction = (
                self._calculate_moving_average_prediction(
                    dataframe=working_dataframe,
                    prediction_index=prediction_index,
                )
            )

            actual_values.append(actual_value)
            model_predictions.append(model_prediction)
            persistence_predictions.append(
                persistence_prediction
            )
            moving_average_predictions.append(
                moving_average_prediction
            )

            prediction_date = pd.to_datetime(
                test_dataframe.iloc[0]["date"],
                utc=True,
            )

            prediction_records.append(
                {
                    "step": len(prediction_records) + 1,
                    "dataset_index": prediction_index,
                    "date": prediction_date.isoformat(),
                    "training_samples": int(
                        len(training_dataframe)
                    ),
                    "actual": actual_value,
                    "actual_direction": self.CLASS_LABELS[
                        actual_value
                    ],
                    "model_prediction": model_prediction,
                    "model_direction": self.CLASS_LABELS[
                        model_prediction
                    ],
                    "model_probability_down": round(
                        float(
                            model_probabilities[down_index]
                        ),
                        6,
                    ),
                    "model_probability_up": round(
                        float(
                            model_probabilities[up_index]
                        ),
                        6,
                    ),
                    "persistence_prediction": (
                        persistence_prediction
                    ),
                    "persistence_direction": self.CLASS_LABELS[
                        persistence_prediction
                    ],
                    "moving_average_prediction": (
                        moving_average_prediction
                    ),
                    "moving_average_direction": self.CLASS_LABELS[
                        moving_average_prediction
                    ],
                }
            )

        if not actual_values:
            raise EvaluationServiceError(
                "Το walk-forward validation δεν παρήγαγε προβλέψεις."
            )

        try:
            comparison_results = (
                self.metrics_service.compare_methods(
                    actual_values=actual_values,
                    predictions_by_method={
                        "model": model_predictions,
                        "persistence": persistence_predictions,
                        (
                            f"moving_average_"
                            f"{self.moving_average_window}"
                        ): moving_average_predictions,
                    },
                )
            )

            best_by_accuracy = (
                self.metrics_service.find_best_method(
                    comparison_results=comparison_results,
                    metric_name="accuracy",
                )
            )

            best_by_f1 = (
                self.metrics_service.find_best_method(
                    comparison_results=comparison_results,
                    metric_name="f1_score",
                )
            )

            summary_table = (
                self.metrics_service.build_summary_table(
                    comparison_results
                )
            )

        except MetricsError as exc:
            raise EvaluationServiceError(
                f"Αποτυχία υπολογισμού μετρικών: {exc}"
            ) from exc

        report = {
            "status": "completed",
            "evaluation_method": (
                "expanding-window walk-forward validation"
            ),
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "configuration": {
                "model": "LogisticRegression",
                "preprocessing": "StandardScaler",
                "min_train_size": self.min_train_size,
                "moving_average_window": (
                    self.moving_average_window
                ),
                "target_column": self.target_column,
                "target_definition": {
                    "0": "DOWN",
                    "1": "UP",
                },
                "random_state": self.random_state,
                "feature_count": len(feature_columns),
                "feature_columns": feature_columns,
            },
            "dataset": {
                "total_samples": int(
                    len(working_dataframe)
                ),
                "evaluation_steps": int(
                    len(actual_values)
                ),
                "skipped_steps": int(skipped_steps),
                "start_date": (
                    working_dataframe["date"]
                    .min()
                    .isoformat()
                ),
                "end_date": (
                    working_dataframe["date"]
                    .max()
                    .isoformat()
                ),
                "first_evaluation_date": (
                    prediction_records[0]["date"]
                ),
                "last_evaluation_date": (
                    prediction_records[-1]["date"]
                ),
            },
            "results": comparison_results,
            "best_method_by_accuracy": best_by_accuracy,
            "best_method_by_f1_score": best_by_f1,
            "summary": summary_table.to_dict(
                orient="records"
            ),
            "predictions": prediction_records,
        }

        return report

    def evaluate_from_file(
        self,
        dataset_file: str | Path = DATASET_FILE,
    ) -> dict[str, Any]:
        """
        Φορτώνει dataset και εκτελεί walk-forward validation.
        """

        dataframe = self.load_dataset(
            dataset_file=dataset_file
        )

        return self.walk_forward_validate(
            dataframe=dataframe
        )

    @staticmethod
    def save_report(
        report: dict[str, Any],
        output_file: str | Path = METRICS_FILE,
    ) -> Path:
        """
        Αποθηκεύει την αναφορά αξιολόγησης σε JSON.
        """

        if not isinstance(report, dict):
            raise TypeError(
                "Το report πρέπει να είναι dictionary."
            )

        if not report:
            raise EvaluationServiceError(
                "Δεν είναι δυνατή η αποθήκευση κενού report."
            )

        output_path = Path(output_file)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            with output_path.open(
                mode="w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    report,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

        except (OSError, PermissionError, TypeError) as exc:
            raise EvaluationServiceError(
                f"Αποτυχία αποθήκευσης report: {output_path}"
            ) from exc

        return output_path

    def evaluate_and_save(
        self,
        dataset_file: str | Path = DATASET_FILE,
        output_file: str | Path = METRICS_FILE,
    ) -> dict[str, Any]:
        """
        Εκτελεί αξιολόγηση και αποθηκεύει το JSON report.
        """

        report = self.evaluate_from_file(
            dataset_file=dataset_file
        )

        saved_path = self.save_report(
            report=report,
            output_file=output_file,
        )

        report["report_file"] = str(saved_path)

        return report


def main() -> None:
    """
    Χειροκίνητη δοκιμή της EvaluationService.

    Εκτελείται με:

        python -m services.evaluation_service
    """

    print("==============================================")
    print("Έναρξη Walk-Forward Validation")
    print("==============================================")

    try:
        service = EvaluationService()

        report = service.evaluate_and_save()

        print()
        print("Η αξιολόγηση ολοκληρώθηκε επιτυχώς.")
        print(
            f"Συνολικές εγγραφές: "
            f"{report['dataset']['total_samples']}"
        )
        print(
            f"Βήματα αξιολόγησης: "
            f"{report['dataset']['evaluation_steps']}"
        )
        print(
            f"Αρχείο αναφοράς: "
            f"{report['report_file']}"
        )

        print()
        print("Αποτελέσματα:")

        for method_name, metrics in report["results"].items():
            print()
            print(f"Μέθοδος: {method_name}")
            print(
                f"Accuracy:  "
                f"{metrics['accuracy']:.6f}"
            )
            print(
                f"Precision: "
                f"{metrics['precision']:.6f}"
            )
            print(
                f"Recall:    "
                f"{metrics['recall']:.6f}"
            )
            print(
                f"F1-score:  "
                f"{metrics['f1_score']:.6f}"
            )
            print("Confusion Matrix:")
            print(
                np.array(
                    metrics[
                        "confusion_matrix"
                    ]["matrix"]
                )
            )

        print()
        print("Καλύτερη μέθοδος βάσει Accuracy:")
        print(
            report[
                "best_method_by_accuracy"
            ]
        )

        print()
        print("Καλύτερη μέθοδος βάσει F1-score:")
        print(
            report[
                "best_method_by_f1_score"
            ]
        )

    except EvaluationServiceError as exc:
        print()
        print("Σφάλμα EvaluationService:")
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