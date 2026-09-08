from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from config import (
    DATA_DIR,
    METRICS_FILE,
    MODEL_FILE,
)


class VisualizationServiceError(Exception):
    """
    Ειδική εξαίρεση για σφάλματα δημιουργίας και αποθήκευσης
    των διαγραμμάτων της πειραματικής αξιολόγησης.
    """


class VisualizationService:
    """
    Υπηρεσία δημιουργίας διαγραμμάτων για την αξιολόγηση
    του μοντέλου πρόβλεψης της κατεύθυνσης του Bitcoin.

    Δημιουργούνται τα ακόλουθα διαγράμματα:

    1. Σύγκριση μετρικών όλων των μεθόδων.
    2. Confusion Matrix του μοντέλου.
    3. Πραγματικές και προβλεπόμενες κατευθύνσεις.
    4. Πιθανότητα πρόβλεψης UP ανά χρονικό βήμα.
    5. Σημαντικότερα χαρακτηριστικά Logistic Regression.
    """

    REQUIRED_METHOD_METRICS = {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    }

    def __init__(
        self,
        metrics_file: str | Path = METRICS_FILE,
        model_file: str | Path = MODEL_FILE,
        output_directory: str | Path | None = None,
        image_dpi: int = 300,
    ) -> None:
        """
        Αρχικοποιεί την υπηρεσία οπτικοποίησης.

        Parameters
        ----------
        metrics_file:
            Το JSON report της walk-forward αξιολόγησης.

        model_file:
            Το αποθηκευμένο μοντέλο Joblib.

        output_directory:
            Ο φάκελος αποθήκευσης των εικόνων.

        image_dpi:
            Ανάλυση των παραγόμενων εικόνων.
        """

        if not isinstance(image_dpi, int) or image_dpi < 72:
            raise ValueError(
                "Το image_dpi πρέπει να είναι ακέραιος "
                "με τιμή τουλάχιστον 72."
            )

        self.metrics_file = Path(metrics_file)
        self.model_file = Path(model_file)

        self.output_directory = (
            Path(output_directory)
            if output_directory is not None
            else DATA_DIR / "figures"
        )

        self.image_dpi = image_dpi

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def load_evaluation_report(self) -> dict[str, Any]:
        """
        Φορτώνει το JSON report της πειραματικής αξιολόγησης.
        """

        if not self.metrics_file.exists():
            raise VisualizationServiceError(
                "Δεν βρέθηκε το αρχείο αξιολόγησης: "
                f"{self.metrics_file}"
            )

        if not self.metrics_file.is_file():
            raise VisualizationServiceError(
                "Η διαδρομή αξιολόγησης δεν είναι αρχείο: "
                f"{self.metrics_file}"
            )

        try:
            with self.metrics_file.open(
                mode="r",
                encoding="utf-8",
            ) as file:
                report = json.load(file)

        except json.JSONDecodeError as exc:
            raise VisualizationServiceError(
                "Το αρχείο αξιολόγησης δεν περιέχει έγκυρο JSON."
            ) from exc

        except (OSError, PermissionError) as exc:
            raise VisualizationServiceError(
                "Δεν ήταν δυνατή η ανάγνωση του αρχείου αξιολόγησης."
            ) from exc

        if not isinstance(report, dict) or not report:
            raise VisualizationServiceError(
                "Το report αξιολόγησης είναι κενό ή μη έγκυρο."
            )

        return report

    def load_model_bundle(self) -> dict[str, Any]:
        """
        Φορτώνει το αποθηκευμένο μοντέλο και τα metadata του.
        """

        if not self.model_file.exists():
            raise VisualizationServiceError(
                f"Δεν βρέθηκε το μοντέλο: {self.model_file}"
            )

        try:
            model_bundle = joblib.load(self.model_file)

        except Exception as exc:
            raise VisualizationServiceError(
                f"Αποτυχία φόρτωσης μοντέλου: {exc}"
            ) from exc

        if not isinstance(model_bundle, dict):
            raise VisualizationServiceError(
                "Το αρχείο μοντέλου δεν έχει την αναμενόμενη δομή."
            )

        return model_bundle

    @staticmethod
    def _validate_results(
        report: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """
        Ελέγχει τις μετρικές των μεθόδων αξιολόγησης.
        """

        results = report.get("results")

        if not isinstance(results, dict) or not results:
            raise VisualizationServiceError(
                "Το report δεν περιέχει αποτελέσματα μεθόδων."
            )

        for method_name, metrics in results.items():
            if not isinstance(metrics, dict):
                raise VisualizationServiceError(
                    f"Μη έγκυρες μετρικές για τη μέθοδο {method_name}."
                )

            missing_metrics = (
                VisualizationService.REQUIRED_METHOD_METRICS
                .difference(metrics.keys())
            )

            if missing_metrics:
                raise VisualizationServiceError(
                    f"Λείπουν μετρικές από τη μέθοδο "
                    f"'{method_name}': {sorted(missing_metrics)}"
                )

        return results

    def _save_figure(
        self,
        figure: plt.Figure,
        filename: str,
    ) -> Path:
        """
        Αποθηκεύει και κλείνει ένα matplotlib Figure.
        """

        output_path = self.output_directory / filename

        try:
            figure.tight_layout()

            figure.savefig(
                output_path,
                dpi=self.image_dpi,
                bbox_inches="tight",
            )

        except (OSError, PermissionError) as exc:
            raise VisualizationServiceError(
                f"Αποτυχία αποθήκευσης εικόνας: {output_path}"
            ) from exc

        finally:
            plt.close(figure)

        return output_path

    def create_metrics_comparison(
        self,
        report: dict[str, Any],
    ) -> Path:
        """
        Δημιουργεί ομαδοποιημένο ραβδόγραμμα σύγκρισης των:

        - Accuracy
        - Precision
        - Recall
        - F1-score
        """

        results = self._validate_results(report)

        method_names = list(results.keys())

        metric_names = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
        ]

        display_metric_names = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
        ]

        values = np.array(
            [
                [
                    float(results[method][metric])
                    for metric in metric_names
                ]
                for method in method_names
            ]
        )

        positions = np.arange(len(metric_names))

        bar_width = 0.8 / len(method_names)

        figure, axis = plt.subplots(
            figsize=(11, 6)
        )

        for method_index, method_name in enumerate(method_names):
            offsets = (
                positions
                - 0.4
                + bar_width / 2
                + method_index * bar_width
            )

            bars = axis.bar(
                offsets,
                values[method_index],
                width=bar_width,
                label=method_name,
            )

            axis.bar_label(
                bars,
                fmt="%.3f",
                padding=3,
                fontsize=8,
            )

        axis.set_title(
            "Σύγκριση μετρικών μοντέλου και baseline μεθόδων"
        )

        axis.set_xlabel("Μετρική")
        axis.set_ylabel("Τιμή")
        axis.set_ylim(0, 1)

        axis.set_xticks(positions)
        axis.set_xticklabels(display_metric_names)

        axis.legend()
        axis.grid(axis="y", alpha=0.3)

        return self._save_figure(
            figure=figure,
            filename="metrics_comparison.png",
        )

    def create_model_confusion_matrix(
        self,
        report: dict[str, Any],
    ) -> Path:
        """
        Δημιουργεί οπτική αναπαράσταση του Confusion Matrix
        του μοντέλου μηχανικής μάθησης.
        """

        results = self._validate_results(report)

        if "model" not in results:
            raise VisualizationServiceError(
                "Δεν βρέθηκαν αποτελέσματα για τη μέθοδο model."
            )

        confusion_data = results["model"].get(
            "confusion_matrix"
        )

        if not isinstance(confusion_data, dict):
            raise VisualizationServiceError(
                "Δεν βρέθηκε Confusion Matrix για το μοντέλο."
            )

        matrix = np.asarray(
            confusion_data.get("matrix")
        )

        if matrix.shape != (2, 2):
            raise VisualizationServiceError(
                "Το Confusion Matrix πρέπει να έχει διάσταση 2×2."
            )

        figure, axis = plt.subplots(
            figsize=(6, 5)
        )

        image = axis.imshow(matrix)

        figure.colorbar(
            image,
            ax=axis,
        )

        axis.set_title(
            "Πίνακας Σύγχυσης – Logistic Regression"
        )

        axis.set_xlabel("Προβλεπόμενη κατηγορία")
        axis.set_ylabel("Πραγματική κατηγορία")

        axis.set_xticks([0, 1])
        axis.set_yticks([0, 1])

        axis.set_xticklabels(["DOWN", "UP"])
        axis.set_yticklabels(["DOWN", "UP"])

        threshold = matrix.max() / 2

        for row_index in range(2):
            for column_index in range(2):
                axis.text(
                    column_index,
                    row_index,
                    str(int(matrix[row_index, column_index])),
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if matrix[row_index, column_index] > threshold
                        else "black"
                    ),
                    fontsize=13,
                )

        return self._save_figure(
            figure=figure,
            filename="model_confusion_matrix.png",
        )

    @staticmethod
    def _load_prediction_dataframe(
        report: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Δημιουργεί DataFrame από τις αναλυτικές walk-forward προβλέψεις.
        """

        predictions = report.get("predictions")

        if not isinstance(predictions, list) or not predictions:
            raise VisualizationServiceError(
                "Το report δεν περιέχει αναλυτικές προβλέψεις."
            )

        dataframe = pd.DataFrame(predictions)

        required_columns = {
            "date",
            "actual",
            "model_prediction",
            "model_probability_up",
        }

        missing_columns = required_columns.difference(
            dataframe.columns
        )

        if missing_columns:
            raise VisualizationServiceError(
                "Λείπουν πεδία από τις αναλυτικές προβλέψεις: "
                f"{sorted(missing_columns)}"
            )

        dataframe["date"] = pd.to_datetime(
            dataframe["date"],
            utc=True,
            errors="coerce",
        )

        numeric_columns = [
            "actual",
            "model_prediction",
            "model_probability_up",
        ]

        for column in numeric_columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

        dataframe = dataframe.dropna(
            subset=["date"] + numeric_columns
        )

        dataframe = dataframe.sort_values(
            by="date",
            ascending=True,
        )

        dataframe = dataframe.reset_index(drop=True)

        if dataframe.empty:
            raise VisualizationServiceError(
                "Δεν απέμειναν έγκυρες προβλέψεις για απεικόνιση."
            )

        return dataframe

    def create_actual_vs_predicted(
        self,
        report: dict[str, Any],
    ) -> Path:
        """
        Δημιουργεί διάγραμμα πραγματικής και προβλεπόμενης
        κατεύθυνσης ανά χρονικό βήμα.
        """

        dataframe = self._load_prediction_dataframe(report)

        figure, axis = plt.subplots(
            figsize=(13, 5)
        )

        axis.step(
            dataframe["date"],
            dataframe["actual"],
            where="post",
            label="Πραγματική κατεύθυνση",
        )

        axis.step(
            dataframe["date"],
            dataframe["model_prediction"],
            where="post",
            label="Πρόβλεψη μοντέλου",
            alpha=0.75,
        )

        axis.set_title(
            "Πραγματική και προβλεπόμενη κατεύθυνση Bitcoin"
        )

        axis.set_xlabel("Ημερομηνία")
        axis.set_ylabel("Κατεύθυνση")

        axis.set_yticks([0, 1])
        axis.set_yticklabels(["DOWN", "UP"])

        axis.legend()
        axis.grid(alpha=0.3)

        figure.autofmt_xdate()

        return self._save_figure(
            figure=figure,
            filename="actual_vs_predicted.png",
        )

    def create_probability_chart(
        self,
        report: dict[str, Any],
    ) -> Path:
        """
        Δημιουργεί διάγραμμα της πιθανότητας πρόβλεψης UP.
        """

        dataframe = self._load_prediction_dataframe(report)

        figure, axis = plt.subplots(
            figsize=(13, 5)
        )

        axis.plot(
            dataframe["date"],
            dataframe["model_probability_up"],
            label="Πιθανότητα UP",
        )

        axis.axhline(
            y=0.5,
            linestyle="--",
            label="Όριο ταξινόμησης 0.50",
        )

        axis.set_title(
            "Πιθανότητα πρόβλεψης ανοδικής κατεύθυνσης"
        )

        axis.set_xlabel("Ημερομηνία")
        axis.set_ylabel("Πιθανότητα UP")
        axis.set_ylim(0, 1)

        axis.legend()
        axis.grid(alpha=0.3)

        figure.autofmt_xdate()

        return self._save_figure(
            figure=figure,
            filename="prediction_probabilities.png",
        )

    def create_feature_importance(
        self,
        model_bundle: dict[str, Any],
        top_n: int = 15,
    ) -> Path:
        """
        Δημιουργεί διάγραμμα των σημαντικότερων χαρακτηριστικών
        με βάση τους συντελεστές της Logistic Regression.

        Θετικός συντελεστής:
            αυξάνει την τάση πρόβλεψης UP.

        Αρνητικός συντελεστής:
            αυξάνει την τάση πρόβλεψης DOWN.
        """

        if not isinstance(top_n, int) or top_n < 1:
            raise ValueError(
                "Το top_n πρέπει να είναι θετικός ακέραιος."
            )

        pipeline = model_bundle.get("pipeline")
        feature_columns = model_bundle.get("feature_columns")

        if not isinstance(pipeline, Pipeline):
            raise VisualizationServiceError(
                "Το model bundle δεν περιέχει έγκυρο Pipeline."
            )

        if not isinstance(feature_columns, list) or not feature_columns:
            raise VisualizationServiceError(
                "Δεν βρέθηκαν χαρακτηριστικά στο model bundle."
            )

        classifier = pipeline.named_steps.get("classifier")

        if classifier is None or not hasattr(classifier, "coef_"):
            raise VisualizationServiceError(
                "Ο ταξινομητής δεν παρέχει συντελεστές coef_."
            )

        coefficients = np.asarray(
            classifier.coef_
        ).reshape(-1)

        if len(coefficients) != len(feature_columns):
            raise VisualizationServiceError(
                "Το πλήθος συντελεστών δεν συμφωνεί με "
                "το πλήθος χαρακτηριστικών."
            )

        importance_dataframe = pd.DataFrame(
            {
                "feature": feature_columns,
                "coefficient": coefficients,
                "absolute_importance": np.abs(coefficients),
            }
        )

        importance_dataframe = (
            importance_dataframe
            .sort_values(
                by="absolute_importance",
                ascending=False,
            )
            .head(top_n)
            .sort_values(
                by="coefficient",
                ascending=True,
            )
        )

        figure, axis = plt.subplots(
            figsize=(10, 7)
        )

        bars = axis.barh(
            importance_dataframe["feature"],
            importance_dataframe["coefficient"],
        )

        axis.bar_label(
            bars,
            fmt="%.3f",
            padding=3,
            fontsize=8,
        )

        axis.axvline(
            x=0,
            linewidth=1,
        )

        axis.set_title(
            f"Τα {len(importance_dataframe)} σημαντικότερα "
            "χαρακτηριστικά του μοντέλου"
        )

        axis.set_xlabel(
            "Τυποποιημένος συντελεστής Logistic Regression"
        )

        axis.set_ylabel("Χαρακτηριστικό")
        axis.grid(axis="x", alpha=0.3)

        return self._save_figure(
            figure=figure,
            filename="feature_importance.png",
        )

    def create_all(self) -> dict[str, str]:
        """
        Δημιουργεί και αποθηκεύει όλα τα διαθέσιμα διαγράμματα.

        Returns
        -------
        dict
            Dictionary με τα ονόματα και τις διαδρομές των εικόνων.
        """

        report = self.load_evaluation_report()
        model_bundle = self.load_model_bundle()

        created_files = {
            "metrics_comparison": str(
                self.create_metrics_comparison(report)
            ),
            "model_confusion_matrix": str(
                self.create_model_confusion_matrix(report)
            ),
            "actual_vs_predicted": str(
                self.create_actual_vs_predicted(report)
            ),
            "prediction_probabilities": str(
                self.create_probability_chart(report)
            ),
            "feature_importance": str(
                self.create_feature_importance(
                    model_bundle=model_bundle,
                    top_n=15,
                )
            ),
        }

        return created_files


def main() -> None:
    """
    Χειροκίνητη δημιουργία όλων των διαγραμμάτων.

    Εκτελείται με:

        python -m services.visualization
    """

    print("==============================================")
    print("Δημιουργία διαγραμμάτων αξιολόγησης")
    print("==============================================")

    try:
        visualization_service = VisualizationService()

        created_files = visualization_service.create_all()

        print()
        print("Η δημιουργία των διαγραμμάτων ολοκληρώθηκε.")

        print()
        print("Αρχεία εικόνων:")

        for figure_name, figure_path in created_files.items():
            print(f"- {figure_name}: {figure_path}")

    except VisualizationServiceError as exc:
        print()
        print("Σφάλμα VisualizationService:")
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