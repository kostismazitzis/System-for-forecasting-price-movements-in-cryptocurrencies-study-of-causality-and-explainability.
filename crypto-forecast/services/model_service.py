from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    DATASET_FILE,
    MODEL_FILE,
    RANDOM_STATE,
    TARGET_COLUMN,
)


class ModelServiceError(Exception):
    """
    Ειδική εξαίρεση για σφάλματα που σχετίζονται με:

    - τη φόρτωση του dataset,
    - την επιλογή χαρακτηριστικών,
    - την εκπαίδευση του μοντέλου,
    - την αποθήκευση και φόρτωση του μοντέλου,
    - την παραγωγή προβλέψεων.
    """


class ModelService:
    """
    Υπηρεσία εκπαίδευσης και πρόβλεψης της κατεύθυνσης του Bitcoin.

    Το μοντέλο επιλύει πρόβλημα δυαδικής ταξινόμησης:

        0 = DOWN
        1 = UP

    Η κλάση πραγματοποιεί τα ακόλουθα στάδια:

    1. Φορτώνει το τελικό dataset.
    2. Εντοπίζει τις αριθμητικές στήλες χαρακτηριστικών.
    3. Εξαιρεί τις στήλες date, target και next_close.
    4. Χωρίζει τα δεδομένα χρονολογικά σε train και test.
    5. Δημιουργεί Pipeline με:
       - StandardScaler
       - LogisticRegression
    6. Εκπαιδεύει το μοντέλο.
    7. Υπολογίζει βασικές μετρικές αξιολόγησης.
    8. Αποθηκεύει το μοντέλο, τα χαρακτηριστικά και metadata.
    9. Φορτώνει το αποθηκευμένο μοντέλο.
    10. Παράγει πρόβλεψη UP/DOWN και πιθανότητες.
    """

    EXCLUDED_FEATURE_COLUMNS = {
        "date",
        "next_close",
    }

    CLASS_LABELS = {
        0: "DOWN",
        1: "UP",
    }

    def __init__(
        self,
        model_file: str | Path = MODEL_FILE,
        target_column: str = TARGET_COLUMN,
        random_state: int = RANDOM_STATE,
        test_ratio: float = 0.20,
        max_iterations: int = 2000,
    ) -> None:
        """
        Αρχικοποιεί την υπηρεσία μοντέλου.

        Parameters
        ----------
        model_file:
            Η διαδρομή στην οποία αποθηκεύεται το μοντέλο.

        target_column:
            Το όνομα της στήλης στόχου.

        random_state:
            Σταθερός αριθμός για αναπαραγωγιμότητα.

        test_ratio:
            Ποσοστό των πιο πρόσφατων δεδομένων που χρησιμοποιείται
            ως σύνολο ελέγχου.

        max_iterations:
            Μέγιστος αριθμός επαναλήψεων της Logistic Regression.
        """

        if not isinstance(target_column, str) or not target_column.strip():
            raise ValueError(
                "Το target_column πρέπει να είναι μη κενή συμβολοσειρά."
            )

        if not isinstance(random_state, int):
            raise TypeError(
                "Το random_state πρέπει να είναι ακέραιος αριθμός."
            )

        if not isinstance(test_ratio, (int, float)):
            raise TypeError(
                "Το test_ratio πρέπει να είναι αριθμητική τιμή."
            )

        if not 0.05 <= float(test_ratio) <= 0.50:
            raise ValueError(
                "Το test_ratio πρέπει να βρίσκεται μεταξύ 0.05 και 0.50."
            )

        if not isinstance(max_iterations, int) or max_iterations < 100:
            raise ValueError(
                "Το max_iterations πρέπει να είναι ακέραιος "
                "με τιμή τουλάχιστον 100."
            )

        self.model_file = Path(model_file)
        self.target_column = target_column.strip()
        self.random_state = random_state
        self.test_ratio = float(test_ratio)
        self.max_iterations = max_iterations

        self.pipeline: Pipeline | None = None
        self.feature_columns: list[str] = []
        self.training_metadata: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}

    @staticmethod
    def load_dataset(
        dataset_file: str | Path = DATASET_FILE,
    ) -> pd.DataFrame:
        """
        Φορτώνει το τελικό dataset από αρχείο CSV.

        Parameters
        ----------
        dataset_file:
            Διαδρομή του αρχείου dataset.

        Returns
        -------
        pandas.DataFrame
            Το φορτωμένο και ελεγμένο dataset.
        """

        dataset_path = Path(dataset_file)

        if not dataset_path.exists():
            raise ModelServiceError(
                f"Δεν βρέθηκε το αρχείο dataset: {dataset_path}"
            )

        if not dataset_path.is_file():
            raise ModelServiceError(
                f"Η διαδρομή δεν αντιστοιχεί σε αρχείο: {dataset_path}"
            )

        try:
            dataframe = pd.read_csv(dataset_path)

        except pd.errors.EmptyDataError as exc:
            raise ModelServiceError(
                "Το αρχείο dataset είναι κενό."
            ) from exc

        except pd.errors.ParserError as exc:
            raise ModelServiceError(
                "Το αρχείο dataset δεν είναι έγκυρο CSV."
            ) from exc

        except (OSError, PermissionError) as exc:
            raise ModelServiceError(
                f"Δεν ήταν δυνατή η ανάγνωση του dataset: {dataset_path}"
            ) from exc

        if dataframe.empty:
            raise ModelServiceError(
                "Το dataset δεν περιέχει εγγραφές."
            )

        if "date" not in dataframe.columns:
            raise ModelServiceError(
                "Το dataset δεν περιέχει τη στήλη date."
            )

        dataframe["date"] = pd.to_datetime(
            dataframe["date"],
            utc=True,
            errors="coerce",
        )

        dataframe = dataframe.dropna(
            subset=["date"]
        )

        dataframe = dataframe.sort_values(
            by="date",
            ascending=True,
        )

        dataframe = dataframe.drop_duplicates(
            subset=["date"],
            keep="last",
        )

        dataframe = dataframe.reset_index(drop=True)

        if dataframe.empty:
            raise ModelServiceError(
                "Δεν απέμειναν έγκυρες εγγραφές μετά "
                "την επεξεργασία ημερομηνιών."
            )

        return dataframe

    def validate_dataset(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Ελέγχει ότι το dataset είναι κατάλληλο για εκπαίδευση.

        Ελέγχονται:

        - η ύπαρξη της μεταβλητής στόχου,
        - το πλήθος των εγγραφών,
        - οι κατηγορίες 0 και 1,
        - οι κενές και άπειρες τιμές,
        - η χρονολογική ταξινόμηση.
        """

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "Το dataframe πρέπει να είναι pandas DataFrame."
            )

        if dataframe.empty:
            raise ModelServiceError(
                "Το dataset είναι κενό."
            )

        if self.target_column not in dataframe.columns:
            raise ModelServiceError(
                "Δεν βρέθηκε η μεταβλητή στόχου "
                f"'{self.target_column}'."
            )

        if "date" not in dataframe.columns:
            raise ModelServiceError(
                "Δεν βρέθηκε η στήλη date."
            )

        validated_dataframe = dataframe.copy()

        validated_dataframe[self.target_column] = pd.to_numeric(
            validated_dataframe[self.target_column],
            errors="coerce",
        )

        validated_dataframe = validated_dataframe.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        validated_dataframe = validated_dataframe.dropna(
            subset=[self.target_column]
        )

        validated_dataframe[self.target_column] = (
            validated_dataframe[self.target_column].astype(int)
        )

        target_values = set(
            validated_dataframe[self.target_column]
            .unique()
            .tolist()
        )

        if not target_values.issubset({0, 1}):
            raise ModelServiceError(
                "Η μεταβλητή στόχου πρέπει να περιέχει "
                "αποκλειστικά τις τιμές 0 και 1."
            )

        if len(target_values) < 2:
            raise ModelServiceError(
                "Το dataset περιέχει μόνο μία κατηγορία στόχου."
            )

        if len(validated_dataframe) < 50:
            raise ModelServiceError(
                "Το dataset πρέπει να περιέχει τουλάχιστον "
                "50 έγκυρες εγγραφές."
            )

        validated_dataframe = validated_dataframe.sort_values(
            by="date",
            ascending=True,
        )

        validated_dataframe = validated_dataframe.reset_index(
            drop=True
        )

        return validated_dataframe

    def get_feature_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> list[str]:
        """
        Επιστρέφει τις αριθμητικές στήλες που χρησιμοποιούνται
        ως χαρακτηριστικά εισόδου.

        Εξαιρούνται:

        - date,
        - target,
        - next_close.
        """

        excluded_columns = set(self.EXCLUDED_FEATURE_COLUMNS)

        excluded_columns.add(self.target_column)

        feature_columns = [
            column
            for column in dataframe.columns
            if column not in excluded_columns
            and pd.api.types.is_numeric_dtype(dataframe[column])
        ]

        if not feature_columns:
            raise ModelServiceError(
                "Δεν βρέθηκαν αριθμητικά χαρακτηριστικά."
            )

        if self.target_column in feature_columns:
            raise ModelServiceError(
                "Η μεταβλητή στόχου συμπεριλήφθηκε κατά λάθος "
                "στα χαρακτηριστικά."
            )

        if "next_close" in feature_columns:
            raise ModelServiceError(
                "Η στήλη next_close δεν επιτρέπεται να χρησιμοποιηθεί "
                "ως χαρακτηριστικό, επειδή προκαλεί data leakage."
            )

        return feature_columns

    def prepare_features_and_target(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Προετοιμάζει τον πίνακα χαρακτηριστικών X και τον στόχο y.

        Returns
        -------
        tuple
            X:
                Πίνακας χαρακτηριστικών.

            y:
                Μεταβλητή στόχου.

            cleaned_dataframe:
                Το καθαρισμένο dataset που αντιστοιχεί στα X και y.
        """

        validated_dataframe = self.validate_dataset(dataframe)

        feature_columns = self.get_feature_columns(
            validated_dataframe
        )

        working_dataframe = validated_dataframe[
            ["date"] + feature_columns + [self.target_column]
        ].copy()

        for column in feature_columns:
            working_dataframe[column] = pd.to_numeric(
                working_dataframe[column],
                errors="coerce",
            )

        working_dataframe = working_dataframe.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        working_dataframe = working_dataframe.dropna(
            subset=feature_columns + [self.target_column]
        )

        working_dataframe = working_dataframe.reset_index(
            drop=True
        )

        if len(working_dataframe) < 50:
            raise ModelServiceError(
                "Μετά τον καθαρισμό απέμειναν λιγότερες "
                "από 50 εγγραφές."
            )

        if working_dataframe[self.target_column].nunique() < 2:
            raise ModelServiceError(
                "Μετά τον καθαρισμό απέμεινε μόνο μία "
                "κατηγορία στόχου."
            )

        self.feature_columns = feature_columns

        features = working_dataframe[
            self.feature_columns
        ].astype(float)

        target = working_dataframe[
            self.target_column
        ].astype(int)

        return features, target, working_dataframe

    def temporal_train_test_split(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        dates: pd.Series,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
    ]:
        """
        Χωρίζει τα δεδομένα σε train και test με χρονολογική σειρά.

        Δεν πραγματοποιείται τυχαία ανάμειξη, επειδή τα δεδομένα
        αποτελούν χρονοσειρά.

        Οι παλαιότερες εγγραφές χρησιμοποιούνται για εκπαίδευση
        και οι πιο πρόσφατες για έλεγχο.
        """

        if len(features) != len(target) or len(features) != len(dates):
            raise ModelServiceError(
                "Τα features, target και dates πρέπει να έχουν "
                "το ίδιο πλήθος εγγραφών."
            )

        number_of_rows = len(features)

        split_index = int(
            number_of_rows * (1.0 - self.test_ratio)
        )

        if split_index <= 0 or split_index >= number_of_rows:
            raise ModelServiceError(
                "Δεν ήταν δυνατός ο χρονολογικός διαχωρισμός "
                "train/test."
            )

        train_features = features.iloc[:split_index].copy()
        test_features = features.iloc[split_index:].copy()

        train_target = target.iloc[:split_index].copy()
        test_target = target.iloc[split_index:].copy()

        train_dates = dates.iloc[:split_index].copy()
        test_dates = dates.iloc[split_index:].copy()

        if train_target.nunique() < 2:
            raise ModelServiceError(
                "Το σύνολο εκπαίδευσης περιέχει μόνο μία "
                "κατηγορία στόχου."
            )

        if train_features.empty or test_features.empty:
            raise ModelServiceError(
                "Το train ή το test σύνολο είναι κενό."
            )

        return (
            train_features,
            test_features,
            train_target,
            test_target,
            train_dates,
            test_dates,
        )

    def create_pipeline(self) -> Pipeline:
        """
        Δημιουργεί το Pipeline του μοντέλου.

        Το Pipeline περιλαμβάνει:

        1. StandardScaler
        2. LogisticRegression
        """

        pipeline = Pipeline(
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

        return pipeline

    @staticmethod
    def calculate_metrics(
        true_target: pd.Series | np.ndarray,
        predicted_target: pd.Series | np.ndarray,
    ) -> dict[str, Any]:
        """
        Υπολογίζει τις βασικές μετρικές ταξινόμησης.

        Οι μετρικές είναι:

        - Accuracy
        - Precision
        - Recall
        - F1-score
        - Confusion Matrix
        """

        true_array = np.asarray(true_target)
        predicted_array = np.asarray(predicted_target)

        if len(true_array) == 0:
            raise ModelServiceError(
                "Δεν υπάρχουν πραγματικές τιμές για αξιολόγηση."
            )

        if len(true_array) != len(predicted_array):
            raise ModelServiceError(
                "Οι πραγματικές και προβλεπόμενες τιμές πρέπει "
                "να έχουν ίδιο μήκος."
            )

        matrix = confusion_matrix(
            true_array,
            predicted_array,
            labels=[0, 1],
        )

        return {
            "accuracy": round(
                float(
                    accuracy_score(
                        true_array,
                        predicted_array,
                    )
                ),
                6,
            ),
            "precision": round(
                float(
                    precision_score(
                        true_array,
                        predicted_array,
                        zero_division=0,
                    )
                ),
                6,
            ),
            "recall": round(
                float(
                    recall_score(
                        true_array,
                        predicted_array,
                        zero_division=0,
                    )
                ),
                6,
            ),
            "f1_score": round(
                float(
                    f1_score(
                        true_array,
                        predicted_array,
                        zero_division=0,
                    )
                ),
                6,
            ),
            "confusion_matrix": {
                "true_down_predicted_down": int(matrix[0, 0]),
                "true_down_predicted_up": int(matrix[0, 1]),
                "true_up_predicted_down": int(matrix[1, 0]),
                "true_up_predicted_up": int(matrix[1, 1]),
                "matrix": matrix.astype(int).tolist(),
            },
        }

    def train(
        self,
        dataframe: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Εκπαιδεύει το μοντέλο και το αξιολογεί σε χρονολογικό
        σύνολο ελέγχου.

        Μετά την αρχική αξιολόγηση, το Pipeline εκπαιδεύεται ξανά
        σε ολόκληρο το διαθέσιμο dataset, ώστε το αποθηκευμένο
        μοντέλο να αξιοποιεί όλες τις διαθέσιμες εγγραφές.

        Returns
        -------
        dict
            Πληροφορίες εκπαίδευσης και μετρικές.
        """

        features, target, cleaned_dataframe = (
            self.prepare_features_and_target(dataframe)
        )

        (
            train_features,
            test_features,
            train_target,
            test_target,
            train_dates,
            test_dates,
        ) = self.temporal_train_test_split(
            features=features,
            target=target,
            dates=cleaned_dataframe["date"],
        )

        evaluation_pipeline = self.create_pipeline()

        try:
            evaluation_pipeline.fit(
                train_features,
                train_target,
            )

            predicted_target = evaluation_pipeline.predict(
                test_features
            )

        except Exception as exc:
            raise ModelServiceError(
                f"Αποτυχία εκπαίδευσης του μοντέλου: {exc}"
            ) from exc

        self.metrics = self.calculate_metrics(
            true_target=test_target,
            predicted_target=predicted_target,
        )

        # Τελική εκπαίδευση σε ολόκληρο το dataset.
        final_pipeline = self.create_pipeline()

        try:
            final_pipeline.fit(
                features,
                target,
            )

        except Exception as exc:
            raise ModelServiceError(
                "Η τελική εκπαίδευση σε ολόκληρο το dataset "
                f"απέτυχε: {exc}"
            ) from exc

        self.pipeline = final_pipeline

        self.training_metadata = {
            "trained_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "model_type": "LogisticRegression",
            "preprocessing": "StandardScaler",
            "target_column": self.target_column,
            "class_labels": self.CLASS_LABELS,
            "random_state": self.random_state,
            "test_ratio": self.test_ratio,
            "total_samples": int(len(features)),
            "training_samples": int(len(train_features)),
            "test_samples": int(len(test_features)),
            "feature_count": int(len(self.feature_columns)),
            "dataset_start_date": (
                cleaned_dataframe["date"].min().isoformat()
            ),
            "dataset_end_date": (
                cleaned_dataframe["date"].max().isoformat()
            ),
            "training_start_date": (
                train_dates.min().isoformat()
            ),
            "training_end_date": (
                train_dates.max().isoformat()
            ),
            "test_start_date": (
                test_dates.min().isoformat()
            ),
            "test_end_date": (
                test_dates.max().isoformat()
            ),
            "target_distribution": {
                "down": int((target == 0).sum()),
                "up": int((target == 1).sum()),
            },
        }

        return {
            "status": "trained",
            "metadata": self.training_metadata,
            "metrics": self.metrics,
            "feature_columns": self.feature_columns,
        }

    def train_from_file(
        self,
        dataset_file: str | Path = DATASET_FILE,
    ) -> dict[str, Any]:
        """
        Φορτώνει το dataset από αρχείο και εκπαιδεύει το μοντέλο.
        """

        dataframe = self.load_dataset(
            dataset_file=dataset_file
        )

        return self.train(
            dataframe=dataframe
        )

    def save_model(
        self,
        output_file: str | Path | None = None,
    ) -> Path:
        """
        Αποθηκεύει το εκπαιδευμένο μοντέλο και τις πληροφορίες του.

        Αποθηκεύονται μαζί:

        - το Pipeline,
        - οι στήλες χαρακτηριστικών,
        - το όνομα του target,
        - οι ετικέτες των κατηγοριών,
        - οι μετρικές,
        - τα metadata εκπαίδευσης.
        """

        if self.pipeline is None:
            raise ModelServiceError(
                "Δεν υπάρχει εκπαιδευμένο μοντέλο για αποθήκευση."
            )

        if not self.feature_columns:
            raise ModelServiceError(
                "Δεν υπάρχουν αποθηκευμένες στήλες χαρακτηριστικών."
            )

        model_path = (
            Path(output_file)
            if output_file is not None
            else self.model_file
        )

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_bundle = {
            "pipeline": self.pipeline,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "class_labels": self.CLASS_LABELS,
            "metrics": self.metrics,
            "training_metadata": self.training_metadata,
        }

        try:
            joblib.dump(
                model_bundle,
                model_path,
                compress=3,
            )

        except (OSError, PermissionError, ValueError) as exc:
            raise ModelServiceError(
                f"Αποτυχία αποθήκευσης μοντέλου: {model_path}"
            ) from exc

        return model_path

    def load_model(
        self,
        model_file: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Φορτώνει το αποθηκευμένο μοντέλο και τα metadata.

        Returns
        -------
        dict
            Το bundle που είχε αποθηκευτεί με joblib.
        """

        model_path = (
            Path(model_file)
            if model_file is not None
            else self.model_file
        )

        if not model_path.exists():
            raise ModelServiceError(
                f"Δεν βρέθηκε αποθηκευμένο μοντέλο: {model_path}"
            )

        if not model_path.is_file():
            raise ModelServiceError(
                f"Η διαδρομή δεν είναι αρχείο: {model_path}"
            )

        try:
            model_bundle = joblib.load(model_path)

        except Exception as exc:
            raise ModelServiceError(
                f"Αποτυχία φόρτωσης μοντέλου: {exc}"
            ) from exc

        if not isinstance(model_bundle, dict):
            raise ModelServiceError(
                "Το αποθηκευμένο μοντέλο δεν έχει την "
                "αναμενόμενη δομή."
            )

        required_keys = {
            "pipeline",
            "feature_columns",
            "target_column",
            "class_labels",
        }

        missing_keys = required_keys.difference(
            model_bundle.keys()
        )

        if missing_keys:
            raise ModelServiceError(
                "Λείπουν στοιχεία από το αποθηκευμένο μοντέλο: "
                f"{sorted(missing_keys)}"
            )

        loaded_pipeline = model_bundle["pipeline"]
        loaded_features = model_bundle["feature_columns"]

        if not isinstance(loaded_pipeline, Pipeline):
            raise ModelServiceError(
                "Το αποθηκευμένο αντικείμενο δεν περιέχει "
                "έγκυρο sklearn Pipeline."
            )

        if not isinstance(loaded_features, list) or not loaded_features:
            raise ModelServiceError(
                "Το αποθηκευμένο μοντέλο δεν περιέχει "
                "έγκυρες στήλες χαρακτηριστικών."
            )

        self.pipeline = loaded_pipeline
        self.feature_columns = loaded_features
        self.target_column = model_bundle["target_column"]
        self.metrics = model_bundle.get(
            "metrics",
            {},
        )
        self.training_metadata = model_bundle.get(
            "training_metadata",
            {},
        )

        return model_bundle

    def _ensure_model_loaded(self) -> None:
        """
        Φορτώνει αυτόματα το μοντέλο αν δεν βρίσκεται ήδη στη μνήμη.
        """

        if self.pipeline is None:
            self.load_model()

    def prepare_prediction_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Προετοιμάζει δεδομένα για πρόβλεψη.

        Οι στήλες εισόδου πρέπει να είναι ίδιες και στην ίδια σειρά
        με αυτές που χρησιμοποιήθηκαν κατά την εκπαίδευση.
        """

        self._ensure_model_loaded()

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "Το dataframe πρέπει να είναι pandas DataFrame."
            )

        if dataframe.empty:
            raise ModelServiceError(
                "Δεν υπάρχουν δεδομένα για πρόβλεψη."
            )

        missing_features = [
            column
            for column in self.feature_columns
            if column not in dataframe.columns
        ]

        if missing_features:
            raise ModelServiceError(
                "Λείπουν χαρακτηριστικά που απαιτεί το μοντέλο: "
                f"{missing_features}"
            )

        prediction_features = dataframe[
            self.feature_columns
        ].copy()

        for column in self.feature_columns:
            prediction_features[column] = pd.to_numeric(
                prediction_features[column],
                errors="coerce",
            )

        prediction_features = prediction_features.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        if prediction_features.isna().any().any():
            columns_with_missing_values = (
                prediction_features.columns[
                    prediction_features.isna().any()
                ].tolist()
            )

            raise ModelServiceError(
                "Υπάρχουν κενές ή μη έγκυρες τιμές στα "
                "χαρακτηριστικά: "
                f"{columns_with_missing_values}"
            )

        return prediction_features.astype(float)

    def predict(
        self,
        dataframe: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """
        Παράγει προβλέψεις για μία ή περισσότερες εγγραφές.

        Returns
        -------
        list[dict]
            Για κάθε εγγραφή επιστρέφονται:

            - prediction
            - direction
            - probability_down
            - probability_up
        """

        self._ensure_model_loaded()

        prediction_features = self.prepare_prediction_features(
            dataframe=dataframe
        )

        if self.pipeline is None:
            raise ModelServiceError(
                "Το μοντέλο δεν είναι διαθέσιμο."
            )

        try:
            predictions = self.pipeline.predict(
                prediction_features
            )

            probabilities = self.pipeline.predict_proba(
                prediction_features
            )

        except Exception as exc:
            raise ModelServiceError(
                f"Αποτυχία παραγωγής πρόβλεψης: {exc}"
            ) from exc

        classifier = self.pipeline.named_steps["classifier"]

        class_order = [
            int(class_value)
            for class_value in classifier.classes_
        ]

        try:
            down_index = class_order.index(0)
            up_index = class_order.index(1)

        except ValueError as exc:
            raise ModelServiceError(
                "Το μοντέλο δεν περιέχει τις αναμενόμενες "
                "κατηγορίες 0 και 1."
            ) from exc

        results: list[dict[str, Any]] = []

        for row_index, prediction in enumerate(predictions):
            prediction_value = int(prediction)

            result: dict[str, Any] = {
                "prediction": prediction_value,
                "direction": self.CLASS_LABELS[prediction_value],
                "probability_down": round(
                    float(probabilities[row_index][down_index]),
                    6,
                ),
                "probability_up": round(
                    float(probabilities[row_index][up_index]),
                    6,
                ),
            }

            if "date" in dataframe.columns:
                original_date = pd.to_datetime(
                    dataframe.iloc[row_index]["date"],
                    utc=True,
                    errors="coerce",
                )

                if not pd.isna(original_date):
                    result["date"] = original_date.isoformat()

            results.append(result)

        return results

    def predict_latest(
        self,
        dataframe: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Παράγει πρόβλεψη για την τελευταία χρονολογικά εγγραφή.
        """

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "Το dataframe πρέπει να είναι pandas DataFrame."
            )

        if dataframe.empty:
            raise ModelServiceError(
                "Δεν υπάρχουν δεδομένα για πρόβλεψη."
            )

        working_dataframe = dataframe.copy()

        if "date" in working_dataframe.columns:
            working_dataframe["date"] = pd.to_datetime(
                working_dataframe["date"],
                utc=True,
                errors="coerce",
            )

            working_dataframe = working_dataframe.dropna(
                subset=["date"]
            )

            working_dataframe = working_dataframe.sort_values(
                by="date",
                ascending=True,
            )

        if working_dataframe.empty:
            raise ModelServiceError(
                "Δεν απέμεινε έγκυρη εγγραφή για πρόβλεψη."
            )

        latest_row = working_dataframe.tail(1).copy()

        results = self.predict(
            dataframe=latest_row
        )

        return results[0]

    def train_and_save(
        self,
        dataset_file: str | Path = DATASET_FILE,
        model_file: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Φορτώνει το dataset, εκπαιδεύει και αποθηκεύει το μοντέλο.
        """

        training_result = self.train_from_file(
            dataset_file=dataset_file
        )

        saved_model_path = self.save_model(
            output_file=model_file
        )

        return {
            **training_result,
            "model_file": str(saved_model_path),
        }

    def predict_latest_from_file(
        self,
        dataset_file: str | Path = DATASET_FILE,
        model_file: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Φορτώνει το μοντέλο και προβλέπει με βάση την τελευταία
        εγγραφή του αποθηκευμένου dataset.
        """

        self.load_model(
            model_file=model_file
        )

        dataframe = self.load_dataset(
            dataset_file=dataset_file
        )

        return self.predict_latest(
            dataframe=dataframe
        )


def main() -> None:
    """
    Χειροκίνητη δοκιμή του ModelService.

    Εκτελείται με:

        python -m services.model_service
    """

    print("==============================================")
    print("Έναρξη εκπαίδευσης μοντέλου")
    print("==============================================")

    try:
        service = ModelService()

        training_result = service.train_and_save()

        print()
        print("Η εκπαίδευση ολοκληρώθηκε επιτυχώς.")
        print(f"Μοντέλο: {training_result['model_file']}")

        metadata = training_result["metadata"]
        metrics = training_result["metrics"]

        print()
        print("Στοιχεία dataset:")
        print(f"Συνολικές εγγραφές: {metadata['total_samples']}")
        print(f"Εγγραφές εκπαίδευσης: {metadata['training_samples']}")
        print(f"Εγγραφές ελέγχου: {metadata['test_samples']}")
        print(f"Πλήθος χαρακτηριστικών: {metadata['feature_count']}")

        print()
        print("Κατανομή στόχου:")
        print(
            f"DOWN: "
            f"{metadata['target_distribution']['down']}"
        )
        print(
            f"UP: "
            f"{metadata['target_distribution']['up']}"
        )

        print()
        print("Μετρικές χρονολογικού συνόλου ελέγχου:")
        print(f"Accuracy:  {metrics['accuracy']:.6f}")
        print(f"Precision: {metrics['precision']:.6f}")
        print(f"Recall:    {metrics['recall']:.6f}")
        print(f"F1-score:  {metrics['f1_score']:.6f}")

        print()
        print("Confusion Matrix:")
        print(
            np.array(
                metrics["confusion_matrix"]["matrix"]
            )
        )

        prediction = service.predict_latest_from_file()

        print()
        print("Πρόβλεψη τελευταίας εγγραφής:")
        print(f"Ημερομηνία: {prediction.get('date', 'N/A')}")
        print(f"Κατεύθυνση: {prediction['direction']}")
        print(
            f"Πιθανότητα DOWN: "
            f"{prediction['probability_down']:.6f}"
        )
        print(
            f"Πιθανότητα UP: "
            f"{prediction['probability_up']:.6f}"
        )

    except ModelServiceError as exc:
        print()
        print("Σφάλμα ModelService:")
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