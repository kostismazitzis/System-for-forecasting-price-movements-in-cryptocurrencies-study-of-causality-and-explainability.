import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

from feature_engineering import get_feature_columns


def evaluate_model(
        dataset_file="data/bitcoin_dataset.csv",
        model_file="saved_models/bitcoin_direction_model.joblib",
        target_column="target",
        min_train_size=60,
        moving_average_window=7,
        save=True
):
    """
    Evaluates the Bitcoin direction prediction model using
    expanding-window walk-forward validation.

    Also compares the model with:
    - persistence baseline
    - moving average baseline
    """

    df = pd.read_csv(dataset_file)
    df = df.dropna().reset_index(drop=True)

    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    if len(df) <= min_train_size:
        raise ValueError(
            f"Dataset is too small. Samples: {len(df)}, min_train_size: {min_train_size}"
        )

    feature_columns = get_feature_columns(
        df,
        target_column=target_column
    )

    y_true = []
    y_pred_model = []
    y_pred_persistence = []
    y_pred_moving_average = []
    evaluation_dates = []

    for i in range(min_train_size, len(df)):
        train_df = df.iloc[:i]
        test_df = df.iloc[i:i + 1]

        X_train = train_df[feature_columns]
        y_train = train_df[target_column]

        X_test = test_df[feature_columns]
        y_test = int(test_df[target_column].iloc[0])

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        model.fit(X_train_scaled, y_train)

        model_prediction = int(model.predict(X_test_scaled)[0])

        # Baseline 1: Persistence
        # Predicts the same direction as the previous actual movement.
        persistence_prediction = int(df[target_column].iloc[i - 1])

        # Baseline 2: Moving average rule
        # If current close is above recent moving average, predict UP.
        if "close" in df.columns and i >= moving_average_window:
            recent_close_values = df["close"].iloc[i - moving_average_window:i]
            moving_average = recent_close_values.mean()
            current_close = df["close"].iloc[i]

            moving_average_prediction = 1 if current_close > moving_average else 0
        else:
            moving_average_prediction = persistence_prediction

        y_true.append(y_test)
        y_pred_model.append(model_prediction)
        y_pred_persistence.append(persistence_prediction)
        y_pred_moving_average.append(moving_average_prediction)

        if "date" in df.columns:
            evaluation_dates.append(str(df["date"].iloc[i]))

    model_metrics = calculate_metrics(y_true, y_pred_model)
    persistence_metrics = calculate_metrics(y_true, y_pred_persistence)
    moving_average_metrics = calculate_metrics(y_true, y_pred_moving_average)

    methods = {
        "model": model_metrics,
        "persistence": persistence_metrics,
        "moving_average": moving_average_metrics
    }

    best_accuracy_method = max(
        methods.items(),
        key=lambda item: item[1]["accuracy"]
    )

    best_f1_method = max(
        methods.items(),
        key=lambda item: item[1]["f1_score"]
    )

    result = {
        "evaluation_method": "expanding-window walk-forward validation",
        "configuration": {
            "dataset_file": dataset_file,
            "model_file": model_file,
            "target_column": target_column,
            "model": "LogisticRegression",
            "preprocessing": "StandardScaler",
            "random_state": 42,
            "moving_average_window": moving_average_window,
            "min_train_size": min_train_size,
            "feature_count": len(feature_columns)
        },
        "feature_columns": feature_columns,
        "dataset": {
            "total_samples": int(len(df)),
            "evaluation_steps": int(len(y_true)),
            "first_evaluation_date": evaluation_dates[0] if evaluation_dates else None,
            "last_evaluation_date": evaluation_dates[-1] if evaluation_dates else None,
            "skipped_steps": 0
        },
        "results": {
            "model": model_metrics,
            "baselines": {
                "persistence": persistence_metrics,
                "moving_average": moving_average_metrics
            }
        },
        "best_method_by_accuracy": {
            "metric": "accuracy",
            "best_method": best_accuracy_method[0],
            "best_value": best_accuracy_method[1]["accuracy"]
        },
        "best_method_by_f1_score": {
            "metric": "f1_score",
            "best_method": best_f1_method[0],
            "best_value": best_f1_method[1]["f1_score"]
        },
        "important_note": (
            "Accuracy close to 50% should not be interpreted as reliable market prediction. "
            "The system should be viewed as an experimental forecasting pipeline."
        )
    }

    if save:
        output_path = Path("data/evaluation_metrics.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        result["report_file"] = str(output_path)

    return result


def calculate_metrics(y_true, y_pred):
    """
    Calculates classification metrics.
    """

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0))
    }


if __name__ == "__main__":
    result = evaluate_model()
    print(json.dumps(result, indent=4, ensure_ascii=False))