from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

from feature_engineering import get_feature_columns


def train_bitcoin_direction_model(
        dataset_file="data/bitcoin_dataset.csv",
        model_file="saved_models/bitcoin_direction_model.joblib",
        target_column="target",
        test_size=0.2
):
    """
    Trains a Logistic Regression model for Bitcoin next-day direction prediction.

    Class 1 = UP
    Class 0 = DOWN
    """

    df = pd.read_csv(dataset_file)
    df = df.dropna().reset_index(drop=True)

    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    feature_columns = get_feature_columns(
        df,
        target_column=target_column
    )

    X = df[feature_columns]
    y = df[target_column]

    split_index = int(len(df) * (1 - test_size))

    if split_index <= 0 or split_index >= len(df):
        raise ValueError("Invalid train/test split. Dataset may be too small.")

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]

    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    latest_row = X.iloc[[-1]]
    latest_row_scaled = scaler.transform(latest_row)

    latest_prediction_class = int(model.predict(latest_row_scaled)[0])
    latest_prediction_proba = model.predict_proba(latest_row_scaled)[0]

    latest_prediction = {
        "class": latest_prediction_class,
        "label": "UP" if latest_prediction_class == 1 else "DOWN",
        "prob_down": float(latest_prediction_proba[0]),
        "prob_up": float(latest_prediction_proba[1])
    }

    model_bundle = {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "classes": {
            0: "DOWN",
            1: "UP"
        }
    }

    model_path = Path(model_file)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_bundle, model_path)

    return {
        "message": "Model trained successfully.",
        "model": "LogisticRegression",
        "preprocessing": "StandardScaler",
        "dataset_file": dataset_file,
        "model_file": model_file,
        "target_column": target_column,
        "samples": int(len(df)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        },
        "latest_prediction": latest_prediction
    }


if __name__ == "__main__":
    result = train_bitcoin_direction_model()
    print(result)