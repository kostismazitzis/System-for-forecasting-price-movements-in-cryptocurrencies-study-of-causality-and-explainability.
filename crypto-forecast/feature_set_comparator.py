import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

from feature_engineering import get_feature_columns


def is_trend_feature(column_name):
    """
    Returns True if the feature is related to Google Trends.
    """

    return "trend" in column_name.lower()


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


def walk_forward_evaluate(
        df,
        feature_columns,
        target_column="target",
        min_train_size=60
):
    """
    Performs expanding-window walk-forward validation.
    """

    y_true = []
    y_pred = []

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

        prediction = int(model.predict(X_test_scaled)[0])

        y_true.append(y_test)
        y_pred.append(prediction)

    metrics = calculate_metrics(y_true, y_pred)
    metrics["evaluation_steps"] = int(len(y_true))

    return metrics


def compare_market_vs_trends(
        dataset_file="data/bitcoin_dataset.csv",
        target_column="target",
        min_train_size=60,
        save=True
):
    """
    Compares two feature sets:

    1. Market-only features
    2. Market features + Google Trends features

    This experiment checks whether Google Trends adds predictive value.
    """

    df = pd.read_csv(dataset_file)
    df = df.dropna().reset_index(drop=True)

    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    if len(df) <= min_train_size:
        raise ValueError(
            f"Dataset is too small. Samples: {len(df)}, min_train_size: {min_train_size}"
        )

    all_feature_columns = get_feature_columns(
        df,
        target_column=target_column
    )

    market_only_features = [
        col for col in all_feature_columns
        if not is_trend_feature(col)
    ]

    market_plus_trends_features = all_feature_columns

    if len(market_only_features) == 0:
        raise ValueError("No market-only features found.")

    if len(market_plus_trends_features) == 0:
        raise ValueError("No features found.")

    market_only_results = walk_forward_evaluate(
        df=df,
        feature_columns=market_only_features,
        target_column=target_column,
        min_train_size=min_train_size
    )

    market_plus_trends_results = walk_forward_evaluate(
        df=df,
        feature_columns=market_plus_trends_features,
        target_column=target_column,
        min_train_size=min_train_size
    )

    accuracy_difference = (
        market_plus_trends_results["accuracy"]
        - market_only_results["accuracy"]
    )

    f1_difference = (
        market_plus_trends_results["f1_score"]
        - market_only_results["f1_score"]
    )

    if accuracy_difference > 0:
        conclusion = (
            "In this experiment, the addition of Google Trends features improved "
            "the model accuracy compared to the market-only feature set."
        )
    elif accuracy_difference < 0:
        conclusion = (
            "In this experiment, the addition of Google Trends features did not improve "
            "the model accuracy compared to the market-only feature set."
        )
    else:
        conclusion = (
            "In this experiment, the addition of Google Trends features produced "
            "the same accuracy as the market-only feature set."
        )

    output = {
        "analysis": "Market-only vs Market + Google Trends comparison",
        "dataset_file": dataset_file,
        "target_column": target_column,
        "min_train_size": min_train_size,
        "feature_counts": {
            "market_only": int(len(market_only_features)),
            "market_plus_trends": int(len(market_plus_trends_features)),
            "google_trends_features": int(
                len(market_plus_trends_features) - len(market_only_features)
            )
        },
        "market_only_features": market_only_features,
        "google_trends_features": [
            col for col in all_feature_columns
            if is_trend_feature(col)
        ],
        "results": {
            "market_only": market_only_results,
            "market_plus_google_trends": market_plus_trends_results
        },
        "differences": {
            "accuracy_difference": float(accuracy_difference),
            "f1_score_difference": float(f1_difference)
        },
        "important_note": (
            "This comparison investigates whether Google Trends features add predictive value. "
            "Accuracy close to 50% should not be interpreted as reliable market prediction."
        ),
        "conclusion": conclusion
    }

    if save:
        output_path = Path("data/feature_set_comparison.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(output, file, indent=4, ensure_ascii=False)

        output["report_file"] = str(output_path)

    return output


if __name__ == "__main__":
    results = compare_market_vs_trends()
    print(json.dumps(results, indent=4, ensure_ascii=False))