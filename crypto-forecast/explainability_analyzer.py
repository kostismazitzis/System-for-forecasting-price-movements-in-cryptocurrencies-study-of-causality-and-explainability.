import json
from pathlib import Path

import joblib
import pandas as pd


def run_logistic_regression_explainability(
        model_file="saved_models/bitcoin_direction_model.joblib",
        dataset_file="data/bitcoin_dataset.csv",
        target_column="target",
        top_n=15,
        save=True
):
    model_bundle = joblib.load(model_file)

    if isinstance(model_bundle, dict):
        model = model_bundle.get("model")
        scaler = model_bundle.get("scaler")
        feature_columns = model_bundle.get("feature_columns")
    else:
        raise ValueError("Expected model file to contain a dictionary with model, scaler and feature_columns.")

    if model is None:
        raise ValueError("Model not found in model bundle.")

    if feature_columns is None:
        df = pd.read_csv(dataset_file)
        feature_columns = [
            col for col in df.columns
            if col not in [target_column, "date", "timestamp", "close_time"]
        ]

    coefficients = model.coef_[0]

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "coefficient": coefficients,
        "absolute_importance": abs(coefficients)
    })

    importance_df = importance_df.sort_values(
        by="absolute_importance",
        ascending=False
    )

    top_positive = importance_df.sort_values(
        by="coefficient",
        ascending=False
    ).head(top_n)

    top_negative = importance_df.sort_values(
        by="coefficient",
        ascending=True
    ).head(top_n)

    output = {
        "analysis": "Logistic Regression coefficient explainability",
        "model_file": model_file,
        "dataset_file": dataset_file,
        "target_column": target_column,
        "interpretation": {
            "positive_coefficients": "Increase the probability of class 1, interpreted as UP.",
            "negative_coefficients": "Increase the probability of class 0, interpreted as DOWN.",
            "note": "Because StandardScaler is used, coefficients are more comparable across features."
        },
        "top_positive_features": top_positive.to_dict(orient="records"),
        "top_negative_features": top_negative.to_dict(orient="records"),
        "top_features_by_absolute_importance": importance_df.head(top_n).to_dict(orient="records")
    }

    if save:
        output_path = Path("data/explainability_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

        output["report_file"] = str(output_path)

        csv_path = Path("data/explainability_coefficients.csv")
        importance_df.to_csv(csv_path, index=False)
        output["coefficients_file"] = str(csv_path)

    return output


if __name__ == "__main__":
    results = run_logistic_regression_explainability()
    print(json.dumps(results, indent=4, ensure_ascii=False))