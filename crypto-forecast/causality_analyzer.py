import json
from pathlib import Path

import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


def adf_stationarity_test(series, name):
    """
    Performs Augmented Dickey-Fuller stationarity test.
    """

    series = series.dropna()

    result = adfuller(series, autolag="AIC")

    return {
        "series": name,
        "adf_statistic": float(result[0]),
        "p_value": float(result[1]),
        "used_lag": int(result[2]),
        "n_observations": int(result[3]),
        "is_stationary_5_percent": bool(result[1] < 0.05)
    }


def run_granger_causality(
        dataset_file="data/bitcoin_dataset.csv",
        target_column="price_return",
        cause_column="trend_change",
        max_lag=7,
        save=True
):
    """
    Performs Granger causality analysis.

    It checks whether past values of Google Trends changes help predict
    Bitcoin price changes.

    Important:
    Granger causality means predictive causality, not necessarily true
    cause-effect causality.
    """

    df = pd.read_csv(dataset_file)

    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    if cause_column not in df.columns:
        raise ValueError(f"Cause column not found: {cause_column}")

    data = df[[target_column, cause_column]].dropna().copy()

    if len(data) <= max_lag + 5:
        raise ValueError("Not enough data for Granger causality test.")

    stationarity_target = adf_stationarity_test(
        data[target_column],
        target_column
    )

    stationarity_cause = adf_stationarity_test(
        data[cause_column],
        cause_column
    )

    differencing_applied = False

    if not stationarity_target["is_stationary_5_percent"]:
        data[target_column] = data[target_column].diff()
        differencing_applied = True

    if not stationarity_cause["is_stationary_5_percent"]:
        data[cause_column] = data[cause_column].diff()
        differencing_applied = True

    data = data.dropna()

    if len(data) <= max_lag + 5:
        raise ValueError("Not enough data after differencing for Granger causality test.")

    # In statsmodels, the first column is the variable being predicted.
    # Here we test whether cause_column Granger-causes target_column.
    test_data = data[[target_column, cause_column]]

    raw_results = grangercausalitytests(
        test_data,
        maxlag=max_lag,
        verbose=False
    )

    lag_results = []

    for lag, result in raw_results.items():
        ssr_ftest = result[0]["ssr_ftest"]

        lag_results.append({
            "lag": int(lag),
            "f_statistic": float(ssr_ftest[0]),
            "p_value": float(ssr_ftest[1]),
            "df_denom": float(ssr_ftest[2]),
            "df_num": float(ssr_ftest[3]),
            "significant_5_percent": bool(ssr_ftest[1] < 0.05)
        })

    best_lag = min(
        lag_results,
        key=lambda item: item["p_value"]
    )

    significant_lags = [
        item for item in lag_results
        if item["significant_5_percent"]
    ]

    if significant_lags:
        conclusion = (
            f"The variable {cause_column} provides statistically significant "
            f"Granger predictive information for {target_column} at the 5% level "
            f"for at least one tested lag."
        )
    else:
        conclusion = (
            f"No statistically significant Granger predictive relationship was found "
            f"from {cause_column} to {target_column} at the 5% level."
        )

    output = {
        "analysis": "Granger causality",
        "dataset_file": dataset_file,
        "target_column": target_column,
        "cause_column": cause_column,
        "max_lag": max_lag,
        "interpretation_note": (
            "This test evaluates predictive causality. It does not prove a true "
            "cause-effect relationship."
        ),
        "stationarity": {
            "target": stationarity_target,
            "cause": stationarity_cause,
            "differencing_applied": differencing_applied
        },
        "lag_results": lag_results,
        "best_lag_by_p_value": best_lag,
        "significant_lags_5_percent": significant_lags,
        "conclusion": conclusion
    }

    if save:
        output_path = Path("data/causality_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(output, file, indent=4, ensure_ascii=False)

        output["report_file"] = str(output_path)

    return output


if __name__ == "__main__":
    results = run_granger_causality()
    print(json.dumps(results, indent=4, ensure_ascii=False))