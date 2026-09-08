from flask import Flask, jsonify, request

from config import HOST, PORT, DEBUG

from dataset_builder import build_bitcoin_dataset
from model_trainer import train_bitcoin_direction_model
from evaluator import evaluate_model

from causality_analyzer import run_granger_causality
from explainability_analyzer import run_logistic_regression_explainability
from feature_set_comparator import compare_market_vs_trends


app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "application": "Crypto Forecast API",
        "version": "1.0.0",
        "status": "running",
        "description": "Prediction of Bitcoin price direction using market data, Google Trends and machine learning.",
        "target": "Bitcoin next-day direction",
        "classes": {
            "0": "DOWN",
            "1": "UP"
        },
        "available_endpoints": [
            "/",
            "/health",
            "/train",
            "/predict",
            "/metrics",
            "/causality",
            "/explainability",
            "/feature-comparison"
        ],
        "research_extensions": {
            "causality": "Granger causality analysis between Google Trends changes and Bitcoin price changes.",
            "explainability": "Logistic Regression coefficient analysis for model interpretability.",
            "feature_comparison": "Comparison between market-only features and market + Google Trends features."
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Crypto Forecast API is running."
    })


@app.route("/train", methods=["POST", "GET"])
def train():
    try:
        bitcoin_limit = int(request.args.get("bitcoin_limit", 365))
        trends_timeframe = request.args.get("trends_timeframe", "today 12-m")
        keyword = request.args.get("keyword", "Bitcoin")

        dataset_result = build_bitcoin_dataset(
            bitcoin_limit=bitcoin_limit,
            trends_timeframe=trends_timeframe,
            keyword=keyword
        )

        dataset_file = dataset_result.get(
            "dataset_file",
            "data/bitcoin_dataset.csv"
        )

        training_result = train_bitcoin_direction_model(
            dataset_file=dataset_file
        )

        return jsonify({
            "message": "Training completed successfully.",
            "dataset": dataset_result,
            "training": training_result
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/predict", methods=["GET"])
def predict():
    try:
        dataset_file = request.args.get(
            "dataset_file",
            "data/bitcoin_dataset.csv"
        )

        model_file = request.args.get(
            "model_file",
            "saved_models/bitcoin_direction_model.joblib"
        )

        training_result = train_bitcoin_direction_model(
            dataset_file=dataset_file,
            model_file=model_file
        )

        prediction = training_result.get("latest_prediction", None)

        return jsonify({
            "message": "Bitcoin direction prediction completed.",
            "dataset_file": dataset_file,
            "model_file": model_file,
            "classes": {
                "0": "DOWN",
                "1": "UP"
            },
            "prediction": prediction,
            "training": training_result
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    try:
        dataset_file = request.args.get(
            "dataset_file",
            "data/bitcoin_dataset.csv"
        )

        model_file = request.args.get(
            "model_file",
            "saved_models/bitcoin_direction_model.joblib"
        )

        min_train_size = int(request.args.get("min_train_size", 60))
        moving_average_window = int(request.args.get("moving_average_window", 7))
        save = request.args.get("save", "1") == "1"

        results = evaluate_model(
            dataset_file=dataset_file,
            model_file=model_file,
            min_train_size=min_train_size,
            moving_average_window=moving_average_window,
            save=save
        )

        return jsonify(results)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/causality", methods=["GET"])
def causality():
    try:
        dataset_file = request.args.get(
            "dataset_file",
            "data/bitcoin_dataset.csv"
        )

        target_column = request.args.get(
            "target_column",
            "price_return"
        )

        cause_column = request.args.get(
            "cause_column",
            "trend_change"
        )

        max_lag = int(request.args.get("max_lag", 7))
        save = request.args.get("save", "1") == "1"

        results = run_granger_causality(
            dataset_file=dataset_file,
            target_column=target_column,
            cause_column=cause_column,
            max_lag=max_lag,
            save=save
        )

        return jsonify(results)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/explainability", methods=["GET"])
def explainability():
    try:
        model_file = request.args.get(
            "model_file",
            "saved_models/bitcoin_direction_model.joblib"
        )

        dataset_file = request.args.get(
            "dataset_file",
            "data/bitcoin_dataset.csv"
        )

        target_column = request.args.get(
            "target_column",
            "target"
        )

        top_n = int(request.args.get("top_n", 15))
        save = request.args.get("save", "1") == "1"

        results = run_logistic_regression_explainability(
            model_file=model_file,
            dataset_file=dataset_file,
            target_column=target_column,
            top_n=top_n,
            save=save
        )

        return jsonify(results)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/feature-comparison", methods=["GET"])
def feature_comparison():
    try:
        dataset_file = request.args.get(
            "dataset_file",
            "data/bitcoin_dataset.csv"
        )

        target_column = request.args.get(
            "target_column",
            "target"
        )

        min_train_size = int(request.args.get("min_train_size", 60))
        save = request.args.get("save", "1") == "1"

        results = compare_market_vs_trends(
            dataset_file=dataset_file,
            target_column=target_column,
            min_train_size=min_train_size,
            save=save
        )

        return jsonify(results)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host=HOST,
        port=PORT,
        debug=DEBUG
    )