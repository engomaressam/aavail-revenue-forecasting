"""
AAVAIL revenue forecasting API.

  GET  /health                      service status and what is loaded
  POST /train   {"data_dir": "..."} load the invoices, backtest the methods, keep the data for forecasting
  POST /predict {"country": "all" | name, "date": "YYYY-MM-DD", "method": "blend" | "naive" | "seasonal" | "arima"}
                                    revenue forecast for the 30 days starting on that date, from the history before it;
                                    returns the actual too when the data already covers those 30 days
  GET  /logs?kind=train|predict&month=YYYY-MM   the API's own log lines (one JSON object per line)

Run: python src/model_api.py  (port from $PORT, default 8080; data from $AAVAIL_DATA_DIR, default data/cs-train)
"""

import json
import logging
import os
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

import evaluation as ev
from data_ingestion import DataIngestion

METHODS = {"blend": ev.blend, "naive": ev.naive, "seasonal": ev.seasonal_naive, "arima": ev.arima}
METHOD_NAMES = {"blend": "blend: last 30 days + last year", "naive": "naive (last 30 days)",
                "seasonal": "seasonal naive (last year)", "arima": "ARIMA(2,1,2)"}
DEFAULT_METHOD = "blend"


def create_app(data_dir: str = None, log_dir: str = None) -> Flask:
    app = Flask(__name__)
    state = {"df": None, "trained_at": None, "data_dir": data_dir or os.environ.get("AAVAIL_DATA_DIR", "data/cs-train"),
             "backtest": {}}
    logs = Path(log_dir or os.environ.get("AAVAIL_LOG_DIR", "logs"))
    logs.mkdir(parents=True, exist_ok=True)

    def write_log(kind: str, record: dict) -> None:
        record = {"timestamp": datetime.now().isoformat(timespec="seconds"), **record}
        with open(logs / f"{kind}-{date.today():%Y-%m}.log", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def error(message: str, status: int):
        return jsonify({"error": message}), status

    @app.get("/health")
    def health():
        df = state["df"]
        return jsonify({"status": "ok", "trained": df is not None, "trained_at": state["trained_at"],
                        "data_from": None if df is None else str(df["date"].min().date()),
                        "data_to": None if df is None else str(df["date"].max().date()),
                        "methods": list(METHODS), "default_method": DEFAULT_METHOD})

    @app.post("/train")
    def train():
        started = time.time()
        body = request.get_json(silent=True) or {}
        data_dir = body.get("data_dir", state["data_dir"])
        if not Path(data_dir).is_dir():
            return error(f"data directory not found: {data_dir}", 400)
        logging.disable(logging.WARNING)
        try:
            df = DataIngestion(data_dir).load_all_data()
        finally:
            logging.disable(logging.NOTSET)
        if df.empty:
            return error(f"no invoices found in {data_dir}", 400)
        # backtest the cheap methods on all countries together, so /predict can report a typical error
        series = ev.daily_revenue(df)
        backtest = {}
        if len(series) >= ev.YEAR + 2 * ev.HORIZON:
            sp = ev.split(series)
            actual = ev.windows(series, sp.test_origins)[1]
            for key in ("blend", "naive", "seasonal"):
                m = ev.metrics(actual, METHODS[key](series, sp.train_origins, sp.test_origins))
                backtest[key] = {"wape_pct": round(m["wape_pct"], 1), "mae": round(m["mae"]), "test_origins": len(sp.test_origins)}
        state.update(df=df, data_dir=data_dir, trained_at=datetime.now().isoformat(timespec="seconds"), backtest=backtest)
        summary = {"status": "trained", "data_dir": data_dir, "invoices": int(len(df)), "countries": int(df["country"].nunique()),
                   "data_from": str(df["date"].min().date()), "data_to": str(df["date"].max().date()),
                   "backtest_all_countries": backtest, "seconds": round(time.time() - started, 1)}
        write_log("train", summary)
        return jsonify(summary)

    @app.post("/predict")
    def predict():
        df = state["df"]
        if df is None:
            return error("no data loaded; call /train first", 409)
        body = request.get_json(silent=True) or {}
        country = body.get("country", "all")
        method = body.get("method", DEFAULT_METHOD)
        if method not in METHODS:
            return error(f"unknown method {method!r}; use one of {list(METHODS)}", 400)
        try:
            when = datetime.strptime(body.get("date", ""), "%Y-%m-%d")
        except ValueError:
            return error("date must be YYYY-MM-DD", 400)
        if country != "all" and country not in set(df["country"]):
            return error(f"unknown country {country!r}", 404)

        series = ev.daily_revenue(df, None if country == "all" else country)
        origin = (when - series.index[0].to_pydatetime()).days
        earliest = ev.YEAR + ev.HORIZON
        if origin < earliest:
            return error(f"need a year of history before the date; earliest is {(series.index[0] + pd.Timedelta(days=earliest)).date()}", 400)
        if origin > len(series):
            return error(f"the data ends {series.index[-1].date()}; the latest forecast date is the day after", 400)

        forecast = float(METHODS[method](series, [], [origin])[0])
        result = {"country": country, "date": when.strftime("%Y-%m-%d"), "method": METHOD_NAMES[method],
                  "forecast_30d_revenue": round(forecast, 2)}
        if origin + ev.HORIZON <= len(series):   # the data already covers the window: report how it did
            actual = float(series.values[origin:origin + ev.HORIZON].sum())
            result.update(actual_30d_revenue=round(actual, 2), error=round(forecast - actual, 2))
        if country == "all" and method in state["backtest"]:
            result["backtest_wape_pct"] = state["backtest"][method]["wape_pct"]
        write_log("predict", result)
        return jsonify(result)

    @app.get("/logs")
    def read_logs():
        kind = request.args.get("kind", "predict")
        if kind not in ("train", "predict"):
            return error("kind must be train or predict", 400)
        month = request.args.get("month", f"{date.today():%Y-%m}")
        path = logs / f"{kind}-{month}.log"
        lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        return jsonify({"kind": kind, "month": month, "entries": lines})

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
