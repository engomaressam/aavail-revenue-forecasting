"""Tests for the Flask API (src/model_api.py), on synthetic invoices in AAVAIL's format."""

import pytest

from model_api import create_app


@pytest.fixture
def client(invoice_dir, tmp_path):
    app = create_app(data_dir=str(invoice_dir), log_dir=str(tmp_path / "logs"))
    return app.test_client()


@pytest.fixture
def trained(client):
    assert client.post("/train", json={}).status_code == 200
    return client


def test_health_before_training(client):
    body = client.get("/health").get_json()
    assert body["status"] == "ok" and body["trained"] is False and body["default_method"] == "blend"


def test_predict_needs_training(client):
    assert client.post("/predict", json={"date": "2019-02-01"}).status_code == 409


def test_train_loads_and_backtests(client):
    body = client.post("/train", json={}).get_json()
    assert body["status"] == "trained" and body["countries"] == 2
    assert body["data_from"] == "2018-01-01" and body["data_to"] == "2019-03-31"
    assert set(body["backtest_all_countries"]) == {"blend", "naive", "seasonal"}


def test_train_rejects_a_missing_directory(client):
    assert client.post("/train", json={"data_dir": "no/such/dir"}).status_code == 400


def test_predict_for_a_date_inside_the_data_reports_the_actual(trained):
    body = trained.post("/predict", json={"date": "2019-02-01"}).get_json()
    assert body["method"].startswith("blend") and body["forecast_30d_revenue"] > 0
    assert "actual_30d_revenue" in body and body["error"] == pytest.approx(body["forecast_30d_revenue"] - body["actual_30d_revenue"], abs=0.02)
    assert "backtest_wape_pct" in body


def test_the_forecast_depends_on_the_date(trained):
    a = trained.post("/predict", json={"date": "2019-02-05"}).get_json()["forecast_30d_revenue"]
    b = trained.post("/predict", json={"date": "2019-03-01"}).get_json()["forecast_30d_revenue"]
    assert a != b


def test_predict_the_day_after_the_data_ends(trained):
    body = trained.post("/predict", json={"date": "2019-04-01", "method": "naive"}).get_json()
    assert body["forecast_30d_revenue"] > 0 and "actual_30d_revenue" not in body


def test_predict_one_country(trained):
    uk = trained.post("/predict", json={"date": "2019-02-01", "country": "United Kingdom"}).get_json()
    fr = trained.post("/predict", json={"date": "2019-02-01", "country": "France"}).get_json()
    assert uk["forecast_30d_revenue"] > fr["forecast_30d_revenue"] > 0


@pytest.mark.parametrize("payload, status", [
    ({"date": "01/02/2019"}, 400),                          # wrong date format
    ({"date": "2018-06-01"}, 400),                          # less than a year of history
    ({"date": "2019-06-01"}, 400),                          # beyond the data
    ({"date": "2019-02-01", "country": "Atlantis"}, 404),   # unknown country
    ({"date": "2019-02-01", "method": "lstm"}, 400),        # unknown method
])
def test_predict_rejects_bad_requests(trained, payload, status):
    assert trained.post("/predict", json=payload).status_code == status


def test_logs_record_training_and_predictions(trained):
    trained.post("/predict", json={"date": "2019-02-01"})
    month = trained.get("/health").get_json()["trained_at"][:7]
    train_log = trained.get(f"/logs?kind=train&month={month}").get_json()["entries"]
    predict_log = trained.get(f"/logs?kind=predict&month={month}").get_json()["entries"]
    assert len(train_log) == 1 and train_log[0]["invoices"] > 0
    assert predict_log[-1]["date"] == "2019-02-01"
    assert trained.get("/logs?kind=other").status_code == 400
