"""Tests for src.inference - uses a small model trained on synthetic data
so no persisted artifacts are required."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import engineer_features, get_feature_columns
from src.inference import RAW_INPUT_COLUMNS, FraudDetector


@pytest.fixture()
def detector(raw_df, config) -> FraudDetector:
    """FraudDetector wrapping a tiny model fit on the synthetic frame."""
    df = engineer_features(raw_df, config)
    cols = get_feature_columns(df, config)
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("model", LogisticRegression(max_iter=1000))])
    pipe.fit(df[cols], df["Class"])
    metadata = {"model": "logistic_regression", "strategy": "test",
                "threshold": 0.5, "feature_names": cols}
    return FraudDetector(pipe, metadata, config)


@pytest.fixture()
def one_raw_transaction(raw_df) -> dict:
    return {c: float(raw_df.iloc[0][c]) for c in RAW_INPUT_COLUMNS}


class TestPredictOne:
    def test_output_contract(self, detector, one_raw_transaction):
        result = detector.predict_one(one_raw_transaction)
        assert set(result) == {"fraud_probability", "prediction",
                               "risk_level", "threshold"}
        assert 0.0 <= result["fraud_probability"] <= 1.0
        assert result["prediction"] in (0, 1)
        assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH")

    def test_prediction_consistent_with_threshold(self, detector,
                                                  one_raw_transaction):
        result = detector.predict_one(one_raw_transaction)
        expected = int(result["fraud_probability"] >= result["threshold"])
        assert result["prediction"] == expected

    def test_missing_column_raises(self, detector, one_raw_transaction):
        del one_raw_transaction["V5"]
        with pytest.raises(ValueError, match="V5"):
            detector.predict_one(one_raw_transaction)

    def test_non_numeric_raises(self, detector, one_raw_transaction):
        one_raw_transaction["Amount"] = "not-a-number"
        with pytest.raises(ValueError):
            detector.predict_one(one_raw_transaction)


class TestPredictBatch:
    def test_batch_scores_all_rows(self, detector, raw_df):
        scored = detector.predict_batch(raw_df)
        assert len(scored) == len(raw_df)
        for col in ("fraud_probability", "prediction", "risk_level"):
            assert col in scored.columns

    def test_extra_columns_ignored(self, detector, raw_df):
        scored = detector.predict_batch(raw_df)  # includes Class column
        assert "Class" in scored.columns  # passthrough, not used as feature


class TestRiskLevels:
    @pytest.mark.parametrize("proba,expected", [
        (0.05, "LOW"), (0.29, "LOW"), (0.30, "MEDIUM"),
        (0.69, "MEDIUM"), (0.70, "HIGH"), (0.99, "HIGH")])
    def test_bands(self, detector, proba, expected):
        assert detector.risk_level(proba) == expected


def test_load_missing_artifacts_raises(config, tmp_path):
    cfg = {**config, "paths": {**config["paths"],
                               "models_dir": str(tmp_path / "nope"),
                               "artifacts_dir": str(tmp_path / "nope2")}}
    with pytest.raises(FileNotFoundError):
        FraudDetector.load(cfg)
