"""Tests for src.data_validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_validation import (
    check_duplicates,
    check_invalid_values,
    check_missing,
    check_target_distribution,
    run_validation,
    validate_schema,
)


class TestSchema:
    def test_valid_frame_passes(self, raw_df, config):
        cfg = {**config, "data": {**config["data"], "expected_rows_min": 100}}
        assert validate_schema(raw_df, cfg)["passed"]

    def test_missing_column_fails(self, raw_df, config):
        cfg = {**config, "data": {**config["data"], "expected_rows_min": 100}}
        result = validate_schema(raw_df.drop(columns=["V7"]), cfg)
        assert not result["passed"]
        assert any("V7" in issue for issue in result["issues"])

    def test_low_row_count_fails(self, raw_df, config):
        assert not validate_schema(raw_df, config)["passed"]  # min is 280k

    def test_wrong_dtype_fails(self, raw_df, config):
        cfg = {**config, "data": {**config["data"], "expected_rows_min": 100}}
        bad = raw_df.assign(Amount=raw_df["Amount"].astype(str))
        assert not validate_schema(bad, cfg)["passed"]


class TestChecks:
    def test_no_missing_passes(self, raw_df):
        assert check_missing(raw_df)["passed"]

    def test_missing_detected(self, raw_df):
        bad = raw_df.copy()
        bad.loc[0:4, "V1"] = np.nan
        result = check_missing(bad)
        assert not result["passed"]
        assert result["missing_by_column"]["V1"] == 5

    def test_duplicates_detected(self, raw_df):
        dup = pd.concat([raw_df, raw_df.head(3)], ignore_index=True)
        assert check_duplicates(dup)["n_duplicates"] == 3

    def test_negative_amount_flagged(self, raw_df, config):
        bad = raw_df.copy()
        bad.loc[0, "Amount"] = -5.0
        assert not check_invalid_values(bad, config)["passed"]

    def test_bad_target_flagged(self, raw_df, config):
        bad = raw_df.copy()
        bad.loc[0, "Class"] = 3
        assert not check_invalid_values(bad, config)["passed"]

    def test_target_distribution(self, raw_df, config):
        result = check_target_distribution(raw_df, config)
        assert result["passed"]
        assert result["n_fraud"] == 40
        assert result["n_genuine"] == 360
        assert abs(result["fraud_rate"] - 0.1) < 1e-9


def test_run_validation_end_to_end(raw_df, config):
    cfg = {**config, "data": {**config["data"], "expected_rows_min": 100}}
    report = run_validation(raw_df, cfg, save=False)
    assert report["passed"]
    assert report["n_rows"] == 400
