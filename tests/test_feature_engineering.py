"""Tests for src.feature_engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    add_amount_log,
    add_hour_feature,
    engineer_features,
    get_feature_columns,
)


class TestHour:
    def test_hour_computation(self):
        df = pd.DataFrame({"Time": [0.0, 3600.0, 3599.0, 25 * 3600.0]})
        out = add_hour_feature(df)
        assert out["Hour"].tolist() == [0, 1, 0, 1]  # wraps past 24h

    def test_missing_time_raises(self):
        with pytest.raises(KeyError):
            add_hour_feature(pd.DataFrame({"Amount": [1.0]}))

    def test_original_not_mutated(self):
        df = pd.DataFrame({"Time": [7200.0]})
        add_hour_feature(df)
        assert "Hour" not in df.columns


class TestAmountLog:
    def test_log1p_values(self):
        df = pd.DataFrame({"Amount": [0.0, 99.0]})
        out = add_amount_log(df)
        assert out["Amount_log"].tolist() == pytest.approx([0.0, np.log(100.0)])

    def test_negative_amount_raises(self):
        with pytest.raises(ValueError):
            add_amount_log(pd.DataFrame({"Amount": [-1.0]}))

    def test_missing_amount_raises(self):
        with pytest.raises(KeyError):
            add_amount_log(pd.DataFrame({"Time": [1.0]}))


class TestEngineerFeatures:
    def test_full_transform(self, raw_df, config):
        out = engineer_features(raw_df, config)
        assert "Hour" in out.columns
        assert "Amount_log" in out.columns
        assert "Time" not in out.columns  # dropped per config
        assert len(out) == len(raw_df)

    def test_feature_columns_exclude_target_and_amount(self, raw_df, config):
        out = engineer_features(raw_df, config)
        cols = get_feature_columns(out, config)
        assert "Class" not in cols
        assert "Amount" not in cols       # replaced by Amount_log
        assert "Amount_log" in cols
        assert "Hour" in cols
        assert len(cols) == 30            # V1-28 + Hour + Amount_log
