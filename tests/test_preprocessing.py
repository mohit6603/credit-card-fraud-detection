"""Tests for src.preprocessing."""

from __future__ import annotations

import pandas as pd
import pytest

from src.feature_engineering import engineer_features, get_feature_columns
from src.preprocessing import drop_duplicates, make_splits


class TestDropDuplicates:
    def test_removes_exact_duplicates(self, raw_df):
        dup = pd.concat([raw_df, raw_df.head(5)], ignore_index=True)
        out = drop_duplicates(dup)
        assert len(out) == len(raw_df)

    def test_noop_when_clean(self, raw_df):
        assert len(drop_duplicates(raw_df)) == len(raw_df)


class TestMakeSplits:
    @pytest.fixture()
    def prepared(self, raw_df, config):
        df = engineer_features(raw_df, config)
        return df, get_feature_columns(df, config)

    def test_split_sizes_and_stratification(self, prepared, config):
        df, cols = prepared
        splits = make_splits(df, cols, config)
        total = len(splits.y_train) + len(splits.y_val) + len(splits.y_test)
        assert total == len(df)
        # test = 20% of all, val = 20% of remaining 80% = 16% of all
        assert len(splits.y_test) == pytest.approx(0.20 * len(df), abs=2)
        assert len(splits.y_val) == pytest.approx(0.16 * len(df), abs=2)
        # stratification keeps fraud rate within a small tolerance everywhere
        overall = df["Class"].mean()
        for ys in (splits.y_train, splits.y_val, splits.y_test):
            assert ys.mean() == pytest.approx(overall, abs=0.02)

    def test_no_row_overlap(self, prepared, config):
        df, cols = prepared
        splits = make_splits(df, cols, config)
        train_idx = set(splits.X_train.index)
        val_idx = set(splits.X_val.index)
        test_idx = set(splits.X_test.index)
        assert not (train_idx & val_idx)
        assert not (train_idx & test_idx)
        assert not (val_idx & test_idx)

    def test_amounts_aligned(self, prepared, config):
        df, cols = prepared
        splits = make_splits(df, cols, config)
        assert (splits.amount_test.index == splits.X_test.index).all()

    def test_missing_target_raises(self, prepared, config):
        df, cols = prepared
        with pytest.raises(ValueError):
            make_splits(df.drop(columns=["Class"]), cols, config)

    def test_reproducible(self, prepared, config):
        df, cols = prepared
        s1 = make_splits(df, cols, config)
        s2 = make_splits(df, cols, config)
        assert s1.X_test.index.equals(s2.X_test.index)
