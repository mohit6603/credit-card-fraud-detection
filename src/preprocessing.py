"""Preprocessing: duplicate removal, stratified splitting, scaling pipeline.

Leakage rules enforced here:
- Duplicates are dropped BEFORE splitting so the same row can never sit in
  both train and test.
- The StandardScaler lives INSIDE the model pipeline, so it is fit only on
  whatever data the pipeline is fit on (training folds), never on test data.
- Resampling (SMOTE etc.) is attached via imblearn pipelines in
  model_training.py, which resample only during fit - never at predict time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import get_logger, load_config

logger = get_logger(__name__)


@dataclass
class DataSplits:
    """Container for the three-way stratified split.

    amount_* series keep the raw dollar Amount aligned with each split for
    business cost analysis (Amount itself is not a model feature).
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    amount_train: pd.Series
    amount_val: pd.Series
    amount_test: pd.Series
    feature_names: list[str]


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove fully duplicated rows, logging how many were dropped."""
    n_before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    logger.info("Dropped %d duplicate rows (%d -> %d)", n_before - len(out),
                n_before, len(out))
    return out


def make_splits(df: pd.DataFrame, feature_cols: list[str],
                config: dict[str, Any] | None = None) -> DataSplits:
    """Create stratified train/validation/test splits.

    Test is carved out first (untouched until final evaluation); validation
    is then carved from the remaining training data for model selection and
    threshold tuning.

    Args:
        df: Feature-engineered DataFrame including target and Amount columns.
        feature_cols: Names of model input columns.
        config: Project configuration; loaded from default if omitted.

    Returns:
        DataSplits with X/y for train, val, test plus aligned Amounts.

    Raises:
        ValueError: If the target column is missing or a split has no frauds.
    """
    if config is None:
        config = load_config()
    target = config["data"]["target"]
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in DataFrame")

    seed = config["split"]["random_state"]
    X = df[feature_cols]
    y = df[target]
    amount = df["Amount"] if "Amount" in df.columns else df["Amount_log"]

    X_tmp, X_test, y_tmp, y_test, amt_tmp, amt_test = train_test_split(
        X, y, amount,
        test_size=config["split"]["test_size"], stratify=y, random_state=seed,
    )
    X_train, X_val, y_train, y_val, amt_train, amt_val = train_test_split(
        X_tmp, y_tmp, amt_tmp,
        test_size=config["split"]["val_size"], stratify=y_tmp, random_state=seed,
    )
    for name, ys in [("train", y_train), ("val", y_val), ("test", y_test)]:
        if ys.sum() == 0:
            raise ValueError(f"Split '{name}' contains no fraud cases")
        logger.info("Split %-5s: %6d rows, %3d frauds (%.4f%%)",
                    name, len(ys), int(ys.sum()), 100 * ys.mean())

    return DataSplits(X_train, y_train, X_val, y_val, X_test, y_test,
                      amt_train, amt_val, amt_test, list(feature_cols))


def make_scaler() -> StandardScaler:
    """Return the shared scaler used inside every model pipeline.

    StandardScaler is chosen because V1-V28 are PCA outputs (roughly
    Gaussian, different variances) and engineered features are continuous.
    Tree models are scale-invariant but scaling is harmless to them, and a
    single uniform pipeline keeps train/serve behavior identical.
    """
    return StandardScaler()
