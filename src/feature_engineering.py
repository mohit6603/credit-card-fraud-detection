"""Feature engineering.

The V1-V28 columns are already PCA components (anonymized by the dataset
publisher), so most signal is pre-extracted. What we CAN engineer comes from
the two raw columns:

- Hour:       Time is seconds elapsed since the first transaction in a 2-day
              window. Hour-of-day (0-23) is the generalizable signal - fraud
              in this dataset spikes in the early-morning hours when
              cardholders are asleep. Raw Time itself is dropped: "seconds
              since collection started" would leak the collection window and
              means nothing for a future transaction.
- Amount_log: log1p(Amount) tames the extreme right skew (median ~$22,
              max ~$25k) so linear models are not dominated by outliers.
              Raw Amount is kept for business cost analysis but excluded
              from the model features in favor of the log version.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config

logger = get_logger(__name__)

SECONDS_PER_HOUR = 3600
HOURS_PER_DAY = 24


def add_hour_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Derive hour-of-day (0-23) from the Time column.

    Args:
        df: DataFrame containing a 'Time' column (seconds from first txn).

    Returns:
        Copy of df with an added integer 'Hour' column.

    Raises:
        KeyError: If 'Time' is absent.
    """
    if "Time" not in df.columns:
        raise KeyError("Column 'Time' required to derive 'Hour'")
    out = df.copy()
    out["Hour"] = ((out["Time"] // SECONDS_PER_HOUR) % HOURS_PER_DAY).astype(int)
    return out


def add_amount_log(df: pd.DataFrame) -> pd.DataFrame:
    """Add log1p-transformed transaction amount.

    Args:
        df: DataFrame containing an 'Amount' column.

    Returns:
        Copy of df with an added float 'Amount_log' column.

    Raises:
        KeyError: If 'Amount' is absent.
        ValueError: If Amount contains negatives (log1p undefined region).
    """
    if "Amount" not in df.columns:
        raise KeyError("Column 'Amount' required to derive 'Amount_log'")
    if (df["Amount"] < 0).any():
        raise ValueError("Negative Amount values found; cannot apply log1p")
    out = df.copy()
    out["Amount_log"] = np.log1p(out["Amount"])
    return out


def engineer_features(df: pd.DataFrame,
                      config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Apply all configured feature engineering steps.

    Row-wise transforms only (no statistics learned from data), so applying
    this before the train/test split cannot leak information.

    Args:
        df: Raw transactions DataFrame.
        config: Project configuration; loaded from default if omitted.

    Returns:
        DataFrame with engineered columns added and configured drops applied.
    """
    if config is None:
        config = load_config()
    out = df.copy()
    eng = config["features"]["engineered"]
    if eng.get("hour", False):
        out = add_hour_feature(out)
    if eng.get("amount_log", False):
        out = add_amount_log(out)
    drops = [c for c in config["features"].get("drop", []) if c in out.columns]
    if drops:
        out = out.drop(columns=drops)
    logger.info("Feature engineering done: %d columns (%s added, %s dropped)",
                out.shape[1],
                [k for k, v in eng.items() if v], drops)
    return out


def get_feature_columns(df: pd.DataFrame, config: dict[str, Any] | None = None) -> list[str]:
    """Return model feature columns: everything except target and raw Amount.

    Raw Amount is retained in the frame for cost analysis but the model uses
    Amount_log instead (when enabled) to avoid redundant, skewed input.
    """
    if config is None:
        config = load_config()
    target = config["data"]["target"]
    exclude = {target}
    if config["features"]["engineered"].get("amount_log", False):
        exclude.add("Amount")
    return [c for c in df.columns if c not in exclude]
