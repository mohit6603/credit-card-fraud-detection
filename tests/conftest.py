"""Shared fixtures: synthetic data shaped like creditcard.csv.

Tests never touch the real 144 MB dataset - they run on a small synthetic
frame with the same schema so the suite is fast and CI-friendly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config  # noqa: E402


@pytest.fixture(scope="session")
def config() -> dict:
    """Real project config (paths are only used by modules we don't test here)."""
    return load_config()


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    """Synthetic frame with the creditcard.csv schema: 400 rows, 40 frauds."""
    rng = np.random.RandomState(0)
    n, n_fraud = 400, 40
    df = pd.DataFrame({"Time": rng.uniform(0, 172800, n)})
    for i in range(1, 29):
        df[f"V{i}"] = rng.normal(0, 1, n)
    df["Amount"] = np.abs(rng.lognormal(3, 1, n)).round(2)
    cls = np.zeros(n, dtype=int)
    cls[:n_fraud] = 1
    rng.shuffle(cls)
    df["Class"] = cls
    # give frauds a real signal so tiny models can learn something
    df.loc[df["Class"] == 1, "V14"] -= 5.0
    return df
