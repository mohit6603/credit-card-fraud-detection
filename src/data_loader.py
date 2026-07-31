"""Data acquisition and loading.

Downloads the Kaggle credit-card fraud dataset (via its public Google-storage
mirror) if not already present, and loads it with basic integrity checks.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import get_logger, load_config, resolve_path

logger = get_logger(__name__)


def download_data(config: dict[str, Any] | None = None, force: bool = False) -> Path:
    """Download creditcard.csv to data/raw/ if it is not already there.

    Args:
        config: Project configuration; loaded from default path if omitted.
        force: Re-download even if the file exists.

    Returns:
        Path to the local CSV file.

    Raises:
        RuntimeError: If the download fails or produces an implausibly small file.
    """
    if config is None:
        config = load_config()
    dest = resolve_path(config["paths"]["raw_data"])
    if dest.exists() and not force:
        logger.info("Dataset already present at %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
        return dest

    url = config["data"]["url"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading dataset from %s", url)
    try:
        urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed https URL from config
    except Exception as exc:  # pragma: no cover - network failure path
        raise RuntimeError(f"Failed to download dataset: {exc}") from exc

    if dest.stat().st_size < 50e6:  # the real file is ~144 MB
        raise RuntimeError(f"Downloaded file is suspiciously small: {dest.stat().st_size} bytes")
    logger.info("Downloaded %.1f MB to %s", dest.stat().st_size / 1e6, dest)
    return dest


def load_raw_data(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load the raw dataset into a DataFrame.

    Args:
        config: Project configuration; loaded from default path if omitted.

    Returns:
        Raw transactions DataFrame with Time, V1-V28, Amount, Class columns.

    Raises:
        FileNotFoundError: If the CSV is missing (run download_data first).
        ValueError: If the file cannot be parsed as CSV.
    """
    if config is None:
        config = load_config()
    path = resolve_path(config["paths"]["raw_data"])
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run src.data_loader.download_data() first."
        )
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not parse {path} as CSV: {exc}") from exc
    logger.info("Loaded raw data: %d rows x %d columns", *df.shape)
    return df
