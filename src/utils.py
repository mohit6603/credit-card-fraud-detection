"""Shared utilities: configuration loading, logging, and reproducibility helpers.

Every other module imports from here so that config and logging behavior
stay consistent across the whole project.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the YAML configuration file.

    Args:
        path: Path to the YAML config. Defaults to configs/config.yaml.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(relative: str | Path) -> Path:
    """Resolve a path from config (relative to project root) to an absolute Path."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def get_logger(name: str, config: dict[str, Any] | None = None) -> logging.Logger:
    """Create (or fetch) a logger that writes to both console and the log file.

    Args:
        name: Logger name, conventionally the module's __name__.
        config: Project config; loaded from default path if omitted.

    Returns:
        Configured logging.Logger instance.
    """
    if config is None:
        config = load_config()
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured -> avoid duplicate handlers
        return logger

    level = getattr(logging, config["logging"]["level"].upper(), logging.INFO)
    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_file = resolve_path(config["logging"]["file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    """Seed python and numpy RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    """Serialize a dictionary to pretty-printed JSON, creating parent dirs.

    Numpy scalar types are converted so json.dump does not fail.
    """
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Not JSON serializable: {type(o)}")

    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_default)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary."""
    path = resolve_path(path)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
