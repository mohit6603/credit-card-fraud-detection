"""Inference pipeline: score single transactions or batches.

Loads the persisted model + metadata (feature order, tuned threshold, risk
bands) and exposes a clean API:

    detector = FraudDetector.load()
    detector.predict_one({"Time": 4000.0, "V1": ..., ..., "Amount": 149.62})
    -> {"fraud_probability": 0.97, "prediction": 1, "risk_level": "HIGH"}

Raw transactions come in with Time and Amount; the same feature engineering
used in training (Hour, Amount_log) is applied here so train and serve can
never drift apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import engineer_features
from src.utils import get_logger, load_config, load_json, resolve_path

logger = get_logger(__name__)

RAW_INPUT_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


class FraudDetector:
    """Loads the trained pipeline and scores transactions."""

    def __init__(self, pipeline: Any, metadata: dict[str, Any],
                 config: dict[str, Any]) -> None:
        """Prefer FraudDetector.load() over calling this directly."""
        self.pipeline = pipeline
        self.metadata = metadata
        self.config = config
        self.threshold: float = float(metadata["threshold"])
        self.feature_names: list[str] = list(metadata["feature_names"])

    @classmethod
    def load(cls, config: dict[str, Any] | None = None) -> "FraudDetector":
        """Load model.joblib + model_metadata.json from configured paths.

        Raises:
            FileNotFoundError: If artifacts are missing (run train_pipeline).
        """
        if config is None:
            config = load_config()
        model_path = resolve_path(config["paths"]["models_dir"]) / "model.joblib"
        meta_path = resolve_path(config["paths"]["artifacts_dir"]) / "model_metadata.json"
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Model artifacts not found ({model_path}, {meta_path}). "
                "Run: python -m src.train_pipeline")
        pipeline = joblib.load(model_path)
        metadata = load_json(meta_path)
        logger.info("Loaded %s + %s (threshold=%.3f)",
                    metadata.get("model"), metadata.get("strategy"),
                    float(metadata["threshold"]))
        return cls(pipeline, metadata, config)

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate raw input and apply training-time feature engineering.

        Raises:
            ValueError: If required raw columns are missing or non-numeric.
        """
        missing = [c for c in RAW_INPUT_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required input columns: {missing}")
        df = df[RAW_INPUT_COLUMNS].apply(pd.to_numeric, errors="coerce")
        if df.isna().any().any():
            bad = df.columns[df.isna().any()].tolist()
            raise ValueError(f"Non-numeric or missing values in columns: {bad}")
        engineered = engineer_features(df, self.config)
        return engineered[self.feature_names]

    def risk_level(self, probability: float) -> str:
        """Map a probability to LOW / MEDIUM / HIGH using configured bands."""
        bands = self.config["risk_levels"]
        if probability >= bands["high"]:
            return "HIGH"
        if probability >= bands["medium"]:
            return "MEDIUM"
        return "LOW"

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a batch of raw transactions.

        Args:
            df: DataFrame with Time, V1-V28, Amount columns.

        Returns:
            Copy of df with fraud_probability, prediction, risk_level added.
        """
        X = self._prepare(df)
        proba = self.pipeline.predict_proba(X)[:, 1]
        out = df.copy()
        out["fraud_probability"] = np.round(proba, 6)
        out["prediction"] = (proba >= self.threshold).astype(int)
        out["risk_level"] = [self.risk_level(p) for p in proba]
        logger.info("Scored batch of %d transactions (%d flagged)",
                    len(out), int(out["prediction"].sum()))
        return out

    def predict_one(self, transaction: dict[str, float]) -> dict[str, Any]:
        """Score a single transaction given as a dict of raw columns.

        Returns:
            {"fraud_probability", "prediction", "risk_level", "threshold"}.
        """
        row = self.predict_batch(pd.DataFrame([transaction])).iloc[0]
        return {
            "fraud_probability": float(row["fraud_probability"]),
            "prediction": int(row["prediction"]),
            "risk_level": str(row["risk_level"]),
            "threshold": self.threshold,
        }
