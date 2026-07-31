"""Data validation: schema, missing values, duplicates, dtypes, target sanity.

Produces a machine-readable validation report (saved to artifacts/) so that
every run leaves an auditable record of data quality.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config, save_json

logger = get_logger(__name__)

EXPECTED_SCHEMA: dict[str, str] = {
    "Time": "float",
    **{f"V{i}": "float" for i in range(1, 29)},
    "Amount": "float",
    "Class": "int",
}


def validate_schema(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Check column names, count, and dtypes against the expected schema.

    Args:
        df: Raw transactions DataFrame.
        config: Project configuration.

    Returns:
        Dict with 'passed' flag and details of any violations.
    """
    issues: list[str] = []
    expected_cols = list(EXPECTED_SCHEMA)
    missing_cols = [c for c in expected_cols if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in expected_cols]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    if extra_cols:
        issues.append(f"Unexpected columns: {extra_cols}")
    if len(df) < config["data"]["expected_rows_min"]:
        issues.append(f"Row count {len(df)} below expected minimum")

    for col, kind in EXPECTED_SCHEMA.items():
        if col not in df.columns:
            continue
        if kind == "float" and not pd.api.types.is_float_dtype(df[col]):
            issues.append(f"{col}: expected float dtype, got {df[col].dtype}")
        if kind == "int" and not pd.api.types.is_integer_dtype(df[col]):
            issues.append(f"{col}: expected integer dtype, got {df[col].dtype}")

    return {"passed": not issues, "issues": issues}


def check_missing(df: pd.DataFrame) -> dict[str, Any]:
    """Report missing-value counts per column (only columns that have any)."""
    counts = df.isna().sum()
    nonzero = counts[counts > 0].to_dict()
    return {"passed": len(nonzero) == 0, "missing_by_column": nonzero,
            "total_missing": int(counts.sum())}


def check_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    """Report fully duplicated rows (identical across every column)."""
    dup_mask = df.duplicated()
    return {"passed": bool(~dup_mask.any()), "n_duplicates": int(dup_mask.sum())}


def check_invalid_values(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Check for values that are impossible in this domain.

    Rules: Amount must be >= 0, Time must be >= 0, Class must be in {0, 1},
    and no column may contain inf.
    """
    issues: list[str] = []
    if (df["Amount"] < 0).any():
        issues.append(f"{int((df['Amount'] < 0).sum())} negative Amount values")
    if (df["Time"] < 0).any():
        issues.append(f"{int((df['Time'] < 0).sum())} negative Time values")
    target = config["data"]["target"]
    bad_target = ~df[target].isin([0, 1])
    if bad_target.any():
        issues.append(f"{int(bad_target.sum())} target values outside {{0,1}}")
    numeric = df.select_dtypes(include=[np.number])
    n_inf = int(np.isinf(numeric.to_numpy()).sum())
    if n_inf:
        issues.append(f"{n_inf} infinite values")
    return {"passed": not issues, "issues": issues}


def check_target_distribution(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Summarize the class balance - the defining property of this dataset."""
    target = config["data"]["target"]
    counts = df[target].value_counts().to_dict()
    n_fraud = int(counts.get(1, 0))
    n_genuine = int(counts.get(0, 0))
    fraud_rate = n_fraud / len(df) if len(df) else 0.0
    return {
        "passed": n_fraud > 0 and n_genuine > 0,
        "n_genuine": n_genuine,
        "n_fraud": n_fraud,
        "fraud_rate": round(fraud_rate, 6),
        "imbalance_ratio": round(n_genuine / n_fraud, 1) if n_fraud else None,
    }


def run_validation(df: pd.DataFrame, config: dict[str, Any] | None = None,
                   save: bool = True) -> dict[str, Any]:
    """Run every validation check and optionally persist the report.

    Args:
        df: Raw transactions DataFrame.
        config: Project configuration; loaded from default if omitted.
        save: Write the report to artifacts/validation_report.json.

    Returns:
        Full validation report with an overall 'passed' flag.
    """
    if config is None:
        config = load_config()
    report: dict[str, Any] = {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "schema": validate_schema(df, config),
        "missing": check_missing(df),
        "duplicates": check_duplicates(df),
        "invalid_values": check_invalid_values(df, config),
        "target_distribution": check_target_distribution(df, config),
    }
    checks = [v["passed"] for k, v in report.items() if isinstance(v, dict)]
    # Duplicates fail the check but are handled (dropped) downstream, so they
    # do not fail overall validation - they are logged for transparency.
    report["passed"] = all(
        v["passed"] for k, v in report.items()
        if isinstance(v, dict) and k != "duplicates"
    )
    for name, result in report.items():
        if isinstance(result, dict):
            logger.info("Validation [%s]: %s", name,
                        "PASS" if result.get("passed") else f"ATTENTION {result}")
    if save:
        save_json(report, f"{config['paths']['artifacts_dir']}/validation_report.json")
        logger.info("Validation report saved to artifacts/validation_report.json")
    return report
