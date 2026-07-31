"""End-to-end training pipeline.

Run with:  python -m src.train_pipeline

Stages: load -> validate -> EDA -> engineer -> split -> experiment grid ->
tune winner -> threshold optimization (validation) -> final test evaluation
-> explainability -> persist model + metadata.

Every stage leaves artifacts on disk (artifacts/, reports/images/, logs/)
so the whole run is auditable after the fact.
"""

from __future__ import annotations

import sys
import time
from typing import Any

from src import data_loader, data_validation, eda
from src.evaluation import (
    evaluate_on_test,
    optimize_threshold,
    plot_evaluation_suite,
    plot_feature_importance,
)
from src.explainability import run_explainability
from src.feature_engineering import engineer_features, get_feature_columns
from src.model_training import refit_best, run_experiment_grid, tune_best
from src.preprocessing import drop_duplicates, make_splits
from src.utils import get_logger, load_config, resolve_path, save_json, set_seed

logger = get_logger(__name__)


def main(skip_eda: bool = False, skip_tuning: bool = False) -> dict[str, Any]:
    """Execute the full pipeline and return a run summary dictionary."""
    t_start = time.perf_counter()
    config = load_config()
    set_seed(config["split"]["random_state"])

    # 1) Data acquisition + validation ------------------------------------
    data_loader.download_data(config)
    df = data_loader.load_raw_data(config)
    report = data_validation.run_validation(df, config)
    if not report["passed"]:
        raise RuntimeError(f"Data validation failed: {report}")

    # 2) EDA (on raw data, before any transformation) ---------------------
    if not skip_eda:
        eda.run_eda(df, config)

    # 3) Cleaning + feature engineering + splits --------------------------
    df = drop_duplicates(df)
    df = engineer_features(df, config)
    feature_cols = get_feature_columns(df, config)
    splits = make_splits(df, feature_cols, config)

    # 4) Model x strategy experiment grid ---------------------------------
    results = run_experiment_grid(splits, config)
    results_path = resolve_path(config["paths"]["artifacts_dir"]) / "experiments.csv"
    results.to_csv(results_path, index=False)
    logger.info("Experiment grid saved to %s", results_path)

    # 5) Refit winner, then hyperparameter tuning -------------------------
    pipe, meta = refit_best(results, splits, config)
    if not skip_tuning:
        pipe, meta = tune_best(pipe, meta, splits, config)

    # 6) Threshold optimization on VALIDATION -----------------------------
    proba_val = pipe.predict_proba(splits.X_val)[:, 1]
    thr = optimize_threshold(splits.y_val.to_numpy(), proba_val,
                             splits.amount_val.to_numpy(), config)
    threshold = thr["best"]["cost_optimal"]

    # 7) Single-shot evaluation on TEST -----------------------------------
    proba_test = pipe.predict_proba(splits.X_test)[:, 1]
    evaluation = evaluate_on_test(splits.y_test.to_numpy(), proba_test,
                                  splits.amount_test.to_numpy(), threshold, config)
    plot_evaluation_suite(splits.y_test.to_numpy(), proba_test, threshold,
                          thr["sweep"], config)
    plot_feature_importance(pipe, splits.feature_names, config)

    # 8) Explainability ----------------------------------------------------
    explain = run_explainability(pipe, splits.X_test, splits.y_test, config)

    # 9) Persist model + metadata -----------------------------------------
    import joblib

    models_dir = resolve_path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, models_dir / "model.joblib")
    metadata = {
        **meta,
        "threshold": threshold,
        "threshold_alternatives": thr["best"],
        "feature_names": splits.feature_names,
        "test_metrics_tuned": evaluation["tuned_threshold"],
        "test_metrics_default": evaluation["default_threshold"],
        "business_cost": evaluation["business_cost"],
        "runtime_seconds": round(time.perf_counter() - t_start, 1),
    }
    save_json(metadata, f"{config['paths']['artifacts_dir']}/model_metadata.json")
    logger.info("Pipeline complete in %.1fs. Model: %s + %s | test PR-AUC=%.4f",
                metadata["runtime_seconds"], meta["model"], meta["strategy"],
                evaluation["tuned_threshold"]["pr_auc"])
    print(evaluation["classification_report"])
    return {"metadata": metadata, "experiments": results, "explainability": explain}


if __name__ == "__main__":
    skip_eda = "--skip-eda" in sys.argv
    skip_tuning = "--skip-tuning" in sys.argv
    main(skip_eda=skip_eda, skip_tuning=skip_tuning)
