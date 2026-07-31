"""Model construction and the model x imbalance-strategy experiment grid.

Why each model:
- logistic_regression: fast, calibrated-ish linear baseline; if trees cannot
  beat it, the extra complexity is unjustified.
- random_forest: robust non-linear bagging benchmark, little tuning needed.
- xgboost / lightgbm: gradient boosting - state of the art on tabular data,
  handles imbalance via scale_pos_weight.
- balanced_random_forest: undersamples the majority class inside each
  bootstrap, an ensemble-native answer to imbalance.

Why each strategy:
- class_weight: reweight the loss - no data modification, cheapest option.
- smote: synthesize minority points by interpolation - more minority signal,
  risk of unrealistic synthetic frauds.
- undersample: throw away majority rows - fast but discards information.
- smote_tomek: SMOTE then remove Tomek links - cleans the class boundary.

All resampling lives inside imblearn Pipelines, so it only ever happens on
the fit data - never on validation or test (leakage-safe by construction).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from src.preprocessing import DataSplits, make_scaler
from src.utils import get_logger, load_config

logger = get_logger(__name__)


def build_estimator(name: str, strategy: str, seed: int,
                    imbalance_ratio: float) -> Any:
    """Instantiate a base estimator configured for the given strategy.

    Args:
        name: One of the supported model names.
        strategy: Imbalance strategy; 'class_weight' configures the estimator
            itself, resampling strategies leave the estimator unweighted.
        seed: Random state for reproducibility.
        imbalance_ratio: n_negative / n_positive in the training data, used
            for scale_pos_weight in the boosting models.

    Returns:
        Unfitted sklearn-compatible estimator.

    Raises:
        ValueError: If the model name is unknown.
    """
    weighted = strategy == "class_weight"
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=3000, random_state=seed,
            class_weight="balanced" if weighted else None)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=seed,
            class_weight="balanced" if weighted else None)
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            tree_method="hist", eval_metric="aucpr", n_jobs=-1,
            random_state=seed,
            scale_pos_weight=imbalance_ratio if weighted else 1.0)
    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=300, learning_rate=0.1, num_leaves=31,
            n_jobs=-1, random_state=seed, verbose=-1,
            scale_pos_weight=imbalance_ratio if weighted else 1.0)
    if name == "balanced_random_forest":
        return BalancedRandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=seed,
            sampling_strategy="all", replacement=True, bootstrap=False)
    raise ValueError(f"Unknown model name: {name}")


def build_pipeline(model_name: str, strategy: str, seed: int,
                   imbalance_ratio: float, smote_k: int = 5) -> ImbPipeline:
    """Assemble scaler + optional resampler + estimator into one pipeline.

    Args:
        model_name: Supported model name.
        strategy: 'class_weight' | 'smote' | 'undersample' | 'smote_tomek'.
        seed: Random state.
        imbalance_ratio: Passed to boosting models when class-weighting.
        smote_k: k_neighbors for SMOTE.

    Returns:
        Unfitted imblearn Pipeline (safe to cross-validate without leakage).

    Raises:
        ValueError: If the strategy is unknown.
    """
    steps: list[tuple[str, Any]] = [("scaler", make_scaler())]
    if strategy == "class_weight":
        pass  # handled inside the estimator
    elif strategy == "smote":
        steps.append(("resampler", SMOTE(random_state=seed, k_neighbors=smote_k)))
    elif strategy == "undersample":
        steps.append(("resampler", RandomUnderSampler(random_state=seed)))
    elif strategy == "smote_tomek":
        steps.append(("resampler", SMOTETomek(
            random_state=seed, smote=SMOTE(random_state=seed, k_neighbors=smote_k))))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    steps.append(("model", build_estimator(model_name, strategy, seed, imbalance_ratio)))
    return ImbPipeline(steps)


def run_experiment_grid(splits: DataSplits,
                        config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Fit every (model, strategy) combination and score it on validation.

    BalancedRandomForest handles imbalance internally, so it runs once with
    strategy='internal' instead of the full strategy grid.

    Args:
        splits: Train/val/test splits from preprocessing.
        config: Project configuration.

    Returns:
        DataFrame of results sorted by validation PR-AUC (descending), with
        columns model, strategy, val_pr_auc, val_roc_auc, fit_seconds.
    """
    import time

    if config is None:
        config = load_config()
    seed = config["split"]["random_state"]
    smote_k = config["training"]["smote_k_neighbors"]
    ratio = float((splits.y_train == 0).sum() / (splits.y_train == 1).sum())

    rows: list[dict[str, Any]] = []
    for model_name in config["training"]["models"]:
        strategies = (["internal"] if model_name == "balanced_random_forest"
                      else config["training"]["strategies"])
        for strategy in strategies:
            eff_strategy = "class_weight" if strategy == "internal" else strategy
            pipe = build_pipeline(model_name, eff_strategy, seed, ratio, smote_k)
            if strategy == "internal":  # BRF resamples internally; no weighting
                pipe = ImbPipeline([("scaler", make_scaler()),
                                    ("model", build_estimator(model_name, "none", seed, ratio))])
            t0 = time.perf_counter()
            try:
                pipe.fit(splits.X_train, splits.y_train)
            except Exception as exc:
                logger.error("FAILED %s + %s: %s", model_name, strategy, exc)
                continue
            fit_s = time.perf_counter() - t0
            proba = pipe.predict_proba(splits.X_val)[:, 1]
            row = {
                "model": model_name,
                "strategy": strategy,
                "val_pr_auc": average_precision_score(splits.y_val, proba),
                "val_roc_auc": roc_auc_score(splits.y_val, proba),
                "fit_seconds": round(fit_s, 1),
            }
            rows.append(row)
            logger.info("%-24s + %-12s PR-AUC=%.4f ROC-AUC=%.4f (%.1fs)",
                        model_name, strategy, row["val_pr_auc"],
                        row["val_roc_auc"], fit_s)

    results = (pd.DataFrame(rows)
               .sort_values("val_pr_auc", ascending=False)
               .reset_index(drop=True))
    return results


PARAM_DISTRIBUTIONS: dict[str, dict[str, list[Any]]] = {
    "logistic_regression": {"model__C": [0.01, 0.1, 1.0, 10.0]},
    "random_forest": {"model__n_estimators": [200, 400],
                      "model__max_depth": [None, 10, 20],
                      "model__min_samples_leaf": [1, 2, 5]},
    "xgboost": {"model__n_estimators": [200, 300, 500],
                "model__max_depth": [4, 6, 8],
                "model__learning_rate": [0.05, 0.1, 0.2],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0]},
    "lightgbm": {"model__n_estimators": [200, 300, 500],
                 "model__num_leaves": [15, 31, 63],
                 "model__learning_rate": [0.05, 0.1, 0.2],
                 "model__subsample": [0.8, 1.0]},
    "balanced_random_forest": {"model__n_estimators": [200, 400],
                               "model__max_depth": [None, 10, 20]},
}


def tune_best(pipe: ImbPipeline, meta: dict[str, Any], splits: DataSplits,
              config: dict[str, Any] | None = None) -> tuple[ImbPipeline, dict[str, Any]]:
    """Randomized hyperparameter search for the winning pipeline.

    Uses stratified CV on the training split with average precision as the
    objective (consistent with the PR-AUC selection metric). If tuning fails
    to beat the untuned validation PR-AUC, the untuned pipeline is kept -
    tuning should never make the model worse.

    Args:
        pipe: Fitted winning pipeline (used as the search template).
        meta: Winner metadata (model name, strategy, untuned scores).
        splits: Data splits.
        config: Project configuration.

    Returns:
        (best fitted pipeline, updated metadata including tuning outcome).
    """
    from sklearn.base import clone
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

    if config is None:
        config = load_config()
    tuning_cfg = config["training"]["tuning"]
    grid = PARAM_DISTRIBUTIONS.get(meta["model"], {})
    if not grid:
        logger.info("No tuning grid for %s; keeping untuned model", meta["model"])
        return pipe, {**meta, "tuned": False}

    cv = StratifiedKFold(n_splits=tuning_cfg["cv_folds"], shuffle=True,
                         random_state=config["split"]["random_state"])
    search = RandomizedSearchCV(
        clone(pipe), grid, n_iter=tuning_cfg["n_iter"],
        scoring=tuning_cfg["scoring"], cv=cv, n_jobs=1, refit=True,
        random_state=config["split"]["random_state"], verbose=0)
    search.fit(splits.X_train, splits.y_train)

    tuned_val = average_precision_score(
        splits.y_val, search.best_estimator_.predict_proba(splits.X_val)[:, 1])
    logger.info("Tuning: CV best AP=%.4f, tuned val PR-AUC=%.4f vs untuned %.4f",
                search.best_score_, tuned_val, meta["val_pr_auc"])
    if tuned_val >= meta["val_pr_auc"]:
        return search.best_estimator_, {
            **meta, "tuned": True, "best_params": search.best_params_,
            "val_pr_auc": float(tuned_val)}
    logger.info("Tuned model did not beat untuned; keeping untuned pipeline")
    return pipe, {**meta, "tuned": False,
                  "tuning_attempted_params": search.best_params_}


def refit_best(results: pd.DataFrame, splits: DataSplits,
               config: dict[str, Any] | None = None) -> tuple[ImbPipeline, dict[str, Any]]:
    """Re-fit the winning (model, strategy) pipeline on the training split.

    Args:
        results: Output of run_experiment_grid (sorted best-first).
        splits: Data splits.
        config: Project configuration.

    Returns:
        (fitted pipeline, winner metadata dict).
    """
    if config is None:
        config = load_config()
    if results.empty:
        raise ValueError("Experiment results are empty; nothing to refit")
    best = results.iloc[0]
    seed = config["split"]["random_state"]
    ratio = float((splits.y_train == 0).sum() / (splits.y_train == 1).sum())
    strategy = best["strategy"]
    if strategy == "internal":
        pipe = ImbPipeline([("scaler", make_scaler()),
                            ("model", build_estimator(best["model"], "none", seed, ratio))])
    else:
        pipe = build_pipeline(best["model"], strategy, seed, ratio,
                              config["training"]["smote_k_neighbors"])
    pipe.fit(splits.X_train, splits.y_train)
    meta = {"model": best["model"], "strategy": strategy,
            "val_pr_auc": float(best["val_pr_auc"]),
            "val_roc_auc": float(best["val_roc_auc"])}
    logger.info("Refit best combo: %s + %s", meta["model"], meta["strategy"])
    return pipe, meta
