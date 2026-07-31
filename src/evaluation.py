"""Evaluation: metrics, threshold optimization, business cost, and plots.

The core design decision: the model outputs probabilities, and the DECISION
threshold is chosen on the validation set to minimize business cost
(missed-fraud dollars + false-alarm handling cost), then applied unchanged
to the held-out test set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils import get_logger, load_config, resolve_path, save_json

logger = get_logger(__name__)


def compute_metrics(y_true: np.ndarray, proba: np.ndarray,
                    threshold: float) -> dict[str, float]:
    """Compute the full metric set at a given decision threshold.

    Args:
        y_true: Ground-truth labels (0/1).
        proba: Predicted fraud probabilities.
        threshold: Decision cutoff applied to proba.

    Returns:
        Dict of threshold-dependent and threshold-free metrics.
    """
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def business_cost(y_true: np.ndarray, pred: np.ndarray, amounts: np.ndarray,
                  fp_cost: float) -> float:
    """Total dollar cost of a decision vector.

    Missed fraud (FN) costs the full transaction amount; a false alarm (FP)
    costs a flat handling fee. Catching fraud (TP) and approving genuine
    (TN) cost nothing here.
    """
    fn_mask = (y_true == 1) & (pred == 0)
    fp_mask = (y_true == 0) & (pred == 1)
    return float(amounts[fn_mask].sum() + fp_cost * fp_mask.sum())


def optimize_threshold(y_val: np.ndarray, proba_val: np.ndarray,
                       amounts_val: np.ndarray,
                       config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Sweep thresholds on the VALIDATION set and pick the cost-minimizing one.

    Also records the F1-optimal and F2-optimal thresholds for comparison so
    the report can show the trade-off space, not just one answer.

    Args:
        y_val: Validation labels.
        proba_val: Validation fraud probabilities.
        amounts_val: Validation transaction amounts (dollars).
        config: Project configuration.

    Returns:
        Dict with best thresholds per objective and the full sweep table.
    """
    if config is None:
        config = load_config()
    fp_cost = config["threshold"]["fp_cost"]
    grid = np.linspace(0.001, 0.999, config["threshold"]["grid_points"])

    records = []
    for t in grid:
        pred = (proba_val >= t).astype(int)
        cost = business_cost(y_val, pred, amounts_val, fp_cost)
        p = precision_score(y_val, pred, zero_division=0)
        r = recall_score(y_val, pred, zero_division=0)
        f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        f2 = 0.0 if 4 * p + r == 0 else 5 * p * r / (4 * p + r)
        records.append({"threshold": t, "cost": cost, "precision": p,
                        "recall": r, "f1": f1, "f2": f2})
    sweep = pd.DataFrame(records)

    best = {
        "cost_optimal": float(sweep.loc[sweep["cost"].idxmin(), "threshold"]),
        "f1_optimal": float(sweep.loc[sweep["f1"].idxmax(), "threshold"]),
        "f2_optimal": float(sweep.loc[sweep["f2"].idxmax(), "threshold"]),
        "min_cost": float(sweep["cost"].min()),
        "cost_at_default_0.5": float(
            sweep.iloc[(sweep["threshold"] - 0.5).abs().idxmin()]["cost"]),
    }
    logger.info("Threshold optimization: cost-optimal=%.3f (cost $%.0f) vs "
                "default 0.5 (cost $%.0f); F1-optimal=%.3f",
                best["cost_optimal"], best["min_cost"],
                best["cost_at_default_0.5"], best["f1_optimal"])
    return {"best": best, "sweep": sweep}


def _save_fig(fig: plt.Figure, name: str, config: dict[str, Any]) -> Path:
    out = resolve_path(config["paths"]["images_dir"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_evaluation_suite(y_test: np.ndarray, proba_test: np.ndarray,
                          threshold: float, sweep: pd.DataFrame,
                          config: dict[str, Any]) -> list[Path]:
    """Confusion matrices, ROC, PR curve, and the threshold-cost curve."""
    paths: list[Path] = []

    # Confusion matrices: default vs tuned threshold, side by side.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, t, title in [(axes[0], 0.5, "Default threshold 0.50"),
                         (axes[1], threshold, f"Tuned threshold {threshold:.3f}")]:
        cm = confusion_matrix(y_test, (proba_test >= t).astype(int))
        ConfusionMatrixDisplay(cm, display_labels=["Genuine", "Fraud"]).plot(
            ax=ax, colorbar=False, values_format="d", cmap="Blues")
        ax.set_title(title)
    fig.suptitle("Test-Set Confusion Matrix: Before vs After Threshold Tuning")
    paths.append(_save_fig(fig, "confusion_matrices.png", config))

    # ROC curve.
    fpr, tpr, _ = roc_curve(y_test, proba_test)
    auc = roc_auc_score(y_test, proba_test)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, lw=2.5, label=f"ROC-AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve (test set)")
    ax.legend()
    paths.append(_save_fig(fig, "roc_curve.png", config))

    # Precision-Recall curve with operating point.
    prec, rec, _ = precision_recall_curve(y_test, proba_test)
    ap = average_precision_score(y_test, proba_test)
    op_p = precision_score(y_test, (proba_test >= threshold).astype(int), zero_division=0)
    op_r = recall_score(y_test, (proba_test >= threshold).astype(int), zero_division=0)
    baseline = y_test.mean()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(rec, prec, lw=2.5, label=f"PR-AUC = {ap:.4f}")
    ax.axhline(baseline, ls="--", color="grey",
               label=f"Chance = {baseline:.4f}")
    ax.plot([op_r], [op_p], "o", ms=12, color="#C44E52",
            label=f"Operating point (t={threshold:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve (test set)")
    ax.legend()
    paths.append(_save_fig(fig, "pr_curve.png", config))

    # Threshold vs business cost (validation sweep).
    fig, ax1 = plt.subplots(figsize=(11, 6.5))
    ax1.plot(sweep["threshold"], sweep["cost"], lw=2.5, color="#C44E52")
    ax1.axvline(threshold, ls="--", color="#333",
                label=f"Chosen threshold {threshold:.3f}")
    ax1.axvline(0.5, ls=":", color="grey", label="Default 0.5")
    ax1.set(xlabel="Decision threshold", ylabel="Business cost ($, validation)",
            title="Business Cost vs Threshold")
    ax1.legend()
    paths.append(_save_fig(fig, "threshold_cost_curve.png", config))

    logger.info("Saved %d evaluation plots", len(paths))
    return paths


def plot_feature_importance(pipe: Any, feature_names: list[str],
                            config: dict[str, Any], top_n: int = 15) -> Path | None:
    """Bar chart of native feature importances (trees) or |coef| (linear)."""
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_).ravel()
    else:
        logger.warning("Model has no native importances; skipping plot")
        return None
    order = np.argsort(imp)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh([feature_names[i] for i in order][::-1], imp[order][::-1],
            color="#4C72B0")
    ax.set(title=f"Top {top_n} Feature Importances ({type(model).__name__})",
           xlabel="Importance")
    return _save_fig(fig, "feature_importance.png", config)


def evaluate_on_test(y_test: np.ndarray, proba_test: np.ndarray,
                     amounts_test: np.ndarray, threshold: float,
                     config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Final, single-shot evaluation on the held-out test set.

    Compares default vs tuned threshold and quantifies business cost for
    three policies: no model, default threshold, tuned threshold.

    Returns:
        Dict with metric blocks and the classification report string.
    """
    if config is None:
        config = load_config()
    fp_cost = config["threshold"]["fp_cost"]

    result: dict[str, Any] = {
        "default_threshold": compute_metrics(y_test, proba_test, 0.5),
        "tuned_threshold": compute_metrics(y_test, proba_test, threshold),
        "business_cost": {
            "no_model_cost": float(amounts_test[y_test == 1].sum()),
            "default_cost": business_cost(
                y_test, (proba_test >= 0.5).astype(int), amounts_test, fp_cost),
            "tuned_cost": business_cost(
                y_test, (proba_test >= threshold).astype(int), amounts_test, fp_cost),
            "fp_unit_cost": fp_cost,
        },
    }
    nm = result["business_cost"]["no_model_cost"]
    result["business_cost"]["savings_vs_no_model_pct"] = round(
        100 * (1 - result["business_cost"]["tuned_cost"] / nm), 1)
    result["classification_report"] = classification_report(
        y_test, (proba_test >= threshold).astype(int),
        target_names=["Genuine", "Fraud"], digits=4)
    save_json({k: v for k, v in result.items() if k != "classification_report"},
              f"{config['paths']['artifacts_dir']}/test_evaluation.json")
    logger.info("Test evaluation saved. Tuned threshold metrics: %s",
                {k: round(v, 4) if isinstance(v, float) else v
                 for k, v in result["tuned_threshold"].items()})
    return result
