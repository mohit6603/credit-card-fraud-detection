"""Model explainability: permutation importance and SHAP values.

Both methods are computed on a sample of the (scaled) test set:
- Permutation importance is model-agnostic ground truth for "does the model
  actually rely on this feature" - it measures metric drop when a feature
  is shuffled.
- SHAP attributes each individual prediction to features with a solid
  game-theoretic foundation; the summary plot shows global structure and
  direction of effects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.utils import get_logger, load_config, resolve_path

logger = get_logger(__name__)


def compute_permutation_importance(pipe: Any, X: pd.DataFrame, y: pd.Series,
                                   config: dict[str, Any],
                                   n_repeats: int = 5,
                                   max_rows: int = 20000) -> tuple[pd.DataFrame, Path]:
    """Permutation importance of the full pipeline on a test sample.

    Args:
        pipe: Fitted pipeline (scaler + model).
        X: Test features (unscaled - the pipeline scales internally).
        y: Test labels.
        config: Project configuration.
        n_repeats: Shuffles per feature.
        max_rows: Stratified-ish cap to keep runtime sane.

    Returns:
        (importance table sorted descending, saved plot path).
    """
    rng = np.random.RandomState(config["split"]["random_state"])
    if len(X) > max_rows:
        # keep every fraud, sample the rest - frauds carry the signal for AP
        fraud_idx = np.flatnonzero(y.to_numpy() == 1)
        gen_idx = np.flatnonzero(y.to_numpy() == 0)
        keep = np.concatenate([
            fraud_idx, rng.choice(gen_idx, max_rows - len(fraud_idx), replace=False)])
        X, y = X.iloc[keep], y.iloc[keep]

    result = permutation_importance(
        pipe, X, y, scoring="average_precision", n_repeats=n_repeats,
        random_state=config["split"]["random_state"], n_jobs=-1)
    table = (pd.DataFrame({"feature": X.columns,
                           "importance_mean": result.importances_mean,
                           "importance_std": result.importances_std})
             .sort_values("importance_mean", ascending=False)
             .reset_index(drop=True))

    top = table.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["importance_mean"],
            xerr=top["importance_std"], color="#4C72B0")
    ax.set(title="Permutation Importance (drop in PR-AUC when shuffled)",
           xlabel="Mean decrease in average precision")
    out = resolve_path(config["paths"]["images_dir"]) / "permutation_importance.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    table.to_csv(resolve_path(config["paths"]["artifacts_dir"]) / "permutation_importance.csv",
                 index=False)
    logger.info("Permutation importance done; top feature: %s", table.iloc[0]["feature"])
    return table, out


def compute_shap(pipe: Any, X: pd.DataFrame, config: dict[str, Any],
                 max_rows: int = 2000) -> Path | None:
    """SHAP summary (beeswarm) for the fitted model on a test sample.

    Uses TreeExplainer for tree ensembles and LinearExplainer for logistic
    regression. The pipeline's scaler is applied manually so SHAP sees the
    exact inputs the model sees, while plots keep original feature names.

    Returns:
        Path to the saved beeswarm plot, or None if SHAP fails.
    """
    import shap

    model = pipe.named_steps["model"]
    scaler = pipe.named_steps["scaler"]
    rng = np.random.RandomState(config["split"]["random_state"])
    sample = X.iloc[rng.choice(len(X), min(max_rows, len(X)), replace=False)]
    X_scaled = pd.DataFrame(scaler.transform(sample), columns=sample.columns)

    try:
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
            if isinstance(shap_values, list):  # sklearn RF returns per-class list
                shap_values = shap_values[1]
            if shap_values.ndim == 3:  # (rows, features, classes)
                shap_values = shap_values[:, :, 1]
        else:
            explainer = shap.LinearExplainer(model, X_scaled)
            shap_values = explainer.shap_values(X_scaled)
    except Exception as exc:
        logger.error("SHAP computation failed: %s", exc)
        return None

    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_scaled, show=False, max_display=15)
    out = resolve_path(config["paths"]["images_dir"]) / "shap_summary.png"
    plt.gcf().savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")

    mean_abs = pd.DataFrame({
        "feature": X_scaled.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    mean_abs.to_csv(resolve_path(config["paths"]["artifacts_dir"]) / "shap_importance.csv",
                    index=False)
    logger.info("SHAP done; top feature: %s", mean_abs.iloc[0]["feature"])
    return out


def run_explainability(pipe: Any, X_test: pd.DataFrame, y_test: pd.Series,
                       config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run both explainability methods and return artifact paths."""
    if config is None:
        config = load_config()
    perm_table, perm_path = compute_permutation_importance(pipe, X_test, y_test, config)
    shap_path = compute_shap(pipe, X_test, config)
    return {"permutation_top5": perm_table.head(5).to_dict("records"),
            "permutation_plot": str(perm_path),
            "shap_plot": str(shap_path) if shap_path else None}
