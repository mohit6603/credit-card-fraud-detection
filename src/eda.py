"""Exploratory data analysis: professional plots saved to reports/images.

Each function saves one figure and returns the path, so notebooks stay pure
orchestration and every visual lands in the report folder for documentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend - safe for scripts and CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import get_logger, load_config, resolve_path

logger = get_logger(__name__)

sns.set_theme(style="whitegrid", context="talk", palette="deep")
GENUINE_COLOR, FRAUD_COLOR = "#4C72B0", "#C44E52"


def _save(fig: plt.Figure, name: str, config: dict[str, Any]) -> Path:
    """Save a figure to the configured images directory and close it."""
    out_dir = resolve_path(config["paths"]["images_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", path.name)
    return path


def plot_class_distribution(df: pd.DataFrame, config: dict[str, Any]) -> Path:
    """Bar chart of genuine vs fraud counts with a log-scaled inset of rates."""
    target = config["data"]["target"]
    counts = df[target].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(["Genuine (0)", "Fraud (1)"], counts.values,
                  color=[GENUINE_COLOR, FRAUD_COLOR])
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:,}\n({100 * val / len(df):.3f}%)",
                ha="center", va="bottom", fontsize=13)
    ax.set_yscale("log")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Class Distribution - Extreme Imbalance")
    return _save(fig, "class_distribution.png", config)


def plot_amount_distribution(df: pd.DataFrame, config: dict[str, Any]) -> Path:
    """Transaction amount distributions per class, raw and log-scaled."""
    target = config["data"]["target"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for cls, color, label in [(0, GENUINE_COLOR, "Genuine"), (1, FRAUD_COLOR, "Fraud")]:
        sub = df.loc[df[target] == cls, "Amount"]
        axes[0].hist(sub, bins=80, range=(0, 500), alpha=0.6, density=True,
                     color=color, label=label)
        axes[1].hist(np.log1p(sub), bins=60, alpha=0.6, density=True,
                     color=color, label=label)
    axes[0].set(title="Amount (0-500 range)", xlabel="Amount ($)", ylabel="Density")
    axes[1].set(title="log1p(Amount) - all transactions", xlabel="log1p(Amount)")
    for ax in axes:
        ax.legend()
    fig.suptitle("Transaction Amount by Class", y=1.02)
    return _save(fig, "amount_distribution.png", config)


def plot_time_distribution(df: pd.DataFrame, config: dict[str, Any]) -> Path:
    """Fraud rate by hour of day - shows the night-time fraud spike."""
    target = config["data"]["target"]
    hours = ((df["Time"] // 3600) % 24).astype(int)
    tmp = pd.DataFrame({"Hour": hours, "Class": df[target]})
    rate = tmp.groupby("Hour")["Class"].mean() * 100
    counts = tmp.groupby("Hour")["Class"].sum()

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.bar(rate.index, rate.values, color=FRAUD_COLOR, alpha=0.75)
    ax1.set(xlabel="Hour of day", ylabel="Fraud rate (%)",
            title="Fraud Rate by Hour of Day", xticks=range(24))
    ax2 = ax1.twinx()
    ax2.plot(counts.index, counts.values, "o-", color="#333333", lw=2, label="Fraud count")
    ax2.set_ylabel("Fraud count")
    ax2.legend(loc="upper right")
    return _save(fig, "time_distribution.png", config)


def plot_correlation_heatmap(df: pd.DataFrame, config: dict[str, Any]) -> Path:
    """Correlation heatmap; V-columns are PCA outputs so are mutually ~0."""
    fig, ax = plt.subplots(figsize=(16, 13))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax,
                cbar_kws={"shrink": 0.7}, xticklabels=True, yticklabels=True)
    ax.set_title("Correlation Heatmap (V1-V28 are orthogonal PCA components)")
    ax.tick_params(labelsize=9)
    return _save(fig, "correlation_heatmap.png", config)


def plot_fraud_vs_genuine_features(df: pd.DataFrame, config: dict[str, Any],
                                   top_n: int = 8) -> Path:
    """Distributions of the V-features most correlated with the target."""
    target = config["data"]["target"]
    v_cols = [c for c in df.columns if c.startswith("V")]
    corr = df[v_cols + [target]].corr()[target].drop(target).abs()
    top = corr.sort_values(ascending=False).head(top_n).index.tolist()

    rows = (top_n + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(20, 4.5 * rows))
    for ax, col in zip(axes.ravel(), top):
        for cls, color, label in [(0, GENUINE_COLOR, "Genuine"), (1, FRAUD_COLOR, "Fraud")]:
            sns.kdeplot(df.loc[df[target] == cls, col], ax=ax, fill=True,
                        alpha=0.45, color=color, label=label, warn_singular=False)
        ax.set_title(f"{col} (|corr|={corr[col]:.2f})", fontsize=13)
        ax.legend(fontsize=10)
    for ax in axes.ravel()[len(top):]:
        ax.set_visible(False)
    fig.suptitle("Most Discriminative Features: Fraud vs Genuine", y=1.01)
    return _save(fig, "fraud_vs_genuine_features.png", config)


def plot_outlier_analysis(df: pd.DataFrame, config: dict[str, Any]) -> Path:
    """Boxplots of Amount per class - fraud amounts are small but heavy-tailed."""
    target = config["data"]["target"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    data = [df.loc[df[target] == 0, "Amount"], df.loc[df[target] == 1, "Amount"]]
    axes[0].boxplot(data, tick_labels=["Genuine", "Fraud"], showfliers=True)
    axes[0].set(title="Amount Boxplot (with outliers)", ylabel="Amount ($)", yscale="log")
    axes[0].set_ylim(bottom=0.01)
    stats = df.groupby(target)["Amount"].describe()[["mean", "50%", "max"]]
    axes[1].axis("off")
    table = axes[1].table(
        cellText=np.round(stats.values, 2),
        rowLabels=["Genuine", "Fraud"], colLabels=["Mean", "Median", "Max"],
        loc="center", cellLoc="center")
    table.scale(1, 2.2)
    table.set_fontsize(14)
    axes[1].set_title("Amount Summary by Class")
    return _save(fig, "outlier_analysis.png", config)


def run_eda(df: pd.DataFrame, config: dict[str, Any] | None = None) -> list[Path]:
    """Generate the full EDA plot suite and return saved paths."""
    if config is None:
        config = load_config()
    paths = [
        plot_class_distribution(df, config),
        plot_amount_distribution(df, config),
        plot_time_distribution(df, config),
        plot_correlation_heatmap(df, config),
        plot_fraud_vs_genuine_features(df, config),
        plot_outlier_analysis(df, config),
    ]
    logger.info("EDA complete: %d plots saved", len(paths))
    return paths
