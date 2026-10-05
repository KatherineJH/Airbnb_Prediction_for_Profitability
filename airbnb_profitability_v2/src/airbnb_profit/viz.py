"""Plot helpers shared by the report stage and the notebook."""
from __future__ import annotations

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")


def close(fig):
    plt.close(fig)


def plot_target_distributions(frames: dict[str, pd.DataFrame], col: str = "target_log_revenue"):
    n = len(frames)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.2 * ((n + 1) // 2), 6), sharex=True)
    for ax, (city, df) in zip(np.ravel(axes), frames.items()):
        ax.hist(df[col], bins=40, color="#4c72b0")
        ax.set_title(city)
    fig.suptitle("Target: log(1 + estimated annual revenue)")
    fig.tight_layout()
    return fig


def plot_tuning_history(results: list[dict], city: str):
    fig, axes = plt.subplots(1, len({r["model"] for r in results}), figsize=(4 * len({r["model"] for r in results}), 3.4), squeeze=False)
    for ax, model in zip(axes[0], sorted({r["model"] for r in results})):
        for r in (x for x in results if x["model"] == model):
            v = pd.Series([t["value"] for t in r["trials"]])
            ax.plot(v.cummin(), label=r["sampler"])
        ax.set_title(f"{city} - {model}")
        ax.set_xlabel("trial")
        ax.set_ylabel("best CV RMSE (log)")
        ax.legend()
    fig.tight_layout()
    return fig


def plot_metric_heatmap(metrics: pd.DataFrame, metric: str = "rmse_log"):
    wide = metrics.pivot(index="city", columns="model", values=metric)
    fig, ax = plt.subplots(figsize=(1.1 * wide.shape[1] + 2, 0.5 * wide.shape[0] + 2))
    sns.heatmap(wide, annot=True, fmt=".2f", cmap="viridis_r" if "rmse" in metric else "viridis", ax=ax)
    ax.set_title(f"Test {metric} (held-out hosts)")
    return fig


def plot_model_comparison(metrics: pd.DataFrame):
    order = metrics.groupby("model")["rmse_log"].median().sort_values().index
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=metrics, x="model", y="rmse_log", order=order, ax=axes[0], color="#8fb1d6")
    sns.stripplot(data=metrics, x="model", y="rmse_log", order=order, ax=axes[0], color="k", size=3)
    axes[0].set_title("Test RMSE (log scale) across cities - lower is better")
    sns.boxplot(data=metrics, x="model", y="r2_log", order=order, ax=axes[1], color="#9ccf9a")
    sns.stripplot(data=metrics, x="model", y="r2_log", order=order, ax=axes[1], color="k", size=3)
    axes[1].set_title("Test R^2 (log scale) across cities - higher is better")
    for ax in axes:
        ax.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    return fig


def plot_importance(importances: dict[str, pd.DataFrame], top_k: int = 10):
    n = len(importances)
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    for ax, (city, d) in zip(axes.ravel(), importances.items()):
        top = d.head(top_k).iloc[::-1]
        ax.barh(top["feature"], top["importance"], xerr=top["std"], color="#c44e52")
        ax.set_title(city)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Permutation importance (increase in test RMSE when a raw column is shuffled)")
    fig.tight_layout()
    return fig


def plot_rank_corr(rank: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(rank, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Spearman correlation of importance vectors between cities")
    return fig


def plot_pred_vs_true(pred_df: pd.DataFrame, model: str, city: str):
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.scatter(pred_df["y_true_log"], pred_df[model], s=6, alpha=.4)
    lim = [min(pred_df["y_true_log"].min(), pred_df[model].min()), max(pred_df["y_true_log"].max(), pred_df[model].max())]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("true log revenue")
    ax.set_ylabel("predicted")
    ax.set_title(f"{city} - {model} (test)")
    return fig


def report_figures(run, metrics: pd.DataFrame, imps: dict, rank: pd.DataFrame) -> dict:
    figs = {"metric_heatmap_rmse_log": plot_metric_heatmap(metrics, "rmse_log"),
            "metric_heatmap_r2_log": plot_metric_heatmap(metrics, "r2_log"),
            "model_comparison": plot_model_comparison(metrics),
            "importance_by_city": plot_importance(imps, run.cfg["evaluation"]["top_k_features"])}
    if not rank.empty:
        figs["importance_rank_corr"] = plot_rank_corr(rank)
    return figs
