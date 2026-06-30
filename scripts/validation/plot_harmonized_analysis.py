#!/usr/bin/env python3
"""Generate analysis plots for harmonized / extended training validation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TARGETS = [
    "kidney_left_delta_x",
    "kidney_left_delta_y",
    "kidney_left_delta_z",
    "kidney_right_delta_x",
    "kidney_right_delta_y",
    "kidney_right_delta_z",
]
SHORT = ["L Δx", "L Δy", "L Δz", "R Δx", "R Δy", "R Δz"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot harmonized training analysis charts")
    p.add_argument("--run-id", default="harmonized_extended_eval_20260610")
    p.add_argument("--baseline-run-id", default="harmonized_train_eval_20260610")
    p.add_argument("--out-dir", default="results/validation_runs")
    p.add_argument("--train-summary", default=None, help="train_source_summary.json path")
    return p.parse_args()


def _run_dir(out_dir: Path, run_id: str) -> Path:
    return out_dir / run_id


def _plots_dir(run_dir: Path) -> Path:
    d = run_dir / "plots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_predictions(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "predictions" / "evaluation_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions: {path}")
    return pd.read_csv(path)


def vector_errors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for side in ("left", "right"):
        t_cols = [f"kidney_{side}_delta_{a}" for a in ("x", "y", "z")]
        p_cols = [f"pred_{c}" for c in t_cols]
        t = out[t_cols].astype(float).to_numpy()
        p = out[p_cols].astype(float).to_numpy()
        out[f"vec_err_{side}_mm"] = np.linalg.norm(t - p, axis=1)
    out["vec_err_mean_mm"] = (out["vec_err_left_mm"] + out["vec_err_right_mm"]) / 2.0
    return out


def plot_model_comparison(
    plots_dir: Path,
    extended_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame | None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SHORT))
    w = 0.35 if baseline_metrics is not None else 0.6
    ext_mae = extended_metrics.set_index("target").loc[TARGETS, "mae_mm"].values
    bars = ax.bar(x - (w / 2 if baseline_metrics is not None else 0), ext_mae, w, label="Extended (409 train)", color="#2563eb")
    if baseline_metrics is not None:
        base_mae = baseline_metrics.set_index("target").loc[TARGETS, "mae_mm"].values
        ax.bar(x + w / 2, base_mae, w, label="Vybor-only (40 train)", color="#94a3b8")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT)
    ax.set_ylabel("MAE (mm)")
    ax.set_title("Holdout Vybor (n=50): MAE per target")
    ax.axhline(5.0, color="#ef4444", ls="--", lw=1, alpha=0.7, label="5 mm")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "01_model_comparison_mae.png", dpi=150)
    plt.close(fig)


def plot_mae_r2(extended_metrics: pd.DataFrame, plots_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SHORT))
    mae = extended_metrics.set_index("target").loc[TARGETS, "mae_mm"].values
    r2 = extended_metrics.set_index("target").loc[TARGETS, "r2"].values
    ax1.bar(x, mae, color="#0ea5e9", alpha=0.85, label="MAE")
    ax1.set_ylabel("MAE (mm)", color="#0369a1")
    ax1.set_xticks(x)
    ax1.set_xticklabels(SHORT)
    ax2 = ax1.twinx()
    ax2.plot(x, r2, "o-", color="#f97316", lw=2, markersize=8, label="R²")
    ax2.set_ylabel("R²", color="#c2410c")
    ax2.axhline(0, color="gray", lw=0.8)
    ax1.set_title("Extended model — per-target holdout metrics")
    fig.tight_layout()
    fig.savefig(plots_dir / "02_per_target_mae_r2.png", dpi=150)
    plt.close(fig)


def plot_scatter_true_pred(df: pd.DataFrame, plots_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, tgt, label in zip(axes.ravel(), TARGETS, SHORT):
        y = df[tgt].astype(float)
        p = df[f"pred_{tgt}"].astype(float)
        ax.scatter(y, p, alpha=0.75, s=40, c="#2563eb", edgecolors="white", linewidths=0.4)
        lim = max(y.abs().max(), p.abs().max()) * 1.1
        lim = max(lim, 5)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, alpha=0.5)
        ax.set_xlabel(f"True {label}")
        ax.set_ylabel(f"Pred {label}")
        ax.set_title(label)
        ax.grid(alpha=0.25)
    fig.suptitle("True vs predicted δ (Vybor holdout)", y=1.02)
    fig.tight_layout()
    fig.savefig(plots_dir / "03_true_vs_pred_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_histograms(df: pd.DataFrame, plots_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, title in zip(
        axes,
        ["vec_err_left_mm", "vec_err_right_mm", "vec_err_mean_mm"],
        ["Left kidney ‖Δ‖ error", "Right kidney ‖Δ‖ error", "Mean vector error"],
    ):
        ax.hist(df[col], bins=12, color="#6366f1", edgecolor="white", alpha=0.9)
        ax.axvline(df[col].median(), color="#ef4444", ls="--", label=f"median={df[col].median():.1f} mm")
        ax.axvline(5, color="#22c55e", ls=":", label="5 mm")
        ax.set_xlabel("Error (mm)")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Vector error distribution (holdout)")
    fig.tight_layout()
    fig.savefig(plots_dir / "04_vector_error_histogram.png", dpi=150)
    plt.close(fig)


def plot_worst_cases(worst: pd.DataFrame, plots_dir: Path, top_n: int = 10) -> None:
    w = worst.head(top_n).iloc[::-1]
    labels = w["case_id"].astype(str) if "case_id" in w.columns else w.index.astype(str)
    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = np.arange(len(w))
    ax.barh(y_pos, w["vector_error_mean_mm"], color="#dc2626", alpha=0.85, label="mean")
    ax.barh(y_pos, w["vector_error_left_mm"], color="#3b82f6", alpha=0.4, height=0.4, label="left")
    ax.barh(y_pos + 0.15, w["vector_error_right_mm"], color="#22c55e", alpha=0.4, height=0.4, label="right")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Vector error (mm)")
    ax.set_title(f"Top-{top_n} worst cases (holdout)")
    ax.axvline(5, color="gray", ls="--", alpha=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(plots_dir / "05_worst_cases.png", dpi=150)
    plt.close(fig)


def plot_train_composition(summary: dict, plots_dir: Path) -> None:
    by_src = summary.get("by_source", {})
    by_lq = summary.get("by_label_quality", {})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if by_src:
        axes[0].pie(
            by_src.values(),
            labels=[f"{k}\n({v})" for k, v in by_src.items()],
            autopct="%1.0f%%",
            colors=["#8b5cf6", "#06b6d4", "#f59e0b"],
            startangle=90,
        )
        axes[0].set_title(f"Train rows (n={summary.get('total_train_rows', '?')}) by source")
    if by_lq:
        axes[1].bar(
            list(by_lq.keys()),
            list(by_lq.values()),
            color=["#22c55e", "#a855f7", "#38bdf8"],
        )
        axes[1].set_title("By label quality")
        axes[1].set_ylabel("Rows")
        axes[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(plots_dir / "06_train_composition.png", dpi=150)
    plt.close(fig)


def plot_within_threshold(df: pd.DataFrame, plots_dir: Path) -> None:
    ratios_5, ratios_10 = [], []
    for tgt in TARGETS:
        err = (df[tgt].astype(float) - df[f"pred_{tgt}"].astype(float)).abs()
        ratios_5.append((err <= 5).mean())
        ratios_10.append((err <= 10).mean())
    x = np.arange(len(SHORT))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, ratios_5, w, label="≤ 5 mm", color="#22c55e")
    ax.bar(x + w / 2, ratios_10, w, label="≤ 10 mm", color="#86efac")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of cases")
    ax.set_title("Clinical accuracy thresholds per axis")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "07_within_threshold_per_axis.png", dpi=150)
    plt.close(fig)


def plot_summary_comparison(
    ext_summary: pd.DataFrame,
    base_summary: pd.DataFrame | None,
    plots_dir: Path,
) -> None:
    keys = ["mae_avg_mm", "vector_error_mean_mae_mm", "within_5mm_ratio", "within_10mm_ratio"]
    labels = ["MAE avg", "Vector MAE", "≤5 mm", "≤10 mm"]
    ext = ext_summary.set_index("metric")["value"]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.35 if base_summary is not None else 0.5
    ext_vals = [ext.get(k, np.nan) for k in keys]
    ax.bar(x - (w / 2 if base_summary is not None else 0), ext_vals, w, label="Extended", color="#2563eb")
    if base_summary is not None:
        base = base_summary.set_index("metric")["value"]
        base_vals = [base.get(k, np.nan) for k in keys]
        ax.bar(x + w / 2, base_vals, w, label="Vybor-only", color="#94a3b8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Aggregate holdout metrics comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "08_aggregate_comparison.png", dpi=150)
    plt.close(fig)


def plot_residuals(df: pd.DataFrame, plots_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, tgt, label in zip(axes.ravel(), TARGETS, SHORT):
        p = df[f"pred_{tgt}"].astype(float)
        res = df[tgt].astype(float) - p
        ax.scatter(p, res, alpha=0.7, s=35, c="#7c3aed")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel(f"Predicted {label}")
        ax.set_ylabel("Residual (mm)")
        ax.set_title(label)
        ax.grid(alpha=0.25)
    fig.suptitle("Residuals vs prediction", y=1.02)
    fig.tight_layout()
    fig.savefig(plots_dir / "09_residuals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    run_dir = _run_dir(out_dir, args.run_id)
    plots_dir = _plots_dir(run_dir)

    ext_metrics = pd.read_csv(run_dir / "metrics" / "metrics_per_target.csv")
    ext_summary = pd.read_csv(run_dir / "metrics" / "metrics_summary.csv")
    baseline_dir = _run_dir(out_dir, args.baseline_run_id) if args.baseline_run_id else None
    base_metrics = None
    base_summary = None
    if baseline_dir and (baseline_dir / "metrics" / "metrics_per_target.csv").exists():
        base_metrics = pd.read_csv(baseline_dir / "metrics" / "metrics_per_target.csv")
        base_summary = pd.read_csv(baseline_dir / "metrics" / "metrics_summary.csv")

    df = vector_errors(load_predictions(run_dir))
    worst_path = run_dir / "metrics" / "worst_cases.csv"
    worst = pd.read_csv(worst_path) if worst_path.exists() else df.nlargest(10, "vec_err_mean_mm")

    train_summary_path = (
        Path(args.train_summary)
        if args.train_summary
        else run_dir / "metrics" / "train_source_summary.json"
    )

    plot_model_comparison(plots_dir, ext_metrics, base_metrics)
    plot_mae_r2(ext_metrics, plots_dir)
    plot_scatter_true_pred(df, plots_dir)
    plot_error_histograms(df, plots_dir)
    plot_worst_cases(worst, plots_dir)
    plot_within_threshold(df, plots_dir)
    plot_summary_comparison(ext_summary, base_summary, plots_dir)
    plot_residuals(df, plots_dir)

    if train_summary_path.exists():
        summary = json.loads(train_summary_path.read_text(encoding="utf-8"))
        plot_train_composition(summary, plots_dir)

    index = sorted(plots_dir.glob("*.png"))
    print(f"[ok] {len(index)} plots -> {plots_dir}")
    for p in index:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
