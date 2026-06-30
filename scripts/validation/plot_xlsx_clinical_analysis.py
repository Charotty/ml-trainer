#!/usr/bin/env python3
"""Analysis plots + written metrics for clinical_xlsx_extended validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "validation"))

from common import (  # noqa: E402
    TARGET_COLUMNS,
    build_or_load_predictor,
    predict_df,
    vector_norm,
)
from plot_harmonized_analysis import (  # noqa: E402
    SHORT,
    TARGETS,
    load_predictions,
    plot_error_histograms,
    plot_mae_r2,
    plot_residuals,
    plot_scatter_true_pred,
    plot_summary_comparison,
    plot_train_composition,
    plot_within_threshold,
    plot_worst_cases,
    vector_errors,
)
from src.features.phase1_schema import normalize_dataframe  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402

RUN_ID_DEFAULT = "clinical_xlsx_extended_20260629"
BASELINE_RUN = "harmonized_extended_eval_20260610"
MODEL_PATH = ROOT / "models" / "adaptive_ensemble_xlsx_extended.pkl"
VAL_PATH = ROOT / "data" / "processed_xlsx" / "validation.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XLSX clinical model analysis plots")
    p.add_argument("--run-id", default=RUN_ID_DEFAULT)
    p.add_argument("--baseline-run-id", default=BASELINE_RUN)
    p.add_argument("--model", type=Path, default=MODEL_PATH)
    p.add_argument("--val-csv", type=Path, default=VAL_PATH)
    return p.parse_args()


def _plots_dir(run_dir: Path) -> Path:
    d = run_dir / "plots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def compute_metrics(df: pd.DataFrame) -> dict:
    per = []
    for tgt in TARGET_COLUMNS:
        y = df[tgt].astype(float)
        p = df[f"pred_{tgt}"].astype(float)
        per.append(
            {
                "target": tgt,
                "mae_mm": float(mean_absolute_error(y, p)),
                "r2": float(r2_score(y, p)) if len(y) > 1 else float("nan"),
                "count": int(len(y)),
            }
        )
    per_df = pd.DataFrame(per)
    left_xyz = df[[f"kidney_left_delta_{a}" for a in "xyz"]].astype(float).to_numpy()
    right_xyz = df[[f"kidney_right_delta_{a}" for a in "xyz"]].astype(float).to_numpy()
    pl = df[[f"pred_kidney_left_delta_{a}" for a in "xyz"]].astype(float).to_numpy()
    pr = df[[f"pred_kidney_right_delta_{a}" for a in "xyz"]].astype(float).to_numpy()
    vl, vr = vector_norm(left_xyz - pl, right_xyz - pr)
    axis_err = (
        df[TARGET_COLUMNS].astype(float)
        - df[[f"pred_{c}" for c in TARGET_COLUMNS]].astype(float)
    ).abs()
    return {
        "mae_avg_mm": float(per_df["mae_mm"].mean()),
        "r2_avg": float(per_df["r2"].mean()),
        "vector_error_left_mae_mm": float(vl.mean()),
        "vector_error_right_mae_mm": float(vr.mean()),
        "vector_error_mean_mae_mm": float((vl.mean() + vr.mean()) / 2),
        "within_5mm_ratio": float((axis_err <= 5).mean().mean()),
        "within_10mm_ratio": float((axis_err <= 10).mean().mean()),
        "sample_count": int(len(df)),
        "per_target": per_df,
    }


def predict_subset(bundle, df: pd.DataFrame) -> pd.DataFrame:
    pred = predict_df(bundle, df)
    out = df.copy()
    for col in TARGET_COLUMNS:
        out[f"pred_{col}"] = pred[col].values
    return out


def plot_old_vs_new(
    plots_dir: Path,
    new_metrics: pd.DataFrame,
    old_metrics: pd.DataFrame | None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SHORT))
    w = 0.35
    new_mae = new_metrics.set_index("target").loc[TARGETS, "mae_mm"].values
    ax.bar(x - w / 2, new_mae, w, label="XLSX 87 (clinical_xlsx_extended)", color="#2563eb")
    if old_metrics is not None:
        old_mae = old_metrics.set_index("target").loc[TARGETS, "mae_mm"].values
        ax.bar(x + w / 2, old_mae, w, label="Legacy 50 (harmonized_extended)", color="#94a3b8")
    ax.axhline(5, color="#ef4444", ls="--", lw=1, alpha=0.7, label="5 mm clinical target")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT)
    ax.set_ylabel("MAE (mm)")
    ax.set_title("Сравнение точности: новая когорта (87) vs старая (50)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "10_old50_vs_new87_mae.png", dpi=150)
    plt.close(fig)


def plot_holdout_split(
    plots_dir: Path,
    full_metrics: dict,
    val_metrics: dict,
) -> None:
    keys = ["mae_avg_mm", "vector_error_mean_mae_mm", "within_5mm_ratio", "within_10mm_ratio"]
    labels = ["MAE средн.", "Вектор MAE", "≤5 mm", "≤10 mm"]
    x = np.arange(len(keys))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    full_vals = [full_metrics[k] for k in keys]
    val_vals = [val_metrics[k] for k in keys]
    ax.bar(x - w / 2, full_vals, w, label=f"Все 87 ({int(full_metrics['sample_count'])})", color="#0ea5e9")
    ax.bar(x + w / 2, val_vals, w, label=f"Holdout val ({int(val_metrics['sample_count'])})", color="#f97316")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Метрики: полная когорта vs честный holdout (18 пациентов)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "11_full87_vs_holdout18.png", dpi=150)
    plt.close(fig)


def plot_error_by_side(df: pd.DataFrame, plots_dir: Path) -> None:
    pred_cols = [f"pred_{c}" for c in TARGET_COLUMNS]
    err = df[TARGET_COLUMNS].astype(float) - df[pred_cols].astype(float)
    left = err[[c for c in TARGET_COLUMNS if "left" in c]].abs().mean(axis=0).values
    right = err[[c for c in TARGET_COLUMNS if "right" in c]].abs().mean(axis=0).values
    labels = ["Δx", "Δy", "Δz"]
    x = np.arange(3)
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, left, w, label="Левая почка", color="#3b82f6")
    ax.bar(x + w / 2, right, w, label="Правая почка", color="#22c55e")
    ax.axhline(5, color="#ef4444", ls="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Средняя |ошибка| (mm)")
    ax.set_title("Где ошибка: по осям и сторонам")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "12_error_by_side_and_axis.png", dpi=150)
    plt.close(fig)


def plot_z_focus(df: pd.DataFrame, plots_dir: Path) -> None:
    z_cols = [c for c in TARGET_COLUMNS if c.endswith("_z")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, side, color in zip(axes, ("left", "right"), ("#3b82f6", "#22c55e")):
        tgt = f"kidney_{side}_delta_z"
        y = df[tgt].astype(float)
        p = df[f"pred_{tgt}"].astype(float)
        ax.scatter(y, p, c=color, alpha=0.75, edgecolors="white", s=45)
        lim = max(y.abs().max(), p.abs().max(), 10) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, alpha=0.5)
        mae = mean_absolute_error(y, p)
        ax.set_title(f"{'Левая' if side == 'left' else 'Правая'} Δz  (MAE={mae:.1f} mm)")
        ax.set_xlabel("Истина (mm)")
        ax.set_ylabel("Предсказание (mm)")
        ax.grid(alpha=0.25)
    fig.suptitle("Проблемная ось Z — наибольшие ошибки", y=1.02)
    fig.tight_layout()
    fig.savefig(plots_dir / "13_z_axis_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_worst_named(worst: pd.DataFrame, preds: pd.DataFrame, plots_dir: Path, top_n: int = 10) -> None:
    name_map = preds.set_index("case_id")["full_name"].to_dict() if "full_name" in preds.columns else {}
    w = worst.head(top_n).iloc[::-1].copy()
    labels = [f"{name_map.get(cid, cid)}" for cid in w["case_id"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(w))
    ax.barh(y_pos, w["vector_error_mean_mm"], color="#dc2626", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Средняя векторная ошибка (mm)")
    ax.set_title("Топ худших случаев (по фамилии)")
    ax.axvline(5, color="gray", ls="--", alpha=0.6, label="5 mm")
    ax.axvline(10, color="#f59e0b", ls=":", alpha=0.6, label="10 mm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "14_worst_cases_named.png", dpi=150)
    plt.close(fig)


def plot_accuracy_pie(df: pd.DataFrame, plots_dir: Path) -> None:
    ve = vector_errors(df)
    bins = ["≤5 mm", "5–10 mm", ">10 mm"]
    counts = [
        (ve["vec_err_mean_mm"] <= 5).sum(),
        ((ve["vec_err_mean_mm"] > 5) & (ve["vec_err_mean_mm"] <= 10)).sum(),
        (ve["vec_err_mean_mm"] > 10).sum(),
    ]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(
        counts,
        labels=[f"{b}\n({c})" for b, c in zip(bins, counts)],
        autopct="%1.0f%%",
        colors=["#22c55e", "#fbbf24", "#ef4444"],
        startangle=90,
    )
    ax.set_title(f"Клиническая точность (n={len(df)})\nпо средней векторной ошибке")
    fig.tight_layout()
    fig.savefig(plots_dir / "15_accuracy_tiers.png", dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    run_dir = ROOT / "results" / "validation_runs" / args.run_id
    plots_dir = _plots_dir(run_dir)

    preds = load_predictions(run_dir)
    preds = vector_errors(preds)
    ext_metrics = pd.read_csv(run_dir / "metrics" / "metrics_per_target.csv")
    ext_summary = pd.read_csv(run_dir / "metrics" / "metrics_summary.csv")

    baseline_dir = ROOT / "results" / "validation_runs" / args.baseline_run_id
    base_metrics = None
    base_summary = None
    if (baseline_dir / "metrics" / "metrics_per_target.csv").exists():
        base_metrics = pd.read_csv(baseline_dir / "metrics" / "metrics_per_target.csv")
        base_summary = pd.read_csv(baseline_dir / "metrics" / "metrics_summary.csv")

    # Standard plots (reuse harmonized module)
    plot_mae_r2(ext_metrics, plots_dir)
    plot_scatter_true_pred(preds, plots_dir)
    plot_error_histograms(preds, plots_dir)
    worst_path = run_dir / "metrics" / "worst_cases.csv"
    worst = pd.read_csv(worst_path) if worst_path.exists() else preds.nlargest(10, "vec_err_mean_mm")
    plot_worst_cases(worst, plots_dir)
    plot_within_threshold(preds, plots_dir)
    plot_summary_comparison(ext_summary, base_summary, plots_dir)
    plot_residuals(preds, plots_dir)

    train_summary_path = run_dir / "metrics" / "train_source_summary.json"
    if train_summary_path.exists():
        plot_train_composition(json.loads(train_summary_path.read_text(encoding="utf-8")), plots_dir)

    # XLSX-specific plots
    plot_old_vs_new(plots_dir, ext_metrics, base_metrics)
    plot_error_by_side(preds, plots_dir)
    plot_z_focus(preds, plots_dir)
    plot_worst_named(worst, preds, plots_dir)
    plot_accuracy_pie(preds, plots_dir)

    # Honest holdout on validation.csv (18 patients not seen in training split)
    full_metrics = compute_metrics(preds)
    val_metrics = None
    if args.val_csv.exists() and args.model.exists():
        val_df = normalize_dataframe(pd.read_csv(args.val_csv))
        bundle, _, _ = build_or_load_predictor(
            val_df, args.model, test_size=0.2, seed=42, holdout_eval=True
        )
        val_pred = predict_subset(bundle, val_df)
        val_metrics = compute_metrics(val_pred)
        plot_holdout_split(plots_dir, full_metrics, val_metrics)
        val_pred.to_csv(run_dir / "predictions" / "holdout_val18_predictions.csv", index=False)
        val_metrics["per_target"].to_csv(
            run_dir / "metrics" / "metrics_per_target_holdout18.csv", index=False
        )

    report = {
        "model": str(args.model),
        "full_cohort_n": full_metrics["sample_count"],
        "full_cohort": {k: full_metrics[k] for k in full_metrics if k != "per_target"},
        "holdout_val_n": val_metrics["sample_count"] if val_metrics else None,
        "holdout_val": (
            {k: val_metrics[k] for k in val_metrics if k != "per_target"} if val_metrics else None
        ),
        "baseline_50_patient_mae": (
            float(base_summary.set_index("metric").loc["mae_avg_mm", "value"])
            if base_summary is not None
            else None
        ),
        "worst_cases": worst.head(5).to_dict(orient="records"),
        "weakest_axes": ext_metrics.sort_values("mae_mm", ascending=False).head(2)["target"].tolist(),
    }
    report_path = run_dir / "metrics" / "analysis_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[ok] plots -> {plots_dir}")
    for p in sorted(plots_dir.glob("*.png")):
        print(f"  {p.name}")
    print(f"[ok] report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
