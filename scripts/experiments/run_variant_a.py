#!/usr/bin/env python3
"""Variant A experiment runner: nested CV + holdout + journal row."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.holdout import (  # noqa: E402
    DEFAULT_HOLDOUT_PATH,
    assert_no_holdout_leak,
    holdout_surname_keys,
)
from src.data.xlsx_displacement_parser import DEFAULT_OUTPUT_CSV, DEFAULT_XLSX_PATH  # noqa: E402
from src.features.na_trend_features import NaTrendStore  # noqa: E402
from src.features.phase1_schema import TARGET_NAMES, filter_any_kidney_targets, normalize_dataframe  # noqa: E402
from src.models.baselines import make_trainer_factory  # noqa: E402
from src.models.ensemble import AdaptiveEnsembleTrainer  # noqa: E402
from src.models.ensemble.serializer import build_runtime_payload  # noqa: E402
from src.models.nested_cv import (  # noqa: E402
    DEFAULT_REPEAT_SEEDS,
    evaluate_repeated_nested_cv,
    paired_compare_repeated,
)
from src.models.z_calibrator_oof import SideZCalibrator, fit_calibrator_oof_gated  # noqa: E402

JOURNAL_PATH = ROOT / "docs" / "EXPERIMENTS_VARIANT_A.md"
EXPERIMENTS_DIR = ROOT / "models" / "experiments"
BEST_STATE_PATH = EXPERIMENTS_DIR / "best_state.json"
Z_TARGETS = ["kidney_left_delta_z", "kidney_right_delta_z"]


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("ascii", errors="ignore").strip() or "unknown"
    except Exception:
        return "unknown"


def _load_best_state() -> dict[str, Any]:
    if BEST_STATE_PATH.is_file():
        return json.loads(BEST_STATE_PATH.read_text(encoding="utf-8"))
    return {
        "avg_mae_mm": float("inf"),
        "z_avg_mae_mm": float("inf"),
        "holdout_trio_mae_mm": float("inf"),
        "run_id": None,
    }


def _save_best_state(state: dict[str, Any]) -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _fmt(v: float | None, digits: int = 2) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.{digits}f}"


def _decision(
    avg_mae: float,
    z_mae: float,
    holdout_mae: float,
    best: dict[str, Any],
) -> str:
    """Acceptance: mean MAE or Z MAE improves ≥0.2 mm AND holdout trio does not worsen >1 mm."""
    best_avg = float(best.get("avg_mae_mm", float("inf")))
    best_z = float(best.get("z_avg_mae_mm", float("inf")))
    best_h = float(best.get("holdout_trio_mae_mm", float("inf")))
    if not np.isfinite(avg_mae) or not np.isfinite(z_mae):
        return "отклонено"
    improved = (
        (np.isfinite(best_avg) and (best_avg - avg_mae) >= 0.2)
        or (np.isfinite(best_z) and (best_z - z_mae) >= 0.2)
        or (not np.isfinite(best_avg) and not np.isfinite(best_z))
    )
    # First real baseline always accepted as reference.
    if best.get("run_id") is None:
        return "принято"
    holdout_ok = True
    if np.isfinite(holdout_mae) and np.isfinite(best_h):
        holdout_ok = (holdout_mae - best_h) <= 1.0
    elif not np.isfinite(holdout_mae) and np.isfinite(best_h):
        holdout_ok = False
    if improved and holdout_ok:
        return "принято"
    if abs(avg_mae - best_avg) < 0.2 and abs(z_mae - best_z) < 0.2 and holdout_ok:
        return "без эффекта"
    return "отклонено"


def append_journal_row(row: dict[str, str]) -> None:
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not JOURNAL_PATH.exists():
        header = (
            "# Журнал экспериментов: Вариант А\n\n"
            "Правило принятия: средняя MAE или MAE по Z улучшается ≥ 0.2 мм "
            "на повторной nested CV (seeds 0,1,2) и ошибка на контрольной тройке "
            "не ухудшается больше чем на 1 мм.\n\n"
            "| этап | изменение | коммит | MAE ср. | CI95 | MAE X | MAE Y | MAE Z | "
            "MAE Zл | MAE Zп | R² | 3D ср. | ≤10 мм | Δ к лучшей | контроль тройка | вывод |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        )
        JOURNAL_PATH.write_text(header, encoding="utf-8")
    line = (
        f"| {row['stage']} | {row['change']} | {row['commit']} | {row['mae_avg']} | "
        f"{row['ci95']} | {row['mae_x']} | {row['mae_y']} | {row['mae_z']} | "
        f"{row['mae_zl']} | {row['mae_zr']} | {row['r2']} | {row['err_3d']} | "
        f"{row['within10']} | {row['delta_best']} | {row['holdout']} | {row['verdict']} |\n"
    )
    with JOURNAL_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line)


def build_training_frame(
    *,
    xlsx: Path,
    holdout: Path,
    skip_build: bool,
    vybor_csv: Path,
) -> pd.DataFrame:
    if not skip_build:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "data" / "build_vybor_from_xlsx.py"),
            "--xlsx",
            str(xlsx),
            "--out",
            str(vybor_csv),
            "--no-boku",
            "--holdout",
            str(holdout),
        ]
        subprocess.run(cmd, cwd=str(ROOT), check=True)
    df = normalize_dataframe(pd.read_csv(vybor_csv))
    df = filter_any_kidney_targets(df).reset_index(drop=True)
    assert_no_holdout_leak(df, banned_keys=holdout_surname_keys())
    return df


def _raw_z_preds(trainer: AdaptiveEnsembleTrainer, frame: pd.DataFrame, target: str) -> np.ndarray:
    from src.features.pipeline import apply_model_preprocessing, build_inference_matrix

    X = build_inference_matrix(trainer, frame, feature_names=trainer.feature_names)
    X_scaled = apply_model_preprocessing(
        X, {"imputer": trainer.imputer, "scaler": trainer.scaler}
    )
    return trainer.trained_models[target].predict(X_scaled)


def fit_and_save_experiment_model(
    df: pd.DataFrame,
    *,
    run_id: str,
    trainer_kwargs: dict[str, Any],
    weight_mode: str,
) -> Path:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    enrichment = trainer_kwargs.get("enrichment_mode", "none")
    na_store = None
    if enrichment == "na_trends":
        na_store = NaTrendStore.fit()
        trainer_kwargs = dict(trainer_kwargs)
        trainer_kwargs["na_trend_store"] = na_store
    trainer = AdaptiveEnsembleTrainer(**trainer_kwargs)
    prepared = trainer.prepare_training_data_fit(df)
    if prepared[0] is None:
        raise RuntimeError("prepare_training_data_fit failed")
    X_train, y_train = prepared
    name_col = "full_name" if "full_name" in df.columns else "case_id"
    groups = df[name_col].astype(str).values
    trainer.fit_final(X_train, y_train, groups=groups, weight_mode=weight_mode)

    left_cal = fit_calibrator_oof_gated(
        SideZCalibrator(side="left"),
        df,
        _raw_z_preds(trainer, df, Z_TARGETS[0]),
        df[Z_TARGETS[0]].astype(float).values,
        groups,
    )
    right_cal = fit_calibrator_oof_gated(
        SideZCalibrator(side="right"),
        df,
        _raw_z_preds(trainer, df, Z_TARGETS[1]),
        df[Z_TARGETS[1]].astype(float).values,
        groups,
    )
    payload = build_runtime_payload(
        trainer,
        extras={
            "left_z_calibrator": left_cal,
            "right_z_calibrator": right_cal,
            "na_trend_store": na_store.to_dict() if na_store is not None else None,
            "training_meta": {
                "run_id": run_id,
                "variant_a": True,
                "holdout_excluded": True,
                "weight_mode": weight_mode,
            },
        },
    )
    out = EXPERIMENTS_DIR / f"{run_id}.pkl"
    joblib.dump(payload, out)
    return out


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    run_id = args.run_id or f"{args.stage}_{args.tag}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    trainer_kwargs: dict[str, Any] = {
        "estimator_profile": args.estimator_profile,
        "model_kind": args.model_kind,
        "enrichment_mode": args.enrichment_mode,
        "yz_target_boost": args.yz_target_boost,
        "add_missing_indicators": args.add_missing_indicators,
        "encode_categoricals": True,
    }
    if args.inner_n_splits is not None:
        trainer_kwargs["inner_n_splits"] = args.inner_n_splits
    if args.loss_profile:
        trainer_kwargs["loss_profile"] = args.loss_profile
    if args.yz_boost_mode:
        trainer_kwargs["yz_boost_mode"] = args.yz_boost_mode
    if args.drop_feature_groups:
        trainer_kwargs["drop_feature_groups"] = tuple(args.drop_feature_groups)
    if args.drop_feature_prefixes:
        trainer_kwargs["drop_feature_prefixes"] = tuple(args.drop_feature_prefixes)
    if args.keep_feature_names:
        trainer_kwargs["keep_feature_names"] = tuple(args.keep_feature_names)
    if args.ensemble_weight_mode:
        trainer_kwargs["ensemble_weight_mode"] = args.ensemble_weight_mode
    if args.estimator_overrides:
        trainer_kwargs["estimator_overrides"] = json.loads(args.estimator_overrides)
    if args.z_postprocess:
        trainer_kwargs["z_postprocess"] = args.z_postprocess

    df = build_training_frame(
        xlsx=args.xlsx,
        holdout=args.holdout,
        skip_build=args.skip_build,
        vybor_csv=args.vybor_csv,
    )
    print(f"[data] n={len(df)} run_id={run_id}", flush=True)

    na_store = None
    factory_kwargs = {k: v for k, v in trainer_kwargs.items() if k != "model_kind"}
    if args.enrichment_mode == "na_trends":
        na_store = NaTrendStore.fit()
        factory_kwargs["enrichment_mode"] = "na_trends"
        # Per-fold store is injected by nested_cv via trainer_factory(na_trend_store=...).
    factory = make_trainer_factory(args.model_kind, **factory_kwargs)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    t0 = time.time()
    repeated = evaluate_repeated_nested_cv(
        df,
        trainer_factory=factory,
        seeds=seeds,
        n_splits=args.n_splits,
        na_trend_store=na_store,
        weight_mode=args.weight_mode,
    )
    cv_seconds = time.time() - t0
    agg = repeated["aggregated"]
    # Prefer seed-0 full result for left/right Z and CI.
    seed0 = repeated["results"][0]
    per_t = seed0.metrics.get("per_target_mae_mm") or {}
    axis = agg.get("axis_mae_mm") or {}
    ci = seed0.metrics.get("avg_mae_ci95") or [float("nan"), float("nan")]
    clinical = seed0.metrics.get("clinical_3d") or {}

    model_path = None
    holdout_mae = float("nan")
    if not args.skip_fit:
        model_path = fit_and_save_experiment_model(
            df,
            run_id=run_id,
            trainer_kwargs=dict(trainer_kwargs),
            weight_mode=args.weight_mode,
        )
        from scripts.validation.eval_holdout import evaluate_holdout

        holdout_report = evaluate_holdout(
            model_path, xlsx=args.xlsx, holdout_path=args.holdout
        )
        holdout_mae = float(holdout_report.get("holdout_trio_mean_mae_mm", float("nan")))
        holdout_path = EXPERIMENTS_DIR / f"{run_id}_holdout.json"
        holdout_path.write_text(
            json.dumps(holdout_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    avg_mae = float((agg.get("avg_mae_mm") or {}).get("mean", float("nan")))
    z_mae = float((agg.get("z_avg_mae_mm") or {}).get("mean", float("nan")))
    best = _load_best_state()
    verdict = _decision(avg_mae, z_mae, holdout_mae, best)
    delta_avg = avg_mae - float(best["avg_mae_mm"]) if np.isfinite(best["avg_mae_mm"]) else float("nan")

    if verdict == "принято":
        _save_best_state(
            {
                "avg_mae_mm": avg_mae,
                "z_avg_mae_mm": z_mae,
                "holdout_trio_mae_mm": holdout_mae,
                "run_id": run_id,
                "model_path": str(model_path) if model_path else None,
            }
        )

    change = args.change or args.tag
    journal = {
        "stage": args.stage,
        "change": change.replace("|", "/"),
        "commit": _git_commit(),
        "mae_avg": _fmt(avg_mae),
        "ci95": f"[{_fmt(ci[0])}; {_fmt(ci[1])}]" if len(ci) == 2 else "—",
        "mae_x": _fmt((axis.get("x") or {}).get("mean")),
        "mae_y": _fmt((axis.get("y") or {}).get("mean")),
        "mae_z": _fmt((axis.get("z") or {}).get("mean")),
        "mae_zl": _fmt(per_t.get("kidney_left_delta_z")),
        "mae_zr": _fmt(per_t.get("kidney_right_delta_z")),
        "r2": _fmt((agg.get("avg_r2") or {}).get("mean"), 3),
        "err_3d": _fmt(clinical.get("endpoint_error_mean_mae_mm")),
        "within10": _fmt(
            100.0 * float(clinical.get("within_10mm_ratio", float("nan"))), 1
        )
        + "%"
        if np.isfinite(float(clinical.get("within_10mm_ratio", float("nan"))))
        else "—",
        "delta_best": _fmt(delta_avg),
        "holdout": _fmt(holdout_mae),
        "verdict": verdict,
    }
    if not args.no_journal:
        append_journal_row(journal)

    payload = {
        "run_id": run_id,
        "stage": args.stage,
        "change": change,
        "commit": journal["commit"],
        "trainer_kwargs": {k: v for k, v in trainer_kwargs.items() if k != "na_trend_store"},
        "weight_mode": args.weight_mode,
        "n_patients": len(df),
        "cv_seconds": cv_seconds,
        "repeated_cv": {
            "seeds": list(seeds),
            "aggregated": agg,
            "per_seed": repeated["per_seed"],
        },
        "holdout_trio_mean_mae_mm": holdout_mae,
        "model_path": str(model_path) if model_path else None,
        "verdict": verdict,
        "journal": journal,
        "vs_best": paired_compare_repeated(
            repeated,
            {
                "aggregated": {
                    "avg_mae_mm": {"mean": best.get("avg_mae_mm")},
                    "z_avg_mae_mm": {"mean": best.get("z_avg_mae_mm")},
                    "within_10mm_ratio": {"mean": float("nan")},
                }
            },
        ),
    }
    out_json = EXPERIMENTS_DIR / f"{run_id}_metrics.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "verdict": verdict, "avg_mae": avg_mae, "z_mae": z_mae, "holdout": holdout_mae}, ensure_ascii=False), flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Variant A experiment runner")
    p.add_argument("--stage", required=True, help="Stage id, e.g. 0, 1, 2a")
    p.add_argument("--tag", required=True, help="Short slug for run_id")
    p.add_argument("--change", default="", help="Russian description for journal")
    p.add_argument("--run-id", default=None)
    p.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX_PATH)
    p.add_argument("--vybor-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT_PATH)
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--skip-fit", action="store_true", help="CV only, no holdout model")
    p.add_argument("--no-journal", action="store_true")
    p.add_argument("--model-kind", default="ensemble")
    p.add_argument("--estimator-profile", default="production", choices=("production", "tiny", "small_n"))
    p.add_argument("--enrichment-mode", default="none", choices=("none", "na_trends", "projection"))
    p.add_argument("--yz-target-boost", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--yz-boost-mode", default=None, choices=("default", "off", "soft", "y_only"))
    p.add_argument("--loss-profile", default=None)
    p.add_argument("--add-missing-indicators", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--drop-feature-groups", nargs="*", default=None)
    p.add_argument("--drop-feature-prefixes", nargs="*", default=None)
    p.add_argument("--keep-feature-names", nargs="*", default=None)
    p.add_argument("--ensemble-weight-mode", default=None)
    p.add_argument("--estimator-overrides", default=None, help="JSON dict of RF/GBT overrides")
    p.add_argument("--z-postprocess", default=None)
    p.add_argument("--weight-mode", default="inner_groupkfold", choices=("inner_groupkfold", "fixed_prior"))
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--inner-n-splits", type=int, default=None)
    p.add_argument("--seeds", default="0,1,2")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
