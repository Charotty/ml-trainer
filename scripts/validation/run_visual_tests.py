#!/usr/bin/env python3
"""Generate kidney displacement visual tests for selected cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import (
    DEFAULT_MODEL_PATH_STR,
    TARGET_COLUMNS,
    build_or_load_predictor,
    ensure_run_dirs,
    load_dataset,
    predict_df,
    save_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/vybor_unified_features.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH_STR)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", default="results/validation_runs")
    parser.add_argument("--num-cases", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="Plot all rows in dataset (no re-split)",
    )
    return parser.parse_args()


def kidney_cloud(center_xyz: np.ndarray, scale_xyz: tuple[float, float, float], n: int = 5000) -> np.ndarray:
    pts = np.random.normal(size=(n, 3))
    pts /= np.linalg.norm(pts, axis=1)[:, None]
    radii = np.random.uniform(0.2, 1.0, size=(n, 1))
    cloud = pts * radii
    cloud[:, 0] *= scale_xyz[0]
    cloud[:, 1] *= scale_xyz[1]
    cloud[:, 2] *= scale_xyz[2]
    cloud += center_xyz
    return cloud


def vertebra_cloud(spine_xyz: np.ndarray, n: int = 4000) -> np.ndarray:
    half = np.array([12.0, 15.0, 14.0])
    points = np.random.uniform(-1.0, 1.0, size=(n, 3)) * half
    return points + spine_xyz


def norm3(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def quality_checks(delta_left: np.ndarray, delta_right: np.ndarray) -> dict:
    checks = {
        "left_delta_norm_lt_80": bool(norm3(delta_left) < 80),
        "right_delta_norm_lt_80": bool(norm3(delta_right) < 80),
        "left_right_x_have_opposite_sign": bool(np.sign(delta_left[0]) != np.sign(delta_right[0])),
        "yz_magnitude_lt_xx2_left": bool(abs(delta_left[1]) + abs(delta_left[2]) < abs(delta_left[0]) * 2.0),
        "yz_magnitude_lt_xx2_right": bool(abs(delta_right[1]) + abs(delta_right[2]) < abs(delta_right[0]) * 2.0),
    }
    checks["all_passed"] = bool(all(checks.values()))
    return checks


def save_case_json(out_path: Path, payload: dict) -> None:
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_single_case_3d(out_path: Path, left_sup: np.ndarray, right_sup: np.ndarray, spine: np.ndarray, dl: np.ndarray, dr: np.ndarray) -> None:
    left_lat = left_sup + dl
    right_lat = right_sup + dr

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    l1 = kidney_cloud(left_sup, (24, 14, 38))
    r1 = kidney_cloud(right_sup, (24, 14, 38))
    v = vertebra_cloud(spine)
    l2 = kidney_cloud(left_lat, (24, 14, 38))
    r2 = kidney_cloud(right_lat, (24, 14, 38))

    ax.scatter(l1[:, 0], l1[:, 1], l1[:, 2], s=1, alpha=0.2, c="crimson", label="Left kidney (supine)")
    ax.scatter(r1[:, 0], r1[:, 1], r1[:, 2], s=1, alpha=0.2, c="dodgerblue", label="Right kidney (supine)")
    ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=1, alpha=0.2, c="black", label="L3 vertebra body")
    ax.scatter(l2[:, 0], l2[:, 1], l2[:, 2], s=1, alpha=0.15, c="darkred", label="Left kidney (predicted lateral)")
    ax.scatter(r2[:, 0], r2[:, 1], r2[:, 2], s=1, alpha=0.15, c="navy", label="Right kidney (predicted lateral)")
    ax.set_xlabel("X (mm)  L->R")
    ax.set_ylabel("Y (mm)  P->A")
    ax.set_zlabel("Z (mm)  I->S")
    ax.set_title("Predicted kidney displacement (supine -> lateral), mm")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_multi_panel(out_path: Path, left_sup: np.ndarray, right_sup: np.ndarray, spine: np.ndarray, dl: np.ndarray, dr: np.ndarray) -> None:
    left_lat = left_sup + dl
    right_lat = right_sup + dr
    l1 = kidney_cloud(left_sup, (24, 14, 38))
    r1 = kidney_cloud(right_sup, (24, 14, 38))
    v = vertebra_cloud(spine)
    l2 = kidney_cloud(left_lat, (24, 14, 38))
    r2 = kidney_cloud(right_lat, (24, 14, 38))

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133, projection="3d")

    ax1.scatter(l1[:, 0], l1[:, 1], s=1, c="crimson", alpha=0.18)
    ax1.scatter(r1[:, 0], r1[:, 1], s=1, c="dodgerblue", alpha=0.18)
    ax1.scatter(v[:, 0], v[:, 1], s=1, c="black", alpha=0.18)
    ax1.scatter(l2[:, 0], l2[:, 1], s=1, c="darkred", alpha=0.12)
    ax1.scatter(r2[:, 0], r2[:, 1], s=1, c="navy", alpha=0.12)
    ax1.set_xlabel("X (mm)  L->R")
    ax1.set_ylabel("Y (mm)  P->A")
    ax1.set_title("Axial view (XY)")
    ax1.grid(alpha=0.3)

    ax2.scatter(l1[:, 0], l1[:, 2], s=1, c="crimson", alpha=0.18)
    ax2.scatter(r1[:, 0], r1[:, 2], s=1, c="dodgerblue", alpha=0.18)
    ax2.scatter(v[:, 0], v[:, 2], s=1, c="black", alpha=0.18)
    ax2.scatter(l2[:, 0], l2[:, 2], s=1, c="darkred", alpha=0.12)
    ax2.scatter(r2[:, 0], r2[:, 2], s=1, c="navy", alpha=0.12)
    ax2.set_xlabel("X (mm)  L->R")
    ax2.set_ylabel("Z (mm)  I->S")
    ax2.set_title("Coronal view (XZ)")
    ax2.grid(alpha=0.3)

    ax3.scatter(l1[:, 0], l1[:, 1], l1[:, 2], s=1, c="crimson", alpha=0.15)
    ax3.scatter(r1[:, 0], r1[:, 1], r1[:, 2], s=1, c="dodgerblue", alpha=0.15)
    ax3.scatter(v[:, 0], v[:, 1], v[:, 2], s=1, c="black", alpha=0.15)
    ax3.scatter(l2[:, 0], l2[:, 1], l2[:, 2], s=1, c="darkred", alpha=0.12)
    ax3.scatter(r2[:, 0], r2[:, 1], r2[:, 2], s=1, c="navy", alpha=0.12)
    ax3.set_title("3D view")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_overlay(out_path: Path, left_sup: np.ndarray, right_sup: np.ndarray, dl: np.ndarray, dr: np.ndarray) -> None:
    left_lat = left_sup + dl
    right_lat = right_sup + dr
    l1 = kidney_cloud(left_sup, (24, 14, 38))
    r1 = kidney_cloud(right_sup, (24, 14, 38))
    l2 = kidney_cloud(left_lat, (24, 14, 38))
    r2 = kidney_cloud(right_lat, (24, 14, 38))

    fig = plt.figure(figsize=(11, 4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.scatter(l1[:, 0], l1[:, 1], s=1, c="crimson", alpha=0.16)
    ax1.scatter(l2[:, 0], l2[:, 1], s=1, c="darkred", alpha=0.1)
    ax1.scatter(r1[:, 0], r1[:, 1], s=1, c="dodgerblue", alpha=0.16)
    ax1.scatter(r2[:, 0], r2[:, 1], s=1, c="navy", alpha=0.1)
    ax1.set_title("Axial overlay")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.grid(alpha=0.3)

    ax2.scatter(l1[:, 0], l1[:, 2], s=1, c="crimson", alpha=0.16)
    ax2.scatter(l2[:, 0], l2[:, 2], s=1, c="darkred", alpha=0.1)
    ax2.scatter(r1[:, 0], r1[:, 2], s=1, c="dodgerblue", alpha=0.16)
    ax2.scatter(r2[:, 0], r2[:, 2], s=1, c="navy", alpha=0.1)
    ax2.set_title("Coronal overlay")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")
    ax2.grid(alpha=0.3)

    ax3.scatter(l1[:, 1], l1[:, 2], s=1, c="crimson", alpha=0.16, label="Left supine")
    ax3.scatter(l2[:, 1], l2[:, 2], s=1, c="darkred", alpha=0.1, label="Left predicted")
    ax3.scatter(r1[:, 1], r1[:, 2], s=1, c="dodgerblue", alpha=0.16, label="Right supine")
    ax3.scatter(r2[:, 1], r2[:, 2], s=1, c="navy", alpha=0.1, label="Right predicted")
    ax3.set_title("Sagittal overlay")
    ax3.set_xlabel("Y")
    ax3.set_ylabel("Z")
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)

    run_dir = ensure_run_dirs(Path(args.out_dir), args.run_id)
    dataset_path = Path(args.dataset)
    model_path = Path(args.model) if args.model else None
    df = load_dataset(dataset_path)
    bundle, train_df, eval_df = build_or_load_predictor(
        df=df,
        model_path=model_path,
        test_size=args.test_size,
        seed=args.seed,
        holdout_eval=args.holdout,
    )
    preds = predict_df(bundle, eval_df)

    selected_idx = eval_df.index.tolist()[: args.num_cases]
    for idx in selected_idx:
        row = eval_df.loc[idx]
        pred_row = preds.loc[idx]
        delta_left = pred_row[
            ["kidney_left_delta_x", "kidney_left_delta_y", "kidney_left_delta_z"]
        ].to_numpy(dtype=float)
        delta_right = pred_row[
            ["kidney_right_delta_x", "kidney_right_delta_y", "kidney_right_delta_z"]
        ].to_numpy(dtype=float)

        left_supine = row[
            ["kidney_left_center_x_rel", "kidney_left_center_y_rel", "kidney_left_center_z_rel"]
        ].to_numpy(dtype=float)
        right_supine = row[
            ["kidney_right_center_x_rel", "kidney_right_center_y_rel", "kidney_right_center_z_rel"]
        ].to_numpy(dtype=float)
        spine = row[["spine_center_x", "spine_center_y", "spine_center_z"]].to_numpy(dtype=float)

        case_id = str(row.get("case_id", f"row_{idx}"))
        base_name = f"case_{case_id}".replace("/", "_")
        case_plot_dir = run_dir / "plots"

        plot_single_case_3d(case_plot_dir / f"{base_name}_single_case_3d.png", left_supine, right_supine, spine, delta_left, delta_right)
        plot_multi_panel(case_plot_dir / f"{base_name}_multi_panel_2d3d.png", left_supine, right_supine, spine, delta_left, delta_right)
        plot_overlay(case_plot_dir / f"{base_name}_overlay_supine_vs_predicted.png", left_supine, right_supine, delta_left, delta_right)

        save_case_json(
            run_dir / "predictions" / f"{base_name}.json",
            {
                "case_id": case_id,
                "predictor_mode": bundle.mode,
                "predicted": {k: float(pred_row[k]) for k in TARGET_COLUMNS},
                "quality_checks": quality_checks(delta_left, delta_right),
                "norms_mm": {
                    "left": norm3(delta_left),
                    "right": norm3(delta_right),
                },
            },
        )

    save_manifest(
        run_dir,
        run_id=args.run_id,
        dataset_path=dataset_path,
        model_path=model_path,
        predictor_mode=bundle.mode,
        train_count=len(train_df),
        eval_count=len(eval_df),
        seed=args.seed,
    )
    print(f"[OK] Visual tests complete. Artifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
