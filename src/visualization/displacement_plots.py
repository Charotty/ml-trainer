"""Displacement charts for single-case CT Workbench reports.

Adapted from scripts/validation/run_visual_tests.py and plot_harmonized_analysis.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

TARGET_KEYS = [
    "kidney_left_delta_x",
    "kidney_left_delta_y",
    "kidney_left_delta_z",
    "kidney_right_delta_x",
    "kidney_right_delta_y",
    "kidney_right_delta_z",
]

SHORT_LABELS = [
    "Лев.\nвправо–влево",
    "Лев.\nвперёд–назад",
    "Лев.\nвверх–вниз",
    "Прав.\nвправо–влево",
    "Прав.\nвперёд–назад",
    "Прав.\nвверх–вниз",
]

RU_LABELS = {
    "kidney_left_delta_x": "Левая почка, вправо–влево (мм)",
    "kidney_left_delta_y": "Левая почка, вперёд–назад (мм)",
    "kidney_left_delta_z": "Левая почка, вверх–вниз (мм)",
    "kidney_right_delta_x": "Правая почка, вправо–влево (мм)",
    "kidney_right_delta_y": "Правая почка, вперёд–назад (мм)",
    "kidney_right_delta_z": "Правая почка, вверх–вниз (мм)",
}

AXIS_HINTS = {
    "x": "вправо–влево (к правой стороне тела)",
    "y": "вперёд–назад (к животу)",
    "z": "вверх–вниз (к голове)",
}


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(out):
        return None
    return out


def predictions_vector(predictions: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    left = np.array(
        [_as_float(predictions.get(f"kidney_left_delta_{a}")) or 0.0 for a in ("x", "y", "z")],
        dtype=float,
    )
    right = np.array(
        [_as_float(predictions.get(f"kidney_right_delta_{a}")) or 0.0 for a in ("x", "y", "z")],
        dtype=float,
    )
    return left, right


def vector_norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def quality_checks(delta_left: np.ndarray, delta_right: np.ndarray) -> Dict[str, bool]:
    checks = {
        "left_delta_norm_lt_80": bool(vector_norm(delta_left) < 80),
        "right_delta_norm_lt_80": bool(vector_norm(delta_right) < 80),
        "left_right_x_have_opposite_sign": bool(np.sign(delta_left[0]) != np.sign(delta_right[0])),
        "yz_magnitude_lt_xx2_left": bool(abs(delta_left[1]) + abs(delta_left[2]) < abs(delta_left[0]) * 2.0),
        "yz_magnitude_lt_xx2_right": bool(abs(delta_right[1]) + abs(delta_right[2]) < abs(delta_right[0]) * 2.0),
    }
    checks["all_passed"] = bool(all(checks.values()))
    return checks


def quality_check_messages(checks: Mapping[str, bool]) -> list[str]:
    """Human-readable clinical notes for automatic sanity checks."""
    notes: list[str] = []
    if not checks.get("left_delta_norm_lt_80", True):
        notes.append("Суммарное смещение левой почки превышает 80 мм — проверьте исходные данные.")
    if not checks.get("right_delta_norm_lt_80", True):
        notes.append("Суммарное смещение правой почки превышает 80 мм — проверьте исходные данные.")
    if not checks.get("left_right_x_have_opposite_sign", True):
        notes.append(
            "Ожидается смещение почек в разные стороны по горизонтали; "
            "сейчас знаки совпадают — интерпретируйте осторожно."
        )
    if not checks.get("yz_magnitude_lt_xx2_left", True) or not checks.get(
        "yz_magnitude_lt_xx2_right", True
    ):
        notes.append(
            "Смещение вверх/вниз или вперёд/назад сопоставимо с боковым — "
            "убедитесь, что анатомия и измерения согласованы."
        )
    if not notes and checks.get("all_passed"):
        notes.append("Автоматические проверки величины и направления смещения пройдены.")
    return notes


def kidney_cloud(center_xyz: np.ndarray, scale_xyz: tuple[float, float, float], n: int = 3500) -> np.ndarray:
    pts = np.random.normal(size=(n, 3))
    pts /= np.linalg.norm(pts, axis=1)[:, None]
    radii = np.random.uniform(0.2, 1.0, size=(n, 1))
    cloud = pts * radii
    cloud[:, 0] *= scale_xyz[0]
    cloud[:, 1] *= scale_xyz[1]
    cloud[:, 2] *= scale_xyz[2]
    return cloud + center_xyz


def vertebra_cloud(spine_xyz: np.ndarray, n: int = 2800) -> np.ndarray:
    half = np.array([12.0, 15.0, 14.0])
    return np.random.uniform(-1.0, 1.0, size=(n, 3)) * half + spine_xyz


def estimate_kidney_centers(base_features: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return left_sup, right_sup, spine; estimated=True if kidney centers were approximated."""
    spine = np.array(
        [
            _as_float(base_features.get("spine_center_x")) or 0.0,
            _as_float(base_features.get("spine_center_y")) or 0.0,
            _as_float(base_features.get("spine_center_z")) or 0.0,
        ],
        dtype=float,
    )
    estimated = False
    centers: list[np.ndarray] = []
    body_width = _as_float(base_features.get("body_width_mm")) or 240.0
    for side, sign in (("left", -1.0), ("right", 1.0)):
        coords = [
            _as_float(base_features.get(f"kidney_{side}_center_x_rel")),
            _as_float(base_features.get(f"kidney_{side}_center_y_rel")),
            _as_float(base_features.get(f"kidney_{side}_center_z_rel")),
        ]
        if all(v is not None for v in coords):
            centers.append(np.array(coords, dtype=float))
            continue
        estimated = True
        centers.append(
            spine
            + np.array([sign * body_width * 0.18, 0.0, -25.0], dtype=float)
        )
    return centers[0], centers[1], spine, estimated


def plot_delta_bars(predictions: Mapping[str, Any]) -> Figure:
    values = [_as_float(predictions.get(k)) or 0.0 for k in TARGET_KEYS]
    colors = ["#4C78A8" if "_x" in k else "#F58518" if "_y" in k else "#E45756" for k in TARGET_KEYS]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars = ax.bar(SHORT_LABELS, values, color=colors)
    ax.axhline(0, color="gray", lw=0.8)
    ax.axhline(5, color="#22c55e", ls="--", lw=1, alpha=0.6, label="±5 мм (ориентир)")
    ax.axhline(-5, color="#22c55e", ls="--", lw=1, alpha=0.6)
    ax.set_ylabel("Смещение, мм")
    ax.set_title("Прогноз смещения почек: со спины → на бок")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    for bar, val in zip(bars, values):
        y = val + (0.8 if val >= 0 else -1.2)
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{val:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_vector_norms(predictions: Mapping[str, Any]) -> Figure:
    left, right = predictions_vector(predictions)
    labels = ["Левая почка\n(суммарно)", "Правая почка\n(суммарно)", "Среднее"]
    values = [vector_norm(left), vector_norm(right), (vector_norm(left) + vector_norm(right)) / 2.0]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, values, color=["#dc2626", "#2563eb", "#7c3aed"])
    ax.axhline(5, color="#22c55e", ls="--", label="5 мм")
    ax.axhline(10, color="#f59e0b", ls=":", label="10 мм")
    ax.set_ylabel("Суммарное смещение, мм")
    ax.set_title("Насколько далеко смещается почка в целом")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_displacement_multipanel(
    left_sup: np.ndarray,
    right_sup: np.ndarray,
    spine: np.ndarray,
    delta_left: np.ndarray,
    delta_right: np.ndarray,
    *,
    estimated_centers: bool = False,
) -> Figure:
    left_lat = left_sup + delta_left
    right_lat = right_sup + delta_right
    l1 = kidney_cloud(left_sup, (24, 14, 38))
    r1 = kidney_cloud(right_sup, (24, 14, 38))
    v = vertebra_cloud(spine)
    l2 = kidney_cloud(left_lat, (24, 14, 38))
    r2 = kidney_cloud(right_lat, (24, 14, 38))

    fig = plt.figure(figsize=(14, 4.8))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133, projection="3d")

    for ax, dims, title in (
        (ax1, (0, 1), "Аксиальная проекция (XY)"),
        (ax2, (0, 2), "Корональная проекция (XZ)"),
    ):
        ax.scatter(l1[:, dims[0]], l1[:, dims[1]], s=1, c="crimson", alpha=0.18, label="Левая, на спине")
        ax.scatter(r1[:, dims[0]], r1[:, dims[1]], s=1, c="dodgerblue", alpha=0.18, label="Правая, на спине")
        ax.scatter(l2[:, dims[0]], l2[:, dims[1]], s=1, c="darkred", alpha=0.12, label="Левая, на боку (прогноз)")
        ax.scatter(r2[:, dims[0]], r2[:, dims[1]], s=1, c="navy", alpha=0.12, label="Правая, на боку (прогноз)")
        ax.scatter(v[:, dims[0]], v[:, dims[1]], s=1, c="black", alpha=0.15, label="Позвоночник")
        ax.set_title(title)
        ax.grid(alpha=0.3)

    ax3.scatter(l1[:, 0], l1[:, 1], l1[:, 2], s=1, c="crimson", alpha=0.15)
    ax3.scatter(r1[:, 0], r1[:, 1], r1[:, 2], s=1, c="dodgerblue", alpha=0.15)
    ax3.scatter(l2[:, 0], l2[:, 1], l2[:, 2], s=1, c="darkred", alpha=0.12)
    ax3.scatter(r2[:, 0], r2[:, 1], r2[:, 2], s=1, c="navy", alpha=0.12)
    ax3.scatter(v[:, 0], v[:, 1], v[:, 2], s=1, c="black", alpha=0.12)
    ax3.set_title("Объёмная схема: спина → бок")
    ax3.set_xlabel("Вправо–влево")
    ax3.set_ylabel("Вперёд–назад")
    ax3.set_zlabel("Вверх–вниз")

    subtitle = " (положение почек оценено по размерам тела)" if estimated_centers else ""
    fig.suptitle(f"Схема смещения почек{subtitle}", y=1.02, fontsize=12)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    return fig


def _wrap_lines(text: str, width: int = 92) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = (" ".join(current + [word])).strip()
        if len(candidate) <= width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def plot_text_page(title: str, paragraphs: Sequence[str]) -> Figure:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    y = 0.94
    fig.text(0.08, y, title, fontsize=16, weight="bold", va="top")
    y -= 0.05
    for paragraph in paragraphs:
        for line in _wrap_lines(paragraph):
            fig.text(0.08, y, line, fontsize=10, va="top", family="sans-serif")
            y -= 0.028
        y -= 0.02
        if y < 0.05:
            break
    return fig


def build_case_report_pdf(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write multi-page clinical PDF report (ReportLab implementation).

    Public API kept here so Cases router and tests keep importing from
    ``displacement_plots``. Implementation lives in ``clinical_report_pdf``.
    """
    from src.visualization.clinical_report_pdf import (
        build_case_report_pdf as _build_clinical_pdf,
    )

    return _build_clinical_pdf(report, output_path)
