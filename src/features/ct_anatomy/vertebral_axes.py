"""Vertebral axes from Lyashchenko, Demin and Urazov (2020).

On the slice through a kidney third the reader draws two lines and does not
use the scanner axes:

* OY runs through the spinous process and the middle of the vertebral body;
* OX is perpendicular to OY and passes through the posterior point of the
  spinal canal.

X is the unsigned distance from the kidney to OY. Y is the signed distance to
OX, positive toward the front of the vertebra. Z along the spine is reported
both from that vertebra and from L1, L2 and L3, because the sheet's vertical
origin is not the vertebra under the kidney.

Four coordinate variants are written for every third: the kidney point is
either the centre of mass of the third or the point nearest each axis, and
the Y origin is either the vertebral-body centre or the posterior canal point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from src.features.ct_anatomy.perirenal import _median_ray
from src.features.ct_anatomy.qc import QC_APPROX, QC_MISSING
from src.features.ct_anatomy.spine import fit_endplate_normal
from src.features.ct_anatomy.volume import PATIENT_Y, PATIENT_Z, AnatomyVolume, mask_points_mm

SPINE_LEVELS: Tuple[str, ...] = ("T11", "T12", "L1", "L2", "L3", "L4", "L5", "S1")
THIRDS: Tuple[str, ...] = ("lower", "middle", "upper")
ORIGINS: Tuple[str, ...] = ("body", "canal")
POINT_KINDS: Tuple[str, ...] = ("com", "near")
Z_REFERENCES: Tuple[str, ...] = ("L1", "L2", "L3")
_MIDLINE_MM = 8.0
_ARCH_GAP_MM = 3.0


@dataclass
class VertebraAxes:
    """One vertebra, with OY pointing from the body toward the spinous process."""

    level: str
    body_center: np.ndarray
    canal_posterior: np.ndarray
    superior: np.ndarray
    posterior: np.ndarray
    left: np.ndarray
    qc: str

    @property
    def anterior(self) -> np.ndarray:
        return -self.posterior


def variant_column(side: str, third: str, axis: str, origin: str, kind: str) -> str:
    return f"kidney_{side}_{third}_{axis}_{origin}_{kind}"


def z_reference_column(side: str, third: str, level: str) -> str:
    return f"kidney_{side}_{third}_z_from_{level.lower()}"


def _superior_axis(body_pts: np.ndarray) -> np.ndarray:
    normal = fit_endplate_normal(body_pts)
    if normal is not None:
        return normal
    if len(body_pts) < 10:
        return PATIENT_Z.copy()
    centered = body_pts - body_pts.mean(axis=0)
    _evals, evecs = np.linalg.eigh(np.cov(centered.T))
    axis = evecs[:, 0].astype(float)
    if axis[2] < 0:
        axis = -axis
    return axis


def _in_plane(vector: np.ndarray, superior: np.ndarray) -> Optional[np.ndarray]:
    flat = vector - superior * float(vector @ superior)
    norm = float(np.linalg.norm(flat))
    if norm < 2.0:
        return None
    return flat / norm


def _body_and_arch(
    volume: AnatomyVolume, level: str, vertebra: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Body of this level, and the rest of the vertebra (arch and processes)."""
    named = volume.mask(f"vertebrae_{level}")
    body_mask = volume.mask("vertebrae_body")
    qc = QC_APPROX
    body = np.zeros((0, 3), dtype=float)
    if named is not None and body_mask is not None and named.shape == body_mask.shape:
        both = named & body_mask
        if np.any(both):
            body = mask_points_mm(both, volume.affine)
            qc = "ok"
    if len(body) < 10 and body_mask is not None and len(vertebra):
        z0, z1 = np.quantile(vertebra[:, 2], [0.1, 0.9])
        pooled = volume.points("vertebrae_body")
        band = pooled[(pooled[:, 2] >= z0 - 2.0) & (pooled[:, 2] <= z1 + 2.0)]
        if len(band) >= 10:
            body = band
            qc = QC_APPROX
    if len(body) < 10 or len(vertebra) < 10:
        return body, np.zeros((0, 3), dtype=float), QC_MISSING
    dist, _idx = cKDTree(body).query(vertebra, k=1)
    arch = vertebra[dist > _ARCH_GAP_MM]
    if len(arch) < 5:
        return body, arch, QC_APPROX
    return body, arch, qc


def _posterior_from_arch(
    body_center: np.ndarray, arch: np.ndarray, superior: np.ndarray
) -> Optional[np.ndarray]:
    """OY: body centre toward the spinous process, not toward a transverse process."""
    posterior = _in_plane(arch.mean(axis=0) - body_center, superior)
    if posterior is None:
        return None
    left = np.cross(posterior, superior)
    left_norm = float(np.linalg.norm(left))
    if left_norm < 1e-6:
        return posterior
    left = left / left_norm
    offset = arch - body_center
    along = offset @ posterior
    side = np.abs(offset @ left)
    tip = arch[(along >= float(np.quantile(along, 0.8))) & (side <= 10.0)]
    if len(tip) < 3:
        return posterior
    refined = _in_plane(tip.mean(axis=0) - body_center, superior)
    return posterior if refined is None else refined


def _canal_posterior(
    body: np.ndarray,
    body_center: np.ndarray,
    arch: np.ndarray,
    superior: np.ndarray,
    posterior: np.ndarray,
) -> np.ndarray:
    """Posterior wall of the canal: anterior face of the lamina on the midline."""
    left = np.cross(posterior, superior)
    left = left / np.linalg.norm(left)
    offset = arch - body_center
    along = offset @ posterior
    side = np.abs(offset @ left)
    body_back = float(np.quantile((body - body_center) @ posterior, 0.9))
    midline = arch[(side <= _MIDLINE_MM) & (along > body_back + 1.0)]
    if len(midline) == 0:
        return body_center + posterior * body_back
    mid_along = (midline - body_center) @ posterior
    order = np.argsort(mid_along)
    count = max(1, int(round(0.15 * len(order))))
    face = midline[order[:count]]
    return face.mean(axis=0)


def _fallback_posterior(superior: np.ndarray) -> np.ndarray:
    flat = _in_plane(PATIENT_Y, superior)
    if flat is not None:
        return flat
    flat = _in_plane(np.array([1.0, 0.0, 0.0]), superior)
    return PATIENT_Y.copy() if flat is None else flat


def build_vertebra_axes(volume: AnatomyVolume) -> Dict[str, VertebraAxes]:
    """Local axes for every named vertebra that has a body and an arch."""
    frames: Dict[str, VertebraAxes] = {}
    for level in SPINE_LEVELS:
        vertebra = volume.points(f"vertebrae_{level}")
        if len(vertebra) < 10:
            continue
        body, arch, qc = _body_and_arch(volume, level, vertebra)
        if len(body) < 10:
            continue
        center = body.mean(axis=0)
        superior = _superior_axis(body)
        if len(arch) >= 5:
            posterior = _posterior_from_arch(center, arch, superior)
        else:
            posterior = None
            qc = QC_APPROX
        if posterior is None:
            posterior = _fallback_posterior(superior)
            qc = QC_APPROX
        left = np.cross(posterior, superior)
        left_norm = float(np.linalg.norm(left))
        if left_norm < 1e-6:
            continue
        left = left / left_norm
        if len(arch) >= 5:
            canal = _canal_posterior(body, center, arch, superior, posterior)
        else:
            canal = center
        frames[level] = VertebraAxes(
            level=level,
            body_center=center,
            canal_posterior=canal,
            superior=superior,
            posterior=posterior,
            left=left,
            qc=qc,
        )
    return frames


def _split_thirds(points: np.ndarray, superior: np.ndarray) -> Dict[str, np.ndarray]:
    """Lower, middle and upper thirds along the spine, upper toward the head."""
    if len(points) < 10:
        return {}
    height = points @ superior
    edges = np.linspace(float(height.min()), float(height.max()), 4)
    edges[-1] = edges[-1] + 1e-3
    found: Dict[str, np.ndarray] = {}
    for name, lo, hi in (
        ("lower", edges[0], edges[1]),
        ("middle", edges[1], edges[2]),
        ("upper", edges[2], edges[3]),
    ):
        band = points[(height >= lo) & (height < hi)]
        if len(band):
            found[name] = band
    return found


def _frame_at(frames: Sequence[VertebraAxes], point: np.ndarray) -> Optional[VertebraAxes]:
    if not frames:
        return None
    superior = frames[0].superior
    height = float(point @ superior)
    return min(frames, key=lambda frame: abs(float(frame.body_center @ superior) - height))


def _level_fraction(frames: Sequence[VertebraAxes], frame: VertebraAxes, point: np.ndarray) -> float:
    ordered = sorted(frames, key=lambda item: float(item.body_center @ item.superior))
    index = next(i for i, item in enumerate(ordered) if item.level == frame.level)
    here = float(frame.body_center @ frame.superior)
    below = ordered[index - 1] if index > 0 else None
    above = ordered[index + 1] if index + 1 < len(ordered) else None
    low = here - 12.0 if below is None else 0.5 * (here + float(below.body_center @ frame.superior))
    high = here + 12.0 if above is None else 0.5 * (here + float(above.body_center @ frame.superior))
    span = high - low
    if span < 1.0:
        return 0.5
    return float(np.clip((float(point @ frame.superior) - low) / span, 0.0, 1.0))


def _origin_of(frame: VertebraAxes, origin: str) -> np.ndarray:
    return frame.body_center if origin == "body" else frame.canal_posterior


def _centroid_xyz(point: np.ndarray, frame: VertebraAxes, origin: str) -> Tuple[float, float, float]:
    delta = point - _origin_of(frame, origin)
    return (
        abs(float(delta @ frame.left)),
        float(delta @ frame.anterior),
        float(delta @ frame.superior),
    )


def _nearest_xy(band: np.ndarray, frame: VertebraAxes, origin: str) -> Tuple[float, float]:
    """Shortest distance to OY, and the Y of the point closest to OX."""
    delta = band - _origin_of(frame, origin)
    lateral = np.abs(delta @ frame.left)
    anterior = delta @ frame.anterior
    return float(lateral.min()), float(anterior[int(np.argmin(np.abs(anterior)))])


def _plane_band(points: np.ndarray, origin: np.ndarray, superior: np.ndarray, half_mm: float) -> np.ndarray:
    if len(points) == 0:
        return points
    return points[np.abs((points - origin) @ superior) <= half_mm]


def _medial_gap(kidney_band: np.ndarray, body_band: np.ndarray, frame: VertebraAxes, side: str) -> Optional[float]:
    if len(kidney_band) == 0 or len(body_band) == 0:
        return None
    kidney_left = (kidney_band - frame.body_center) @ frame.left
    body_left = (body_band - frame.body_center) @ frame.left
    if side == "left":
        return float(kidney_left.min() - body_left.max())
    return float(body_left.min() - kidney_left.max())


def _psoas_on_plane(
    volume: AnatomyVolume, side: str, origin: np.ndarray, frame: VertebraAxes, half_mm: float
) -> Tuple[Optional[float], Optional[float], str]:
    major = volume.points(f"psoas_major_{side}")
    qc = "ok"
    points = major
    if len(points) == 0:
        points = volume.points(f"iliopsoas_{side}")
        qc = QC_APPROX
    band = _plane_band(points, origin, frame.superior, half_mm)
    if len(band) < 3:
        return None, None, QC_MISSING
    along = band @ frame.posterior
    thickness = float(along.max() - along.min())
    heights = band @ frame.superior
    span = max(float(heights.max() - heights.min()), float(np.min(volume.zooms)))
    area_cm2 = len(band) * float(np.prod(volume.zooms)) / span / 100.0
    return thickness, area_cm2, qc


def _empty_coordinates() -> Dict[str, object]:
    out: Dict[str, object] = {}
    for side in ("left", "right"):
        for third in THIRDS:
            out[f"kidney_{side}_{third}_vert_level"] = None
            out[f"kidney_{side}_{third}_vert_level_frac"] = None
            for level in Z_REFERENCES:
                out[z_reference_column(side, third, level)] = None
            for origin in ORIGINS:
                for kind in POINT_KINDS:
                    for axis in ("x", "y", "z"):
                        key = variant_column(side, third, axis, origin, kind)
                        out[key] = None
                        out[f"{key}_qc"] = QC_MISSING
        for suffix in (
            "perirenal_dorsal_vert_mm",
            "perirenal_ventral_vert_mm",
            "psoas_thickness_vert_mm",
            "psoas_area_vert_cm2",
            "medial_to_spine_vert_mm",
        ):
            out[f"kidney_{side}_{suffix}"] = None
            out[f"kidney_{side}_{suffix}_qc"] = QC_MISSING
    return out


def measure_lyashchenko(volume: AnatomyVolume) -> Dict[str, object]:
    """Four coordinate variants, vertebral level, and vertebra-aligned contacts."""
    out = _empty_coordinates()
    frames = build_vertebra_axes(volume)
    if not frames:
        return out
    ordered = list(frames.values())
    column = np.mean([frame.superior for frame in ordered], axis=0)
    column = column / np.linalg.norm(column)
    half_mm = max(1.0, float(np.min(volume.zooms)))

    for side in ("left", "right"):
        kidney = volume.points(f"kidney_{side}")
        bands = _split_thirds(kidney, column)
        for third, band in bands.items():
            center = band.mean(axis=0)
            frame = _frame_at(ordered, center)
            if frame is None:
                continue
            qc = frame.qc
            out[f"kidney_{side}_{third}_vert_level"] = frame.level
            out[f"kidney_{side}_{third}_vert_level_frac"] = _level_fraction(ordered, frame, center)
            for level in Z_REFERENCES:
                reference = frames.get(level)
                if reference is None:
                    continue
                out[z_reference_column(side, third, level)] = float(
                    (center - reference.body_center) @ reference.superior
                )
            for origin in ORIGINS:
                x_com, y_com, z_com = _centroid_xyz(center, frame, origin)
                x_near, y_near = _nearest_xy(band, frame, origin)
                values = {"com": (x_com, y_com, z_com), "near": (x_near, y_near, z_com)}
                for kind, (x_val, y_val, z_val) in values.items():
                    for axis, value in (("x", x_val), ("y", y_val), ("z", z_val)):
                        key = variant_column(side, third, axis, origin, kind)
                        out[key] = value
                        out[f"{key}_qc"] = qc

        middle = bands.get("middle")
        if middle is None:
            continue
        mid_point = middle.mean(axis=0)
        frame = _frame_at(ordered, mid_point)
        if frame is None:
            continue
        plane = _plane_band(kidney, mid_point, frame.superior, half_mm)
        if len(plane) < 5:
            plane = middle
        dorsal = _median_ray(volume, plane, frame.posterior) if volume.hu is not None else None
        ventral = _median_ray(volume, plane, frame.anterior) if volume.hu is not None else None
        contact_qc = frame.qc if frame.qc == "ok" else QC_APPROX
        out[f"kidney_{side}_perirenal_dorsal_vert_mm"] = dorsal
        out[f"kidney_{side}_perirenal_ventral_vert_mm"] = ventral
        out[f"kidney_{side}_perirenal_dorsal_vert_mm_qc"] = contact_qc if dorsal is not None else QC_MISSING
        out[f"kidney_{side}_perirenal_ventral_vert_mm_qc"] = contact_qc if ventral is not None else QC_MISSING

        thickness, area, psoas_qc = _psoas_on_plane(volume, side, mid_point, frame, half_mm)
        if psoas_qc == "ok" and frame.qc != "ok":
            psoas_qc = QC_APPROX
        out[f"kidney_{side}_psoas_thickness_vert_mm"] = thickness
        out[f"kidney_{side}_psoas_area_vert_cm2"] = area
        out[f"kidney_{side}_psoas_thickness_vert_mm_qc"] = psoas_qc if thickness is not None else QC_MISSING
        out[f"kidney_{side}_psoas_area_vert_cm2_qc"] = psoas_qc if area is not None else QC_MISSING

        body_name = f"vertebrae_{frame.level}"
        body_band = _plane_band(volume.points(body_name), mid_point, frame.superior, half_mm + 2.0)
        gap = _medial_gap(plane, body_band, frame, side)
        out[f"kidney_{side}_medial_to_spine_vert_mm"] = gap
        out[f"kidney_{side}_medial_to_spine_vert_mm_qc"] = contact_qc if gap is not None else QC_MISSING
    return out


def score_variant_pairs(
    pairs: Sequence[Tuple[dict, dict]],
    pose: str,
) -> List[dict]:
    """MAE of each origin/point variant, and of each vertical reference.

    ``pairs`` are ``(excel row, ct row)``. Excel keys are
    ``{side}_{pose}_{third}_{axis}``.
    """
    rows: List[dict] = []
    for origin in ORIGINS:
        for kind in POINT_KINDS:
            for axis in ("x", "y", "z"):
                for third in THIRDS:
                    manual: List[float] = []
                    auto: List[float] = []
                    for excel, ct in pairs:
                        for side in ("left", "right"):
                            manual.append(_number(excel.get(f"{side}_{pose}_{third}_{axis}")))
                            auto.append(_number(ct.get(variant_column(side, third, axis, origin, kind))))
                    rows.append(_score_row(f"{origin}_{kind}", "variant_z" if axis == "z" else "variant", axis, third, manual, auto))
    for level in Z_REFERENCES:
        for third in THIRDS:
            manual = []
            auto = []
            for excel, ct in pairs:
                for side in ("left", "right"):
                    manual.append(_number(excel.get(f"{side}_{pose}_{third}_z")))
                    auto.append(_number(ct.get(z_reference_column(side, third, level))))
            rows.append(_score_row(f"centroid_from_{level.lower()}", "spine_level", "z", third, manual, auto))
    return rows


def _number(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(number):
        return float("nan")
    return number


def _score_row(
    variant: str,
    z_source: str,
    axis: str,
    third: str,
    manual: Sequence[float],
    auto: Sequence[float],
) -> dict:
    manual_arr = np.asarray(manual, dtype=float)
    auto_arr = np.asarray(auto, dtype=float)
    ok = np.isfinite(manual_arr) & np.isfinite(auto_arr)
    diff = auto_arr[ok] - manual_arr[ok]
    if diff.size == 0:
        mae = bias = median_abs = float("nan")
        count = 0
    else:
        mae = float(np.mean(np.abs(diff)))
        bias = float(diff.mean())
        median_abs = float(np.median(np.abs(diff)))
        count = int(diff.size)
    return {
        "variant": variant,
        "z_source": z_source,
        "axis": axis,
        "third": third,
        "n": count,
        "mae": mae,
        "bias_ct_minus_excel": bias,
        "median_abs": median_abs,
    }
