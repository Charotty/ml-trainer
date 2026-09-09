"""Medical LPS/RAS coordinate helpers (production geometry contract).

This module used to live as ``src/coordinate_system.py``, which collided with
the ``src/coordinate_system/`` package. Import from ``src.coordinate_system``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np

PACKAGE_STATUS = "production"


class CoordinateSystem(Enum):
    LPS = "LPS"
    RAS = "RAS"
    PATIENT = "PATIENT"
    ANATOMICAL = "ANATOMICAL"


@dataclass
class CoordinateSystemDefinition:
    name: str
    description: str
    axis_directions: Dict[str, str]
    origin_description: str
    units: str = "mm"


class MedicalCoordinateSystem:
    SYSTEMS = {
        CoordinateSystem.LPS: CoordinateSystemDefinition(
            name="LPS (DICOM)",
            description="Left, Posterior, Superior - DICOM стандарт",
            axis_directions={
                "x": "from right towards left",
                "y": "from anterior towards posterior",
                "z": "from inferior towards superior",
            },
            origin_description="Произвольная точка, обычно центр изображения или геометрический центр сканера",
            units="mm",
        ),
        CoordinateSystem.RAS: CoordinateSystemDefinition(
            name="RAS (3D Slicer)",
            description="Right, Anterior, Superior - 3D Slicer стандарт",
            axis_directions={
                "x": "from left towards right",
                "y": "from posterior towards anterior",
                "z": "from inferior towards superior",
            },
            origin_description="Произвольная точка, обычно центр изображения",
            units="mm",
        ),
        CoordinateSystem.PATIENT: CoordinateSystemDefinition(
            name="Patient-Centric",
            description="Пациент-центричная система с началом в центре тела",
            axis_directions={
                "x": "from midline towards left (медиально-латерально)",
                "y": "from posterior towards anterior (антеро-постериорно)",
                "z": "from inferior towards superior (кранио-каудально)",
            },
            origin_description="Центр тела пациента (примерно пупок)",
            units="mm",
        ),
    }

    def __init__(self, current_system: CoordinateSystem = CoordinateSystem.LPS):
        self.current_system = current_system
        self.system_def = self.SYSTEMS[current_system]

    def get_system_info(self) -> CoordinateSystemDefinition:
        return self.system_def

    def transform_to_system(
        self,
        coordinates: np.ndarray,
        target_system: CoordinateSystem,
        transformation_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.current_system == target_system:
            return coordinates.copy()
        if self.current_system == CoordinateSystem.LPS and target_system == CoordinateSystem.RAS:
            transformed = coordinates.copy()
            transformed[:, 0] *= -1
            transformed[:, 1] *= -1
            return transformed
        if self.current_system == CoordinateSystem.RAS and target_system == CoordinateSystem.LPS:
            transformed = coordinates.copy()
            transformed[:, 0] *= -1
            transformed[:, 1] *= -1
            return transformed
        if transformation_matrix is not None:
            homogeneous = np.column_stack([coordinates, np.ones(len(coordinates))])
            transformed_h = (transformation_matrix @ homogeneous.T).T
            return transformed_h[:, :3]
        raise ValueError(f"Неизвестное преобразование: {self.current_system} -> {target_system}")

    def create_transformation_matrix(
        self,
        origin_source: np.ndarray,
        origin_target: np.ndarray,
        rotation_angles: Optional[Tuple[float, float, float]] = None,
    ) -> np.ndarray:
        matrix = np.eye(4)
        translation = origin_target - origin_source
        matrix[:3, 3] = translation
        if rotation_angles:
            rx, ry, rz = np.radians(rotation_angles)
            Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
            matrix[:3, :3] = Rz @ Ry @ Rx
        return matrix

    def validate_coordinates(self, coordinates: np.ndarray) -> Dict[str, Any]:
        results: Dict[str, Any] = {"valid": True, "warnings": [], "errors": [], "statistics": {}}
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            results["valid"] = False
            results["errors"].append(f"Неверная размерность: {coordinates.shape}, ожидается (N, 3)")
            return results
        nan_mask = np.isnan(coordinates)
        inf_mask = np.isinf(coordinates)
        if nan_mask.any():
            results["warnings"].append(f"Обнаружены NaN значения: {nan_mask.sum()} шт.")
        if inf_mask.any():
            results["warnings"].append(f"Обнаружены Inf значения: {inf_mask.sum()} шт.")
        results["statistics"] = {
            "num_points": len(coordinates),
            "x_range": (np.nanmin(coordinates[:, 0]), np.nanmax(coordinates[:, 0])),
            "y_range": (np.nanmin(coordinates[:, 1]), np.nanmax(coordinates[:, 1])),
            "z_range": (np.nanmin(coordinates[:, 2]), np.nanmax(coordinates[:, 2])),
            "mean_point": np.nanmean(coordinates, axis=0),
            "std_point": np.nanstd(coordinates, axis=0),
        }
        if self.current_system == CoordinateSystem.LPS:
            if (coordinates[:, 0] > 0).any():
                results["warnings"].append(
                    f"Точки с положительным X ({(coordinates[:, 0] > 0).sum()} шт.) - могут быть справа"
                )
            if (coordinates[:, 1] > 0).any():
                results["warnings"].append(
                    f"Точки с положительным Y ({(coordinates[:, 1] > 0).sum()} шт.) - могут быть сзади"
                )
            if (coordinates[:, 2] < 0).any():
                results["warnings"].append(
                    f"Точки с отрицательным Z ({(coordinates[:, 2] < 0).sum()} шт.) - могут быть снизу"
                )
        return results

    def get_anatomical_directions(self) -> Dict[str, str]:
        return self.system_def.axis_directions

    def print_system_info(self) -> None:
        print(f"=== {self.system_def.name} ===")
        print(f"Описание: {self.system_def.description}")
        print("Направления осей:")
        for axis, direction in self.system_def.axis_directions.items():
            print(f"  {axis.upper()}: {direction}")
        print(f"Начало координат: {self.system_def.origin_description}")
        print(f"Единицы: {self.system_def.units}")
        print()


def create_standard_coordinate_system() -> MedicalCoordinateSystem:
    return MedicalCoordinateSystem(CoordinateSystem.LPS)


def validate_and_fix_coordinates(
    coordinates: np.ndarray,
    system: MedicalCoordinateSystem,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    validation = system.validate_coordinates(coordinates)
    fixed_coords = coordinates.copy()
    nan_mask = np.isnan(fixed_coords)
    inf_mask = np.isinf(fixed_coords)
    if nan_mask.any():
        median_vals = np.nanmedian(fixed_coords, axis=0)
        for i in range(3):
            fixed_coords[nan_mask[:, i], i] = median_vals[i]
        validation["warnings"].append("NaN значения заменены на медианные")
    if inf_mask.any():
        fixed_coords[inf_mask] = 1e6
        validation["warnings"].append("Inf значения заменены на 1e6")
    return fixed_coords, validation
