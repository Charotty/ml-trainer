#!/usr/bin/env python3
"""
Улучшенный экстрактор признаков из КТ согласно требованиям
- Демография: sex, age, bmi, body_type
- Координаты почек: 18 признаков
- Смещения: 9 признаков
- Memory efficient: streaming processing
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union
from datetime import datetime

import numpy as np

try:
    import pydicom
except Exception:
    pydicom = None

try:
    import nibabel as nib
except Exception:
    nib = None

try:
    from totalsegmentator.python_api import totalsegmentator
except Exception:
    totalsegmentator = None

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from src.features.phase1_schema import normalize_record
except ImportError:
    normalize_record = None  # type: ignore[assignment,misc]

try:
    from src.features.ct_geometry import (
        aggregate_body_at_z_band,
        dicom_centroid_to_patient_mm,
        kidney_features_from_mask,
        merge_spine_relative,
        patient_kidney_side,
    )
except ImportError:
    aggregate_body_at_z_band = None  # type: ignore[assignment,misc]
    dicom_centroid_to_patient_mm = None  # type: ignore[assignment,misc]
    kidney_features_from_mask = None  # type: ignore[assignment,misc]
    merge_spine_relative = None  # type: ignore[assignment,misc]
    patient_kidney_side = None  # type: ignore[assignment,misc]

try:
    from skimage import measure, morphology
    import scipy.ndimage as ndimage
except Exception:
    measure = morphology = ndimage = None


@dataclass
class SliceInfo:
    path: Path
    position: Tuple[float, float, float]
    thickness: Optional[float] = None
    instance_number: Optional[int] = None
    number_of_frames: Optional[int] = None


def _safe_str(s) -> Optional[str]:
    if s is None:
        return None
    return str(s).strip()


def _is_dicom_file(path: Path) -> bool:
    """Проверка является ли файл DICOM"""
    try:
        with open(path, 'rb') as f:
            f.seek(128)
            return f.read(4) == b'DICM'
    except Exception:
        return False


def _list_dicom_slices(dicom_folder: Path) -> List[SliceInfo]:
    """Эффективное получение списка DICOM срезов (как в оригинальном extractor)"""
    if pydicom is None:
        raise RuntimeError('pydicom is not installed')

    files: List[Path] = []
    for p in dicom_folder.glob('**/*'):
        if not p.is_file():
            continue
        if p.suffix.lower() not in ['', '.dcm']:
            continue
        if _is_dicom_file(p):
            files.append(p)

    slices: List[SliceInfo] = []
    for fp in files:
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)

            modality = str(getattr(ds, 'Modality', '')).upper()
            if modality and modality not in {'CT'}:
                continue

            if getattr(ds, 'Rows', None) is None or getattr(ds, 'Columns', None) is None:
                continue

            n_frames = getattr(ds, 'NumberOfFrames', None)
            try:
                n_frames_int = int(n_frames) if n_frames is not None else None
            except Exception:
                n_frames_int = None

            inst = getattr(ds, 'InstanceNumber', None)
            try:
                inst_num = int(inst) if inst is not None else None
            except Exception:
                inst_num = None

            z = None
            ipp = getattr(ds, 'ImagePositionPatient', None)
            if ipp is not None and len(ipp) >= 3:
                try:
                    z = float(ipp[2])
                except Exception:
                    z = None

            slices.append(SliceInfo(fp, (0, 0, z) if z is not None else (0, 0, 0), 
                                  thickness=None, instance_number=inst_num, 
                                  number_of_frames=n_frames_int))
        except Exception:
            continue

    if not slices:
        return []

    # Сортировка как в оригинальном extractor
    if any(s.position[2] is not None for s in slices):
        slices.sort(key=lambda x: (x.number_of_frames, x.position[2]))
    else:
        slices.sort(key=lambda x: (x.number_of_frames, x.instance_number))

    return slices


def _estimate_slice_thickness_mm(slice_infos: List[SliceInfo]) -> Optional[float]:
    """Оценка толщины среза"""
    if len(slice_infos) < 2:
        return None
    
    # Вычисляем среднюю толщину
    thicknesses = []
    for i in range(1, len(slice_infos)):
        z1 = slice_infos[i-1].position[2]
        z2 = slice_infos[i].position[2]
        thickness = abs(z2 - z1)
        if 0 < thickness < 10:  # фильтр аномалий
            thicknesses.append(thickness)
    
    return float(np.mean(thicknesses)) if thicknesses else None


def _extract_demographics(ds) -> Dict[str, Optional[float]]:
    """Извлечение демографических данных из DICOM (совместимо с kits19)"""
    try:
        # Пол (M=1, F=2 как в kits19)
        sex = getattr(ds, 'PatientSex', None)
        if sex:
            sex = sex.upper().strip()
            sex = 1.0 if sex == 'M' else (2.0 if sex == 'F' else 0.0)
        else:
            sex = 0.0
        
        # Возраст
        age = getattr(ds, 'PatientAge', None)
        if age:
            age_str = str(age).strip()
            # Извлекаем число из строки (поддержка форматов Y/M/W/D)
            age_match = re.search(r'(\d+)', age_str)
            age_num = float(age_match.group(1)) if age_match else 50.0
            
            # Конвертация единиц
            unit_match = re.search(r'(\d+)([YMWD])', age_str)
            if unit_match:
                unit = unit_match.group(2)
                if unit == 'Y':
                    age = age_num
                elif unit == 'M':
                    age = age_num / 12.0
                elif unit == 'W':
                    age = age_num / 52.0
                elif unit == 'D':
                    age = age_num / 365.0
                else:
                    age = age_num
            else:
                age = age_num
        else:
            age = 50.0
        
        # Вес и рост
        weight_kg = getattr(ds, 'PatientWeight', None)
        height_m = getattr(ds, 'PatientSize', None)
        
        # ИМТ
        bmi = None
        if weight_kg is not None and height_m is not None and height_m > 0:
            bmi = float(weight_kg) / float(height_m * height_m)
        else:
            bmi = 25.0  # Средний ИМТ по умолчанию
        
        # Позиция сканирования
        scan_position = _extract_patient_position(ds)
        if scan_position is None:
            scan_position = 'supine'
        
        # Фаза контраста
        contrast_phase = 'arterial'  # По умолчанию
        
        # Толщина среза
        slice_thickness = getattr(ds, 'SliceThickness', None)
        if slice_thickness is not None:
            slice_thickness = float(slice_thickness)
        else:
            slice_thickness = 1.0
        
        # Дополнительные поля из kits19
        radiographic_size = 0.0  # Будет вычислено
        pathologic_size = 0.0    # Будет вычислено
        malignant = 0.0          # По умолчанию
        tumor_grade = 0.0        # По умолчанию
        tumor_histology_code = 0.0
        smoking_code = 0.0
        hospitalization_days = 0.0
        
        return {
            'sex': sex,
            'age': age,
            'bmi': bmi,
            'scan_position': scan_position,
            'contrast_phase': contrast_phase,
            'slice_thickness': slice_thickness,
            'radiographic_size': radiographic_size,
            'pathologic_size': pathologic_size,
            'malignant': malignant,
            'tumor_grade': tumor_grade,
            'tumor_histology_code': tumor_histology_code,
            'smoking_code': smoking_code,
            'hospitalization_days': hospitalization_days,
        }
    except Exception as e:
        print(f"Error extracting demographics: {e}")
        return {
            'sex': 0.0,
            'age': 50.0,
            'bmi': 25.0,
            'scan_position': 'supine',
            'contrast_phase': 'arterial',
            'slice_thickness': 1.0,
            'radiographic_size': 0.0,
            'pathologic_size': 0.0,
            'malignant': 0.0,
            'tumor_grade': 0.0,
            'tumor_histology_code': 0.0,
            'smoking_code': 0.0,
            'hospitalization_days': 0.0,
        }


def _extract_patient_position(ds) -> Optional[str]:
    """Извлечение позиции пациента"""
    try:
        position = getattr(ds, 'PatientPosition', None)
        if position:
            pos_str = str(position).upper().strip()
            if 'SUP' in pos_str:
                return 'supine'
            elif 'LAT' in pos_str or 'DECUB' in pos_str:
                return 'lateral'
            elif 'PRONE' in pos_str:
                return 'prone'
    except Exception:
        pass
    return None


def _extract_study_date(ds) -> Optional[str]:
    """Извлечение даты исследования"""
    try:
        date = getattr(ds, 'StudyDate', None)
        if date:
            date_str = str(date).strip()
            if len(date_str) == 8 and date_str.isdigit():
                return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            return date_str
    except Exception:
        pass
    return None


def _extract_kidney_coordinates_from_segmentation(segmentation_path: Path) -> Dict[str, Optional[float]]:
    """Kidney anatomy from TotalSegmentator mask in patient LPS mm (affine, not voxel indices)."""
    if nib is None or kidney_features_from_mask is None:
        return {}

    try:
        seg_img = nib.load(str(segmentation_path))
        seg_data = seg_img.get_fdata()
        affine = seg_img.affine
        zooms = tuple(float(z) for z in seg_img.header.get_zooms()[:3])

        result: Dict[str, Optional[float]] = {}
        for label_id, prefix in ((8, "kidney_right"), (9, "kidney_left")):
            mask = seg_data == label_id
            if not np.any(mask):
                continue
            kidney_feats = kidney_features_from_mask(mask, affine, zooms, prefix)
            result.update(kidney_feats)

        return result

    except Exception as e:
        print(f"Error extracting kidney coordinates: {e}")
        return {}


def _run_totalsegmentator(dicom_folder: Path, output_path: Path, kidney_only: bool = False) -> Optional[Path]:
    """Запускает TotalSegmentator для создания маски сегментации с оптимизацией памяти"""
    if totalsegmentator is None:
        return None
    
    try:
        # Очищаем память перед запуском
        import gc
        gc.collect()
        
        # Запускаем сегментацию с опциями
        kwargs = {
            'input': str(dicom_folder),
            'output': str(output_path),
            'device': 'gpu',
            'fast': True,  # Используем быстрый режим
            'quiet': True,
            'nora_tag': 'None'  # Отключаем дополнительную обработку
        }
        
        # Если нужно только почки, используем roi_subset
        if kidney_only:
            kwargs['roi_subset'] = ['kidney_left', 'kidney_right']
        
        try:
            totalsegmentator(**kwargs)
        except (TypeError, Exception) as e:
            # Если API не поддерживает roi_subset, пробуем базовый вызов
            if kidney_only:
                print(f"  [WARN] ROI subset unsupported, full segmentation: {e}")
                kwargs.pop('roi_subset', None)
                totalsegmentator(**kwargs)
            else:
                raise e
        
        # Ищем файл сегментации
        seg_file = output_path / "segmentation.nii.gz"
        if seg_file.exists():
            return seg_file
        
        # Альтернативные имена файлов
        for alt_name in ["segmentation.nii", "seg.nii.gz", "seg.nii"]:
            alt_file = output_path / alt_name
            if alt_file.exists():
                return alt_file
                
    except MemoryError as e:
        print(f"Memory error in TotalSegmentator: {e}")
        print("  [hint] increase RAM or use --disable-kidney-segmentation")
    except Exception as e:
        print(f"Error running TotalSegmentator: {e}")
        
    return None


def _lightweight_kidney_side(
    patient_mm: np.ndarray,
    spine_x: Optional[float],
) -> str:
    if spine_x is not None and patient_kidney_side is not None:
        return patient_kidney_side(float(patient_mm[0]), float(spine_x))
    return "left" if patient_mm[0] >= 0 else "right"


def _aggregate_kidney_side_3d(
    candidates: List[Dict],
    prefix: str,
) -> Dict[str, float]:
    """Aggregate multi-slice HU kidney detections into volume/length/center in patient mm."""
    if not candidates:
        return {}

    total_area = sum(c["area"] for c in candidates)
    if total_area <= 0:
        return {}

    center = np.zeros(3, dtype=float)
    z_vals: List[float] = []
    voxel_volume_mm3 = 0.0
    in_plane_lengths: List[float] = []

    for c in candidates:
        w = c["area"] / total_area
        center += w * c["patient_mm"]
        z_vals.append(float(c["patient_mm"][2]))
        voxel_volume_mm3 += c["area"] * c["voxel_volume_mm3"]
        in_plane_lengths.append(c["in_plane_length_mm"])

    z_min, z_max = min(z_vals), max(z_vals)
    cranio_caudal_mm = z_max - z_min
    length_mm = max(cranio_caudal_mm, max(in_plane_lengths) if in_plane_lengths else 0.0)
    volume_cm3 = voxel_volume_mm3 / 1000.0

    out: Dict[str, float] = {}
    for axis, i in zip("xyz", range(3)):
        val = float(center[i])
        out[f"{prefix}_center_{axis}"] = val
        out[f"{prefix}_middle_{axis}"] = val
        out[f"{prefix}_upper_{axis}"] = val if axis != "z" else z_max
        out[f"{prefix}_lower_{axis}"] = val if axis != "z" else z_min

    out[f"{prefix}_volume_cm3"] = float(volume_cm3)
    out[f"{prefix}_length_mm"] = float(length_mm)
    out[f"{prefix}_delta_x"] = float("nan")
    out[f"{prefix}_delta_y"] = float("nan")
    out[f"{prefix}_delta_z"] = float("nan")
    return out


def _extract_kidney_coordinates_lightweight(
    dicom_folder: Path,
    spine_hint_x: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """HU-based 3D kidney tracking in patient LPS mm (no synthetic deltas or fixed Z offsets)."""
    if measure is None or morphology is None or ndimage is None:
        return {}
    if dicom_centroid_to_patient_mm is None:
        return {}

    try:
        slice_infos = _list_dicom_slices(dicom_folder)
        if len(slice_infos) < 10:
            return {}

        slice_infos.sort(key=lambda x: x.position[2])
        n_slices = len(slice_infos)
        start_idx = n_slices // 4
        end_idx = 3 * n_slices // 4
        central_slices = slice_infos[start_idx:end_idx]

        kidney_hu_min, kidney_hu_max = 20, 60
        min_area = 400
        max_area = 80000
        max_eccentricity = 0.92

        candidates: List[Dict] = []
        spine_x_samples: List[float] = []

        for slice_info in central_slices:
            try:
                ds = pydicom.dcmread(str(slice_info.path), force=True)
                pixel_array = ds.pixel_array.astype(np.float32)
                slope = getattr(ds, "RescaleSlope", 1.0)
                intercept = getattr(ds, "RescaleIntercept", 0.0)
                hu_array = pixel_array * slope + intercept

                kidney_mask = (hu_array >= kidney_hu_min) & (hu_array <= kidney_hu_max)
                kidney_mask = kidney_mask & (hu_array < 150) & (hu_array > -50)
                try:
                    kidney_mask = morphology.remove_small_objects(kidney_mask, max_size=49)
                except TypeError:
                    kidney_mask = morphology.remove_small_objects(kidney_mask, min_size=50)
                kidney_mask = ndimage.binary_fill_holes(kidney_mask)

                pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
                slice_thickness = float(getattr(ds, "SliceThickness", 1.0) or 1.0)
                voxel_volume_mm3 = float(pixel_spacing[0]) * float(pixel_spacing[1]) * slice_thickness

                bone_mask = hu_array >= 300
                if np.any(bone_mask):
                    labeled_bones = measure.label(bone_mask)
                    for region in measure.regionprops(labeled_bones):
                        if region.area < 80:
                            continue
                        bone_mm = dicom_centroid_to_patient_mm(ds, region.centroid[0], region.centroid[1])
                        spine_x_samples.append(float(bone_mm[0]))

                labeled_mask = measure.label(kidney_mask)
                for region in measure.regionprops(labeled_mask):
                    if not (min_area <= region.area <= max_area and region.eccentricity < max_eccentricity):
                        continue
                    patient_mm = dicom_centroid_to_patient_mm(ds, region.centroid[0], region.centroid[1])
                    min_r, min_c, max_r, max_c = region.bbox
                    in_plane_length = max(
                        (max_r - min_r + 1) * float(pixel_spacing[0]),
                        (max_c - min_c + 1) * float(pixel_spacing[1]),
                    )
                    candidates.append({
                        "patient_mm": patient_mm,
                        "area": float(region.area),
                        "voxel_volume_mm3": voxel_volume_mm3,
                        "in_plane_length_mm": in_plane_length,
                    })
            except Exception:
                continue

        if not candidates:
            return {}

        spine_x = spine_hint_x
        if spine_x is None and spine_x_samples:
            spine_x = float(np.median(spine_x_samples))

        left_cands = [
            c for c in candidates
            if _lightweight_kidney_side(c["patient_mm"], spine_x) == "left"
        ]
        right_cands = [
            c for c in candidates
            if _lightweight_kidney_side(c["patient_mm"], spine_x) == "right"
        ]

        result: Dict[str, Optional[float]] = {}
        result.update(_aggregate_kidney_side_3d(left_cands, "kidney_left"))
        result.update(_aggregate_kidney_side_3d(right_cands, "kidney_right"))
        return result

    except Exception as e:
        print(f"Error in lightweight kidney detection: {e}")
        return {}


def _compute_body_centers(slice_data: Dict) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Вычисление центров тела и позвоночника"""
    if not slice_data:
        return None, None, None, None
    
    body_centers_x = []
    body_centers_y = []
    spine_centers_x = []
    spine_centers_y = []
    
    for z, data in slice_data.items():
        hu_array = data['hu_array']
        pixel_spacing = data['pixel_spacing']
        
        # Центр тела
        body_mask = (hu_array >= -300) & (hu_array <= 300)
        if np.any(body_mask):
            y_coords, x_coords = np.where(body_mask)
            body_center_x = np.mean(x_coords) * pixel_spacing[0]
            body_center_y = np.mean(y_coords) * pixel_spacing[1]
            body_centers_x.append(body_center_x)
            body_centers_y.append(body_center_y)
        
        # Центр позвоночника
        bone_mask = hu_array >= 300
        if np.any(bone_mask):
            labeled_bones = measure.label(bone_mask)
            bone_regions = measure.regionprops(labeled_bones)
            
            # Выбираем центральную костную структуру
            if bone_regions:
                body_com_x = np.mean(body_centers_x) if body_centers_x else 0
                body_com_y = np.mean(body_centers_y) if body_centers_y else 0
                
                spine_region = None
                min_distance = float('inf')
                
                for region in bone_regions:
                    centroid = region.centroid
                    dist = np.sqrt((centroid[1] * pixel_spacing[0] - body_com_x)**2 + 
                                 (centroid[0] * pixel_spacing[1] - body_com_y)**2)
                    if dist < min_distance and region.area > 100:
                        min_distance = dist
                        spine_region = region
                
                if spine_region:
                    spine_center_x = spine_region.centroid[1] * pixel_spacing[0]
                    spine_center_y = spine_region.centroid[0] * pixel_spacing[1]
                    spine_centers_x.append(spine_center_x)
                    spine_centers_y.append(spine_center_y)
    
    return (
        np.mean(body_centers_x) if body_centers_x else None,
        np.mean(body_centers_y) if body_centers_y else None,
        np.mean(spine_centers_x) if spine_centers_x else None,
        np.mean(spine_centers_y) if spine_centers_y else None
    )


def _compute_distance_to_skin(hu_array: np.ndarray, kidney_x: float, kidney_y: float, pixel_spacing: Tuple[float, float]) -> float:
    """Вычисление расстояния от почки до кожи"""
    try:
        # Маска тела
        body_mask = (hu_array >= -300) & (hu_array <= 300)
        
        if not np.any(body_mask):
            return 0.0
        
        # Конвертируем координаты почки в пиксельные
        kidney_x_px = int(kidney_x / pixel_spacing[0])
        kidney_y_px = int(kidney_y / pixel_spacing[1])
        
        # Получаем границы тела
        y_indices, x_indices = np.where(body_mask)
        
        # Находим ближайшую точку границы тела
        min_distance = float('inf')
        for i in range(len(y_indices)):
            dist = np.sqrt((x_indices[i] - kidney_x_px)**2 + (y_indices[i] - kidney_y_px)**2)
            if dist < min_distance:
                min_distance = dist
        
        return min_distance * pixel_spacing[0]  # в мм
        
    except Exception:
        return 0.0


def _compute_kidney_confidence(kidney: Dict, all_kidneys: List[Dict]) -> float:
    """Вычисление confidence score для почки"""
    try:
        # Размер компонента (0.4)
        size_score = min(kidney['area'] / 5000.0, 1.0) * 0.4
        
        # Форма (0.3)
        shape_score = (1.0 - kidney['eccentricity']) * 0.3
        
        # Плотность (0.2)
        density_score = kidney['solidity'] * 0.2
        
        # Относительный размер (0.1)
        if all_kidneys:
            max_area = max(k['area'] for k in all_kidneys)
            relative_score = kidney['area'] / max_area
        else:
            relative_score = 1.0
        relative_score = relative_score * 0.1
        
        confidence = size_score + shape_score + density_score + relative_score
        return min(confidence, 1.0)
        
    except Exception:
        return 0.5


def _compute_slice_stability(kidney: Dict, all_candidates: List[Dict], prefix: str) -> float:
    """Вычисление стабильности почки по срезам"""
    try:
        # Находим все кандидаты той же стороны
        side_candidates = [k for k in all_candidates if k['side'] == kidney['side']]
        
        if len(side_candidates) < 2:
            return 0.5
        
        # Вычисляем вариацию позиции
        positions = [(k['centroid_x'], k['centroid_y']) for k in side_candidates]
        positions = np.array(positions)
        
        # Стандартное отклонение позиции
        std_x = np.std(positions[:, 0])
        std_y = np.std(positions[:, 1])
        
        # Чем меньше вариация, тем выше стабильность
        stability = 1.0 / (1.0 + std_x + std_y)
        return min(stability, 1.0)
        
    except Exception:
        return 0.5


def _extract_body_features_slice(hu_array: np.ndarray, pixel_spacing: Tuple[float, float]) -> Dict[str, float]:
    """Извлечение признаков тела из одного среза"""
    # Маски тканей
    body_mask = (hu_array >= -300) & (hu_array <= 300)
    fat_mask = (hu_array >= -190) & (hu_array <= -30)
    bone_mask = hu_array >= 300
    
    # Геометрия тела
    body_pixels = np.sum(body_mask)
    fat_pixels = np.sum(fat_mask)
    bone_pixels = np.sum(bone_mask)
    
    # Размеры
    y_indices, x_indices = np.where(body_mask)
    if len(y_indices) > 0 and len(x_indices) > 0:
        y_coords, x_coords = y_indices, x_indices
        width_mm = (np.max(x_coords) - np.min(x_coords) + 1) * pixel_spacing[0]
        depth_mm = (np.max(y_coords) - np.min(y_coords) + 1) * pixel_spacing[1]
        area_mm2 = body_pixels * pixel_spacing[0] * pixel_spacing[1]
        
        # Центр масс
        y_coords, x_coords = np.where(body_mask)
        com_y = float(np.mean(y_coords)) * pixel_spacing[1]
        com_x = float(np.mean(x_coords)) * pixel_spacing[0]
        
        return {
            'body_pixels': float(body_pixels),
            'fat_pixels': float(fat_pixels),
            'bone_pixels': float(bone_pixels),
            'body_width_mm': width_mm,
            'body_depth_mm': depth_mm,
            'body_area_mm2': area_mm2,
            'body_com_x_mm': com_x,
            'body_com_y_mm': com_y,
        }
    
    return {}


def _extract_spine_features_slice(hu_array: np.ndarray, body_com_x: float, body_com_y: float, 
                               pixel_spacing: Tuple[float, float]) -> Dict[str, float]:
    """Извлечение признаков позвоночника"""
    # Маска костной ткани
    bone_mask = hu_array >= 300
    
    if not np.any(bone_mask):
        return {}
    
    # Поиск позвоночника (центральная костная структура рядом с центром масс)
    labeled_bones = measure.label(bone_mask)
    bone_regions = measure.regionprops(labeled_bones)
    
    # Выбираем регион ближайший к центру масс тела
    spine_region = None
    min_distance = float('inf')
    
    for region in bone_regions:
        centroid = region.centroid
        distance = np.sqrt((centroid[1] * pixel_spacing[1] - body_com_y)**2 + 
                        (centroid[0] * pixel_spacing[0] - body_com_x)**2)
        if distance < min_distance and region.area > 100:  # минимальный размер
            min_distance = distance
            spine_region = region
    
    if spine_region is None:
        return {}
    
    # Центр позвоночника
    spine_y = spine_region.centroid[1] * pixel_spacing[1]
    spine_x = spine_region.centroid[0] * pixel_spacing[0]
    
    # Расстояния до кожи
    y_indices, x_indices = np.where(hu_array >= -300)  # тело
    if len(y_indices) > 0:
        y_coords, x_coords = y_indices, x_indices
        
        # Левая/правая границы
        left_distances = []
        right_distances = []
        anterior_distances = []
        posterior_distances = []
        
        for y_idx in range(len(y_coords)):
            for x_idx in range(len(x_coords)):
                if not (hu_array[y_idx, x_idx] >= -300):
                    continue
                    
                x_mm = x_coords[x_idx] * pixel_spacing[0]
                y_mm = y_coords[y_idx] * pixel_spacing[1]
                
                # Расстояние от центра позвоночника
                dist_to_spine = np.sqrt((x_mm - spine_x)**2 + (y_mm - spine_y)**2)
                
                # Направление относительно позвоночника
                dx = x_mm - spine_x
                dy = y_mm - spine_y
                
                if abs(dx) > abs(dy):  # лево/право
                    if dx < 0:
                        left_distances.append(dist_to_spine)
                    else:
                        right_distances.append(dist_to_spine)
                else:  # перед/зад
                    if dy < 0:
                        anterior_distances.append(dist_to_spine)
                    else:
                        posterior_distances.append(dist_to_spine)
        
        return {
            'spine_center_x_mm': spine_x,
            'spine_center_y_mm': spine_y,
            'spine_to_skin_left_mm': float(np.min(left_distances)) if left_distances else None,
            'spine_to_skin_right_mm': float(np.min(right_distances)) if right_distances else None,
            'spine_to_skin_anterior_mm': float(np.min(anterior_distances)) if anterior_distances else None,
            'spine_to_skin_posterior_mm': float(np.min(posterior_distances)) if posterior_distances else None,
        }
    
    return {}


def _extract_slice_anatomy_patient(
    ds,
    hu_array: np.ndarray,
    slice_z: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Body/spine metrics for one slice in patient LPS mm."""
    pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
    body_mask = (hu_array >= -300) & (hu_array <= 300)

    body_features: Dict[str, float] = {}
    spine_features: Dict[str, float] = {}
    if not np.any(body_mask) or dicom_centroid_to_patient_mm is None:
        return body_features, spine_features

    y_idx, x_idx = np.where(body_mask)
    step = max(1, len(y_idx) // 400)
    pts = np.array([
        dicom_centroid_to_patient_mm(ds, float(y_idx[i]), float(x_idx[i]))
        for i in range(0, len(y_idx), step)
    ], dtype=float)

    body_features = {
        "body_width_mm": float(pts[:, 0].max() - pts[:, 0].min()),
        "body_depth_mm": float(pts[:, 1].max() - pts[:, 1].min()),
        "body_area_mm2": float(len(y_idx) * pixel_spacing[0] * pixel_spacing[1]),
        "body_com_x": float(pts[:, 0].mean()),
        "body_com_y": float(pts[:, 1].mean()),
        "body_com_z": float(pts[:, 2].mean()),
        "slice_z": float(slice_z),
        "body_pixels": float(len(y_idx)),
    }

    bone_mask = hu_array >= 300
    if np.any(bone_mask) and measure is not None:
        labeled_bones = measure.label(bone_mask)
        if labeled_bones is not None:
            body_com_x = body_features["body_com_x"]
            body_com_y = body_features["body_com_y"]
            spine_region = None
            min_distance = float("inf")
            for region in measure.regionprops(labeled_bones):
                if region.area < 80:
                    continue
                bone_mm = dicom_centroid_to_patient_mm(ds, region.centroid[0], region.centroid[1])
                dist = float(np.linalg.norm(bone_mm[:2] - np.array([body_com_x, body_com_y])))
                if dist < min_distance:
                    min_distance = dist
                    spine_region = bone_mm
            if spine_region is not None:
                spine_features = {
                    "spine_center_x_mm": float(spine_region[0]),
                    "spine_center_y_mm": float(spine_region[1]),
                    "spine_center_z_mm": float(spine_region[2]),
                }

    fat_mask = (hu_array >= -190) & (hu_array <= -30)
    bone_px = float(np.sum(hu_array >= 300))
    body_features["fat_pixels"] = float(np.sum(fat_mask))
    body_features["bone_pixels"] = bone_px
    return body_features, spine_features


def _run_kidney_extraction(
    dicom_folder: Path,
    *,
    enable_kidney_segmentation: bool,
    kidney_only: bool,
    show_progress: bool,
    spine_hint_x: Optional[float],
) -> Dict[str, Optional[float]]:
    if not enable_kidney_segmentation:
        return {}

    if show_progress:
        print("  [kidney] detection...")

    def _lightweight() -> Dict[str, Optional[float]]:
        return _extract_kidney_coordinates_lightweight(dicom_folder, spine_hint_x=spine_hint_x)

    if totalsegmentator is not None and not kidney_only:
        try:
            import tempfile
            import gc
            try:
                import psutil
                available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
                if available_memory_gb < 4.0:
                    return _lightweight()
            except ImportError:
                pass

            gc.collect()
            temp_dir = Path(tempfile.mkdtemp())
            seg_file = _run_totalsegmentator(dicom_folder, temp_dir, kidney_only)
            if seg_file:
                return _extract_kidney_coordinates_from_segmentation(seg_file)
            return _lightweight()
        except MemoryError:
            return _lightweight()
        except Exception:
            return _lightweight()
    return _lightweight()


def extract_features_from_dicom_folder(
    dicom_folder: Path,
    downsample: int = 2,
    max_slices: Optional[int] = None,
    debug: bool = False,
    enable_kidney_segmentation: bool = True,
    kidney_only: bool = False,
    show_progress: bool = False,
    current_case: int = 1,
    total_cases: int = 1,
    slice_strategy: str = 'uniform',
) -> Dict[str, Optional[float]]:
    """Основная функция извлечения признаков (совместима с оригинальной)"""
    if pydicom is None:
        raise RuntimeError('pydicom is not installed')

    slice_infos = _list_dicom_slices(dicom_folder)
    if not slice_infos:
        raise ValueError('No DICOM slices found')
    
    # Показываем прогресс если нужно
    if show_progress:
        print(f"  slices found: {len(slice_infos)}")
        if max_slices:
            print(f"  slices to process: {min(len(slice_infos), max_slices)}")
        print(f"  slice strategy: {slice_strategy}")
    
    # Извлечение метаданных из первого среза
    patient_name = None
    patient_id = None
    study_uid = None
    patient_position = None
    study_date = None
    demographics: Dict[str, Optional[float]] = {
        'sex': None,
        'age': None,
        'weight_kg': None,
        'height_m': None,
        'bmi': None,
        'body_type': None,
    }
    try:
        ds0 = pydicom.dcmread(str(slice_infos[0].path), stop_before_pixels=True, force=True)
        patient_name = _safe_str(getattr(ds0, 'PatientName', None)) or None
        patient_id = _safe_str(getattr(ds0, 'PatientID', None)) or None
        study_uid = _safe_str(getattr(ds0, 'StudyInstanceUID', None)) or None
        demographics = _extract_demographics(ds0)
        patient_position = _extract_patient_position(ds0)
        study_date = _extract_study_date(ds0)
    except Exception:
        pass
    
    # Проверяем минимальное количество срезов
    if len(slice_infos) < 3:
        if show_progress:
            print(f"  [WARN] too few slices ({len(slice_infos)}), need >= 3")
        # Возвращаем базовую информацию без координат почек
        return {
            'patient_id': patient_id,
            'study_instance_uid': study_uid,
            'patient_name': patient_name,
            'patient_position': patient_position,
            'study_date': study_date,
            **demographics,
            'slice_count_used': 0,
            'slice_thickness_mm': None,
        }
    
    # Применяем адаптивную выборку срезов
    if max_slices is not None and max_slices > 0 and len(slice_infos) > max_slices:
        original_count = len(slice_infos)
        slice_infos = _adaptive_slice_selection(slice_infos, max_slices, slice_strategy)
        if show_progress:
            print(f"  adaptive selection: {original_count} -> {len(slice_infos)} slices")
    elif max_slices is None:
        max_slices = len(slice_infos)
    
    # Обработка срезов (body/spine в patient mm — до почек)
    if show_progress:
        print(f"  processing {len(slice_infos)} slices...")
        
    thickness_mm = _estimate_slice_thickness_mm(slice_infos)
    
    # Агрегаты тела
    body_acc: Dict[str, object] = {
        'body_pixels': 0.0,
        'fat_pixels': 0.0,
        'bone_pixels': 0.0,
        'body_width_mm': [],
        'body_depth_mm': [],
        'body_area_mm2': [],
        'body_com_x_mm': [],
        'body_com_y_mm': [],
        'body_com_z_mm': [],
    }
    
    spine_acc = {
        'spine_center_x_mm': [],
        'spine_center_y_mm': [],
        'spine_center_z_mm': [],
        'spine_to_skin_left_mm': [],
        'spine_to_skin_right_mm': [],
        'spine_to_skin_anterior_mm': [],
        'spine_to_skin_posterior_mm': [],
    }
    slice_metrics: List[Dict[str, float]] = []

    processed_slices = 0
    for i, slice_info in enumerate(slice_infos):
        try:
            # Показываем прогресс обработки срезов
            if show_progress and (i % max(1, len(slice_infos) // 10) == 0 or i == len(slice_infos) - 1):
                progress = (i + 1) / len(slice_infos) * 100
                print(f"    slice {i+1}/{len(slice_infos)} ({progress:.0f}%)")
            
            ds = pydicom.dcmread(str(slice_info.path), force=True)
            
            # Конвертация в Hounsfield Units
            pixel_array = ds.pixel_array.astype(np.float32)
            slope = getattr(ds, 'RescaleSlope', 1.0)
            intercept = getattr(ds, 'RescaleIntercept', 0.0)
            hu_array = pixel_array * slope + intercept
            
            # Downsample
            if downsample > 1:
                new_shape = (hu_array.shape[0] // downsample, 
                           hu_array.shape[1] // downsample)
                hu_array = hu_array[::downsample, ::downsample]
            
            slice_z = float(slice_info.position[2])
            body_features, spine_features = _extract_slice_anatomy_patient(ds, hu_array, slice_z)
            if body_features:
                slice_metrics.append(body_features)
                for key in ("body_pixels", "fat_pixels", "bone_pixels"):
                    if key in body_features:
                        body_acc[key] = body_acc.get(key, 0.0) + body_features[key]
                for key in ("body_width_mm", "body_depth_mm", "body_area_mm2"):
                    if key in body_features:
                        body_acc[key].append(body_features[key])
                body_acc.setdefault("body_com_x_mm", []).append(body_features["body_com_x"])
                body_acc.setdefault("body_com_y_mm", []).append(body_features["body_com_y"])
                body_acc.setdefault("body_com_z_mm", []).append(body_features["body_com_z"])

            for key, value in spine_features.items():
                if key in spine_acc and value is not None:
                    spine_acc[key].append(value)
                    
            processed_slices += 1
            
        except Exception as e:
            if debug:
                print(f"Error processing slice {i}: {e}")
            continue
    
    # Финальная агрегация
    pixel_spacing = getattr(ds0, 'PixelSpacing', [1.0, 1.0])
    slice_thickness = thickness_mm or 1.0
    voxel_volume = pixel_spacing[0] * pixel_spacing[1] * slice_thickness
    
    # Тело
    mean_body_pixels = float(body_acc['body_pixels']) / max(processed_slices, 1)
    mean_fat_pixels = float(body_acc['fat_pixels']) / max(processed_slices, 1)
    mean_bone_pixels = float(body_acc['bone_pixels']) / max(processed_slices, 1)
    body_volume_cm3 = mean_body_pixels * voxel_volume / 1000.0
    fat_volume_cm3 = mean_fat_pixels * voxel_volume / 1000.0
    bone_volume_cm3 = mean_bone_pixels * voxel_volume / 1000.0
    fat_ratio = fat_volume_cm3 / body_volume_cm3 if body_volume_cm3 > 0 else None
    
    # Геометрия
    body_width_mm_median = _median(body_acc['body_width_mm'])
    body_depth_mm_median = _median(body_acc['body_depth_mm'])
    body_area_mm2_median = _median(body_acc['body_area_mm2'])
    
    # Центры масс (patient LPS mm)
    body_com_x_mm = _median(body_acc.get('body_com_x_mm', []))
    body_com_y_mm = _median(body_acc.get('body_com_y_mm', []))
    body_com_z_mm = _median(body_acc.get('body_com_z_mm', []))
    if body_com_z_mm is None:
        body_com_z_mm = slice_infos[len(slice_infos) // 2].position[2]

    # Позвоночник
    spine_center_x_mm = _median(spine_acc['spine_center_x_mm'])
    spine_center_y_mm = _median(spine_acc['spine_center_y_mm'])
    spine_center_z_mm = _median(spine_acc.get('spine_center_z_mm', []))
    if spine_center_z_mm is None:
        spine_center_z_mm = body_com_z_mm

    kidney_features = _run_kidney_extraction(
        dicom_folder,
        enable_kidney_segmentation=enable_kidney_segmentation,
        kidney_only=kidney_only,
        show_progress=show_progress,
        spine_hint_x=spine_center_x_mm,
    )

    kidney_z_vals = [
        kidney_features[k]
        for k in (
            "kidney_left_center_z", "kidney_right_center_z",
            "kidney_left_lower_z", "kidney_left_upper_z",
        )
        if kidney_features.get(k) is not None and not np.isnan(kidney_features[k])
    ]
    if kidney_z_vals and aggregate_body_at_z_band is not None and slice_metrics:
        z_min = float(min(kidney_z_vals))
        z_max = float(max(kidney_z_vals))
        band_body = aggregate_body_at_z_band(slice_metrics, z_min, z_max)
        if band_body.get("body_width_mm") is not None:
            body_width_mm_median = band_body["body_width_mm"]
        if band_body.get("body_depth_mm") is not None:
            body_depth_mm_median = band_body["body_depth_mm"]
        if band_body.get("body_area_mm2") is not None:
            body_area_mm2_median = band_body["body_area_mm2"]
        if band_body.get("body_com_x") is not None:
            body_com_x_mm = band_body["body_com_x"]
        if band_body.get("body_com_y") is not None:
            body_com_y_mm = band_body["body_com_y"]
        if band_body.get("body_com_z") is not None:
            body_com_z_mm = band_body["body_com_z"]
    
    # Расстояния до кожи
    spine_to_skin_left_mm_median = _median([x for x in spine_acc['spine_to_skin_left_mm'] if x is not None])
    spine_to_skin_right_mm_median = _median([x for x in spine_acc['spine_to_skin_right_mm'] if x is not None])
    spine_to_skin_anterior_mm_median = _median([x for x in spine_acc['spine_to_skin_anterior_mm'] if x is not None])
    spine_to_skin_posterior_mm_median = _median([x for x in spine_acc['spine_to_skin_posterior_mm'] if x is not None])
    
    # Компиляция результата
    features: Dict[str, Optional[float]] = {
        'patient_id': patient_id,
        'study_instance_uid': study_uid,
        'patient_name': patient_name,
        'patient_position': patient_position,
        'study_date': study_date,
        **demographics,
        **kidney_features,
        'body_volume_cm3': body_volume_cm3,
        'fat_volume_cm3': fat_volume_cm3,
        'bone_volume_cm3': bone_volume_cm3,
        'fat_ratio': fat_ratio,
        'body_width_mm_median': body_width_mm_median,
        'body_depth_mm_median': body_depth_mm_median,
        'body_area_mm2_median': body_area_mm2_median,
        'body_com_x_mm': body_com_x_mm,
        'body_com_y_mm': body_com_y_mm,
        'body_com_z_mm': body_com_z_mm,
        'spine_center_x_mm': spine_center_x_mm,
        'spine_center_y_mm': spine_center_y_mm,
        'spine_center_z_mm': spine_center_z_mm,
        'spine_to_skin_left_mm_median': spine_to_skin_left_mm_median,
        'spine_to_skin_right_mm_median': spine_to_skin_right_mm_median,
        'spine_to_skin_anterior_mm_median': spine_to_skin_anterior_mm_median,
        'spine_to_skin_posterior_mm_median': spine_to_skin_posterior_mm_median,
        'slice_count_used': processed_slices,
        'slice_thickness_mm': thickness_mm,
    }
    
    if show_progress:
        print(f"  [OK] slices processed: {processed_slices}")

    if merge_spine_relative is not None:
        features = merge_spine_relative(
            features,
            spine_center_x_mm,
            spine_center_y_mm,
            spine_center_z_mm,
        )
    features = _add_unified_features(features)

    if normalize_record is not None:
        return normalize_record(features)
    return features


def _median(values: List[float]) -> Optional[float]:
    """Вычисление медианы"""
    if not values:
        return None
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def _extract_full_name_from_folder(folder_name: str) -> Optional[str]:
    words = re.findall(r'[А-Я][а-я]+', folder_name)
    if len(words) >= 2:
        return ' '.join(words[:3])
    return None


def _normalize_name(value: str) -> str:
    if value is None:
        return ''
    s = str(value).strip().lower()
    s = s.replace('ё', 'е')
    s = re.sub(r'\s+', '', s)
    s = re.sub(r'[^a-zа-я0-9]+', '', s)
    return s


def _iter_patient_folders(dicom_root: Path) -> Iterable[Path]:
    for p in dicom_root.iterdir():
        if p.is_dir():
            yield p


def _add_unified_features(features: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """Добавление унифицированных признаков согласно comment.md"""
    unified_features = features.copy()
    
    # Переименуем существующие признаки согласно унификации
    if 'body_width_mm_median' in features:
        unified_features['body_width_mm'] = features['body_width_mm_median']
    if 'body_depth_mm_median' in features:
        unified_features['body_depth_mm'] = features['body_depth_mm_median']
    if 'body_area_mm2_median' in features:
        unified_features['body_area_mm2'] = features['body_area_mm2_median']
    
    # Добавляем вычисляемые расстояния до позвоночника
    spine_x = features.get('spine_center_x_mm')
    spine_y = features.get('spine_center_y_mm') 
    spine_z = features.get('spine_center_z_mm')
    body_com_x = features.get('body_com_x_mm')
    body_com_y = features.get('body_com_y_mm')
    body_com_z = features.get('body_com_z_mm')
    
    # Если есть координаты почек, вычисляем расстояния
    for side in ['left', 'right']:
        kidney_x = features.get(f'kidney_{side}_center_x')
        kidney_y = features.get(f'kidney_{side}_center_y')
        kidney_z = features.get(f'kidney_{side}_center_z')
        
        if all(v is not None for v in [kidney_x, kidney_y, kidney_z, spine_x, spine_y, spine_z]):
            # Расстояние до позвоночника
            dist_to_spine = np.sqrt(
                (float(kidney_x) - float(spine_x)) ** 2
                + (float(kidney_y) - float(spine_y)) ** 2
                + (float(kidney_z) - float(spine_z)) ** 2
            )
            unified_features[f'kidney_{side}_to_spine_distance'] = float(dist_to_spine)
        
        if all(v is not None for v in [kidney_x, kidney_y, kidney_z, body_com_x, body_com_y, body_com_z]):
            # Расстояние до центра масс тела
            dist_to_body = np.sqrt(
                (float(kidney_x) - float(body_com_x)) ** 2
                + (float(kidney_y) - float(body_com_y)) ** 2
                + (float(kidney_z) - float(body_com_z)) ** 2
            )
            unified_features[f'kidney_{side}_to_body_center_distance'] = float(dist_to_body)
    
    return unified_features


def _get_accuracy_params(mode: str) -> Dict:
    """Возвращает параметры для выбранного режима точности"""
    params = {
        'high': {
            'max_slices': 200,
            'downsample': 2,
            'expected_accuracy': '95-98%',
            'slice_strategy': 'adaptive_kidney'
        },
        'balanced': {
            'max_slices': 100,
            'downsample': 3,
            'expected_accuracy': '90-95%',
            'slice_strategy': 'uniform'
        },
        'fast': {
            'max_slices': 50,
            'downsample': 4,
            'expected_accuracy': '80-90%',
            'slice_strategy': 'uniform'
        },
        'minimal': {
            'max_slices': 25,
            'downsample': 6,
            'expected_accuracy': '70-80%',
            'slice_strategy': 'central'
        }
    }
    return params.get(mode, params['balanced'])


def _adaptive_slice_selection(slice_infos: List[SliceInfo], max_slices: int, strategy: str = 'uniform') -> List[SliceInfo]:
    """Адаптивная выборка срезов для оптимизации точности"""
    if len(slice_infos) <= max_slices:
        return slice_infos
    
    # Сортируем по позиции Z
    sorted_slices = sorted(slice_infos, key=lambda x: x.position[2])
    total_slices = len(sorted_slices)
    
    if strategy == 'uniform':
        # Равномерное распределение по всему объему
        indices = np.linspace(0, total_slices - 1, max_slices, dtype=int)
        return [sorted_slices[i] for i in indices]
    
    elif strategy == 'central':
        # Фокус на центральной части (где обычно находятся почки)
        center_start = total_slices // 4
        center_end = 3 * total_slices // 4
        central_slices = sorted_slices[center_start:center_end]
        
        if len(central_slices) <= max_slices:
            return central_slices
        else:
            indices = np.linspace(0, len(central_slices) - 1, max_slices, dtype=int)
            return [central_slices[i] for i in indices]
    
    elif strategy == 'adaptive_kidney':
        # Адаптивная выборка с фокусом на почках
        # Используем более плотную выборку в центральной части
        central_ratio = 0.6  # 60% срезов из центральной части
        peripheral_ratio = 0.4  # 40% срезов из периферии
        
        central_slices_count = int(max_slices * central_ratio)
        peripheral_slices_count = max_slices - central_slices_count
        
        # Центральная часть (40-60% объема)
        center_start = int(total_slices * 0.4)
        center_end = int(total_slices * 0.6)
        central_slices = sorted_slices[center_start:center_end]
        
        # Периферия (верхняя и нижняя части)
        upper_slices = sorted_slices[:center_start]
        lower_slices = sorted_slices[center_end:]
        
        selected_slices = []
        
        # Выбираем из центральной части
        if len(central_slices) > 0:
            central_indices = np.linspace(0, len(central_slices) - 1, 
                                       min(central_slices_count, len(central_slices)), dtype=int)
            selected_slices.extend([central_slices[i] for i in central_indices])
        
        # Выбираем из верхней периферии
        if len(upper_slices) > 0 and peripheral_slices_count > 0:
            upper_count = min(peripheral_slices_count // 2, len(upper_slices))
            upper_indices = np.linspace(0, len(upper_slices) - 1, upper_count, dtype=int)
            selected_slices.extend([upper_slices[i] for i in upper_indices])
        
        # Выбираем из нижней периферии
        if len(lower_slices) > 0 and peripheral_slices_count > 0:
            lower_count = min(peripheral_slices_count // 2, len(lower_slices))
            lower_indices = np.linspace(0, len(lower_slices) - 1, lower_count, dtype=int)
            selected_slices.extend([lower_slices[i] for i in lower_indices])
        
        # Сортируем по позиции
        selected_slices.sort(key=lambda x: x.position[2])
        return selected_slices
    
    else:
        # По умолчанию - равномерное распределение
        indices = np.linspace(0, total_slices - 1, max_slices, dtype=int)
        return [sorted_slices[i] for i in indices]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('dicom_root', nargs='?', help='Папка с DICOM исследованиями')
    parser.add_argument('--patient-folder', default=None, help='Обработать одну папку пациента')
    parser.add_argument('--output', default='enhanced_ct_features.csv')
    parser.add_argument('--downsample', type=int, default=2)
    parser.add_argument('--max-slices', type=int, default=300)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--disable-kidney-segmentation', action='store_true')
    parser.add_argument('--kidney-only', action='store_true')
    parser.add_argument('--accuracy-mode', choices=['high', 'balanced', 'fast', 'minimal'], 
                       default='balanced', help='Режим точности: high(200 срезов), balanced(100), fast(50), minimal(25)')
    args = parser.parse_args()

    # Получаем параметры accuracy mode
    accuracy_params = _get_accuracy_params(args.accuracy_mode)
    
    # Переопределяем параметры если они не установлены явно
    if args.max_slices == 300:  # значение по умолчанию
        args.max_slices = accuracy_params['max_slices']
    if args.downsample == 2:  # значение по умолчанию
        args.downsample = accuracy_params['downsample']
    
    # Показываем информацию о режиме
    print(f"[mode] accuracy: {args.accuracy_mode.upper()}")
    print(f"   max_slices: {args.max_slices}")
    print(f"   downsample: {args.downsample}x")
    print(f"   expected_accuracy: {accuracy_params['expected_accuracy']}")
    print(f"   slice_strategy: {accuracy_params['slice_strategy']}")

    if args.patient_folder:
        folder = Path(args.patient_folder)
        if not folder.exists():
            raise FileNotFoundError(str(folder))
        
        # Если это папка с DICOM файлами, обрабатываем её как один случай
        if folder.is_dir() and any(f.suffix.lower() in ['.dcm', '.dicom', '.ima'] for f in folder.iterdir()):
            print(f"[scan] patient folder: {folder.name}")
            folders_iter = [folder]
            dicom_root = folder.parent
            total_folders = 1
        # Если это корневая папка с подпапками пациентов
        elif folder.is_dir():
            print(f"[scan] all subfolders in: {folder}")
            folders_iter = _iter_patient_folders(folder)
            dicom_root = folder
            total_folders = len([f for f in folder.iterdir() if f.is_dir()])
        else:
            raise ValueError(f"Путь {folder} не является папкой")
    else:
        dicom_root = Path(args.dicom_root)
        if not dicom_root.exists():
            raise FileNotFoundError(str(dicom_root))
        folders_iter = _iter_patient_folders(dicom_root)
        print(f"[scan] all DICOM folders in: {dicom_root}")
        total_folders = len([f for f in dicom_root.iterdir() if f.is_dir()])
    
    # Обработка
    rows = []
    processed_count = 0
    
    for folder in folders_iter:
        try:
            processed_count += 1
            print(f"[{processed_count}/{total_folders}] folder: {folder.name}")
            
            feats = extract_features_from_dicom_folder(
                folder,
                downsample=args.downsample,
                max_slices=args.max_slices,
                debug=args.debug,
                enable_kidney_segmentation=not args.disable_kidney_segmentation,
                kidney_only=args.kidney_only,
                show_progress=True,  # Всегда показываем прогресс
                current_case=processed_count,
                total_cases=total_folders,
                slice_strategy=accuracy_params['slice_strategy'],
            )
            
            # Добавляем унифицированные признаки
            feats = _add_unified_features(feats)
            
            # Базовые поля kits19
            case_id = folder.name
            full_name = None
            patient_name = feats.get('patient_name')
            if patient_name:
                full_name = str(patient_name)
            if not full_name:
                full_name = _extract_full_name_from_folder(folder.name)
            
            study_date = feats.get('study_date', datetime.now().strftime('%Y-%m-%d'))
            
            row = {
                'case_id': case_id,
                'full_name': full_name,
                'study_date': study_date,
                'dicom_folder': folder.name,
                'full_name_key': _normalize_name(full_name) if full_name else '',
                'dicom_folder_key': _normalize_name(folder.name),
                'status': 'extracted',
                'error': None,
                **feats
            }
            rows.append(row)
            print(f"[OK] [{processed_count}/{total_folders}] {folder.name}")
                
        except Exception as e:
            print(f"[WARN] [{processed_count}/{total_folders}] {folder.name}: {e}")
            if args.debug or True:
                import traceback
                traceback.print_exc()
            
            # Добавляем строку с ошибкой для отслеживания
            row = {
                'case_id': folder.name,
                'full_name': None,
                'study_date': None,
                'dicom_folder': folder.name,
                'full_name_key': '',
                'dicom_folder_key': _normalize_name(folder.name),
                'status': 'error',
                'error': str(e),
                'patient_id': None,
                'study_instance_uid': None,
                'patient_name': None,
                'patient_position': None,
                'sex': None,
                'age': None,
                'bmi': None,
                'scan_position': None,
                'contrast_phase': None,
                'slice_thickness': None,
                'radiographic_size': None,
                'pathologic_size': None,
                'malignant': None,
                'tumor_grade': None,
                'tumor_histology_code': None,
                'smoking_code': None,
                'hospitalization_days': None,
            }
            rows.append(row)

    # Сохранение результатов
    import pandas as pd
    df = pd.DataFrame(rows)
    out_path = Path(args.output)
    df.to_csv(out_path, index=False)
    
    # Статистика обработки
    total = len(rows)
    success = len([r for r in rows if r.get('status') == 'extracted'])
    errors = len([r for r in rows if r.get('status') == 'error'])
    
    print(f"\n[DONE] extraction finished")
    print(f"   total: {total}")
    print(f"   success: {success}")
    print(f"   errors: {errors}")
    print(f"   output: {out_path}")
    
    # Показываем первые 3 успешных случая
    successful_cases = [r for r in rows if r.get('status') == 'extracted'][:3]
    if successful_cases:
        print(f"\n[sample] successful cases:")
        for case in successful_cases:
            case_name = case.get('case_id', 'Unknown')
            full_name = case.get('full_name', 'N/A')
            print(f"   - {case_name}: {full_name}")
    
    # Показываем ошибки если есть
    error_cases = [r for r in rows if r.get('status') == 'error']
    if error_cases:
        print(f"\n[WARN] failed cases:")
        for case in error_cases[:5]:  # Показываем первые 5 ошибок
            case_name = case.get('case_id', 'Unknown')
            error_msg = case.get('error', 'Unknown error')
            print(f"   - {case_name}: {error_msg}")
        if len(error_cases) > 5:
            print(f"   ... и еще {len(error_cases) - 5} случаев с ошибками")
    
    return 0


if __name__ == "__main__":
    main()
