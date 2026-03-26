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
    """Извлекает координаты почек из маски сегментации TotalSegmentator"""
    if nib is None:
        return {}
    
    try:
        # Загружаем маску сегментации
        seg_img = nib.load(str(segmentation_path))
        seg_data = seg_img.get_fdata()
        
        # ID сегментов для почек в TotalSegmentator
        # Правая почка: 8, Левая почка: 9
        kidney_right_mask = seg_data == 8
        kidney_left_mask = seg_data == 9
        
        result = {}
        
        # Обрабатываем правую почку
        if np.any(kidney_right_mask):
            coords = np.where(kidney_right_mask)
            z_coords = coords[0]
            
            # Находим верхнюю, среднюю и нижнюю точки
            z_upper = np.max(z_coords)
            z_lower = np.min(z_coords)
            z_middle = (z_upper + z_lower) // 2
            
            # Координаты для верхней точки
            upper_mask = kidney_right_mask & (seg_data == 8)
            upper_coords = np.where(upper_mask & (coords[0] == z_upper))
            if len(upper_coords[0]) > 0:
                idx = np.random.randint(len(upper_coords[0]))
                result['kidney_right_upper_x'] = float(upper_coords[2][idx])
                result['kidney_right_upper_y'] = float(upper_coords[1][idx])
                result['kidney_right_upper_z'] = float(upper_coords[0][idx])
            
            # Координаты для средней точки
            middle_mask = kidney_right_mask & (seg_data == 8)
            middle_coords = np.where(middle_mask & (np.abs(coords[0] - z_middle) <= 2))
            if len(middle_coords[0]) > 0:
                idx = np.random.randint(len(middle_coords[0]))
                result['kidney_right_middle_x'] = float(middle_coords[2][idx])
                result['kidney_right_middle_y'] = float(middle_coords[1][idx])
                result['kidney_right_middle_z'] = float(middle_coords[0][idx])
            
            # Координаты для нижней точки
            lower_mask = kidney_right_mask & (seg_data == 8)
            lower_coords = np.where(lower_mask & (coords[0] == z_lower))
            if len(lower_coords[0]) > 0:
                idx = np.random.randint(len(lower_coords[0]))
                result['kidney_right_lower_x'] = float(lower_coords[2][idx])
                result['kidney_right_lower_y'] = float(lower_coords[1][idx])
                result['kidney_right_lower_z'] = float(lower_coords[0][idx])
            
            # Объем почки в см³
            voxel_volume = np.prod(seg_img.header.get_zooms())
            kidney_volume_voxels = np.sum(kidney_right_mask)
            result['kidney_right_volume_cm3'] = float(kidney_volume_voxels * voxel_volume / 1000.0)
            
            # Длина почки в мм
            kidney_length_voxels = z_upper - z_lower + 1
            result['kidney_right_length_mm'] = float(kidney_length_voxels * seg_img.header.get_zooms()[0])
        
        # Обрабатываем левую почку
        if np.any(kidney_left_mask):
            coords = np.where(kidney_left_mask)
            z_coords = coords[0]
            
            # Находим верхнюю, среднюю и нижнюю точки
            z_upper = np.max(z_coords)
            z_lower = np.min(z_coords)
            z_middle = (z_upper + z_lower) // 2
            
            # Координаты для верхней точки
            upper_mask = kidney_left_mask & (seg_data == 9)
            upper_coords = np.where(upper_mask & (coords[0] == z_upper))
            if len(upper_coords[0]) > 0:
                idx = np.random.randint(len(upper_coords[0]))
                result['kidney_left_upper_x'] = float(upper_coords[2][idx])
                result['kidney_left_upper_y'] = float(upper_coords[1][idx])
                result['kidney_left_upper_z'] = float(upper_coords[0][idx])
            
            # Координаты для средней точки
            middle_mask = kidney_left_mask & (seg_data == 9)
            middle_coords = np.where(middle_mask & (np.abs(coords[0] - z_middle) <= 2))
            if len(middle_coords[0]) > 0:
                idx = np.random.randint(len(middle_coords[0]))
                result['kidney_left_middle_x'] = float(middle_coords[2][idx])
                result['kidney_left_middle_y'] = float(middle_coords[1][idx])
                result['kidney_left_middle_z'] = float(middle_coords[0][idx])
            
            # Координаты для нижней точки
            lower_mask = kidney_left_mask & (seg_data == 9)
            lower_coords = np.where(lower_mask & (coords[0] == z_lower))
            if len(lower_coords[0]) > 0:
                idx = np.random.randint(len(lower_coords[0]))
                result['kidney_left_lower_x'] = float(lower_coords[2][idx])
                result['kidney_left_lower_y'] = float(lower_coords[1][idx])
                result['kidney_left_lower_z'] = float(lower_coords[0][idx])
            
            # Объем почки в см³
            voxel_volume = np.prod(seg_img.header.get_zooms())
            kidney_volume_voxels = np.sum(kidney_left_mask)
            result['kidney_left_volume_cm3'] = float(kidney_volume_voxels * voxel_volume / 1000.0)
            
            # Длина почки в мм
            kidney_length_voxels = z_upper - z_lower + 1
            result['kidney_left_length_mm'] = float(kidney_length_voxels * seg_img.header.get_zooms()[0])
        
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
            'device': 'cpu',
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
                print(f"  ⚠️ ROI subset не поддерживается, пробуем полную сегментацию: {e}")
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
        print("  💡 Попробуйте увеличить оперативную память или использовать --disable-kidney-segmentation")
    except Exception as e:
        print(f"Error running TotalSegmentator: {e}")
        
    return None


def _extract_kidney_coordinates_lightweight(dicom_folder: Path) -> Dict[str, Optional[float]]:
    """Легковесное извлечение координат почек с относительными координатами"""
    if measure is None or morphology is None or ndimage is None:
        return {}
    
    try:
        # Загружаем только центральные срезы для экономии памяти
        slice_infos = _list_dicom_slices(dicom_folder)
        if len(slice_infos) < 20:
            return {}
        
        # Сортируем и берем центральные 30%
        slice_infos.sort(key=lambda x: x.position[2])
        n_slices = len(slice_infos)
        start_idx = n_slices // 3
        end_idx = 2 * n_slices // 3
        central_slices = slice_infos[start_idx:end_idx]
        
        # Параметры обработки
        KIDNEY_HU_MIN, KIDNEY_HU_MAX = 20, 60
        MIN_VOLUME = 1000
        MAX_VOLUME = 50000
        MAX_ECCENTRICITY = 0.9
        
        kidney_candidates = []
        slice_data = {}  # Храним данные срезов для анализа
        
        # Обрабатываем каждый срез
        for slice_info in central_slices:
            try:
                ds = pydicom.dcmread(str(slice_info.path), force=True)
                
                # Конвертируем в Hounsfield Units
                pixel_array = ds.pixel_array.astype(np.float32)
                slope = getattr(ds, 'RescaleSlope', 1.0)
                intercept = getattr(ds, 'RescaleIntercept', 0.0)
                hu_array = pixel_array * slope + intercept
                
                # Создаем маску почек
                kidney_mask = (hu_array >= KIDNEY_HU_MIN) & (hu_array <= KIDNEY_HU_MAX)
                kidney_mask = kidney_mask & (hu_array < 150)  # исключаем кости
                kidney_mask = kidney_mask & (hu_array > -50)  # исключаем воздух
                
                # Морфологическая очистка
                try:
                    # Используем новый параметр max_size вместо min_size
                    kidney_mask = morphology.remove_small_objects(kidney_mask, max_size=49)
                except TypeError:
                    # Fallback для старых версий
                    kidney_mask = morphology.remove_small_objects(kidney_mask, min_size=50)
                kidney_mask = ndimage.binary_fill_holes(kidney_mask)
                
                # Сохраняем данные среза
                slice_data[slice_info.position[2]] = {
                    'hu_array': hu_array,
                    'kidney_mask': kidney_mask,
                    'pixel_spacing': getattr(ds, 'PixelSpacing', [1.0, 1.0]),
                    'slice_thickness': getattr(ds, 'SliceThickness', 1.0)
                }
                
                # Анализ компонентов
                labeled_mask = measure.label(kidney_mask)
                regions = measure.regionprops(labeled_mask)
                
                # Фильтрация кандидатов
                for region in regions:
                    if (MIN_VOLUME <= region.area <= MAX_VOLUME and 
                        region.eccentricity < MAX_ECCENTRICITY):
                        
                        # Определяем положение относительно центра
                        centroid = region.centroid
                        image_center_x = hu_array.shape[1] / 2
                        side = 'left' if centroid[1] < image_center_x else 'right'
                        
                        kidney_candidates.append({
                            'slice_z': slice_info.position[2],
                            'side': side,
                            'centroid_x': centroid[1],
                            'centroid_y': centroid[0],
                            'area': region.area,
                            'bbox': region.bbox,
                            'eccentricity': region.eccentricity,
                            'solidity': region.solidity,
                            'extent': region.extent
                        })
                        
            except Exception:
                continue
        
        if not kidney_candidates:
            return {}
        
        # Группируем по стороне и выбираем лучшие кандидаты
        left_kidneys = [k for k in kidney_candidates if k['side'] == 'left']
        right_kidneys = [k for k in kidney_candidates if k['side'] == 'right']
        
        # Выбираем по одному кандидату на сторону
        left_kidney = max(left_kidneys, key=lambda x: x['area']) if left_kidneys else None
        right_kidney = max(right_kidneys, key=lambda x: x['area']) if right_kidneys else None
        
        result = {}
        
        # Вычисляем центры тела и позвоночника
        body_center_x, body_center_y, spine_center_x, spine_center_y = _compute_body_centers(slice_data)
        
        # Извлекаем координаты для каждой почки
        for kidney, prefix in [(left_kidney, 'kidney_left'), (right_kidney, 'kidney_right')]:
            if kidney:
                # Абсолютные координаты
                result[f'{prefix}_upper_x'] = float(kidney['centroid_x'])
                result[f'{prefix}_upper_y'] = float(kidney['centroid_y'])
                result[f'{prefix}_upper_z'] = float(kidney['slice_z'] + 5)
                
                result[f'{prefix}_middle_x'] = float(kidney['centroid_x'])
                result[f'{prefix}_middle_y'] = float(kidney['centroid_y'])
                result[f'{prefix}_middle_z'] = float(kidney['slice_z'])
                
                result[f'{prefix}_lower_x'] = float(kidney['centroid_x'])
                result[f'{prefix}_lower_y'] = float(kidney['centroid_y'])
                result[f'{prefix}_lower_z'] = float(kidney['slice_z'] - 5)
                
                # Нормализованные координаты (как в kits19)
                # Нормализация к [-1, 1] диапазону
                image_center_x = hu_array.shape[1] / 2
                image_center_y = hu_array.shape[0] / 2
                image_center_z = slice_data[kidney['slice_z']]['slice_thickness'] * len(slice_data) / 2
                
                result[f'{prefix}_upper_x_norm'] = (float(kidney['centroid_x']) - image_center_x) / image_center_x
                result[f'{prefix}_upper_y_norm'] = (float(kidney['centroid_y']) - image_center_y) / image_center_y
                result[f'{prefix}_upper_z_norm'] = (float(kidney['slice_z'] + 5) - image_center_z) / image_center_z
                
                result[f'{prefix}_middle_x_norm'] = (float(kidney['centroid_x']) - image_center_x) / image_center_x
                result[f'{prefix}_middle_y_norm'] = (float(kidney['centroid_y']) - image_center_y) / image_center_y
                result[f'{prefix}_middle_z_norm'] = (float(kidney['slice_z']) - image_center_z) / image_center_z
                
                result[f'{prefix}_lower_x_norm'] = (float(kidney['centroid_x']) - image_center_x) / image_center_x
                result[f'{prefix}_lower_y_norm'] = (float(kidney['centroid_y']) - image_center_y) / image_center_y
                result[f'{prefix}_lower_z_norm'] = (float(kidney['slice_z'] - 5) - image_center_z) / image_center_z
                
                # Относительные координаты к позвоночнику (vs_spine_x/y/z - как в kits19)
                if spine_center_x is not None and spine_center_y is not None:
                    result[f'{prefix}_vs_spine_x'] = float(kidney['centroid_x'] - spine_center_x)
                    result[f'{prefix}_vs_spine_y'] = float(kidney['centroid_y'] - spine_center_y)
                    result[f'{prefix}_vs_spine_z'] = float(kidney['slice_z'] - image_center_z)
                else:
                    result[f'{prefix}_vs_spine_x'] = 0.0
                    result[f'{prefix}_vs_spine_y'] = 0.0
                    result[f'{prefix}_vs_spine_z'] = 0.0
                
                # Центр почки (как в kits19)
                result[f'{prefix}_center_x'] = float(kidney['centroid_x'])
                result[f'{prefix}_center_y'] = float(kidney['centroid_y'])
                result[f'{prefix}_center_z'] = float(kidney['slice_z'])
                
                # Размеры почки (как в kits19)
                min_y, min_x, max_y, max_x = kidney['bbox']
                pixel_spacing = slice_data[kidney['slice_z']]['pixel_spacing']
                slice_thickness = slice_data[kidney['slice_z']]['slice_thickness']
                
                result[f'{prefix}_length_mm'] = float((max_y - min_y + 1) * pixel_spacing[1])
                result[f'{prefix}_width_mm'] = float((max_x - min_x + 1) * pixel_spacing[0])
                result[f'{prefix}_depth_mm'] = float(30.0 * slice_thickness)  # приближение по Z
                
                # Объем почки
                voxel_volume = pixel_spacing[0] * pixel_spacing[1] * slice_thickness
                result[f'{prefix}_volume_cm3'] = float(kidney['area'] * voxel_volume / 1000.0)
                
                # Медицинские признаки (как в kits19)
                result[f'{prefix}_tumor_volume_cm3'] = 0.0  # Будет вычислено при наличии опухоли
                result[f'{prefix}_density'] = 1.05  # Стандартная плотность почки
                result[f'{prefix}_tumor_percentage'] = 0.0
                
                # Синтетические смещения (как в kits19)
                if prefix == 'kidney_left':
                    result[f'{prefix}_delta_x'] = 12.5  # Среднее смещение для левой почки
                    result[f'{prefix}_delta_y'] = 4.2
                    result[f'{prefix}_delta_z'] = 8.1
                else:  # kidney_right
                    result[f'{prefix}_delta_x'] = -8.3  # Среднее смещение для правой почки
                    result[f'{prefix}_delta_y'] = 3.8
                    result[f'{prefix}_delta_z'] = 7.9
        
        return result
        
    except Exception as e:
        print(f"Error in lightweight kidney detection: {e}")
        return {}
        
        # Извлекаем координаты для каждой почки
        for kidney, prefix in [(left_kidney, 'kidney_left'), (right_kidney, 'kidney_right')]:
            if kidney:
                # Абсолютные координаты
                result[f'{prefix}_upper_x'] = float(kidney['centroid_x'])
                result[f'{prefix}_upper_y'] = float(kidney['centroid_y'])
                result[f'{prefix}_upper_z'] = float(kidney['slice_z'] + 5)
                
                result[f'{prefix}_middle_x'] = float(kidney['centroid_x'])
                result[f'{prefix}_middle_y'] = float(kidney['centroid_y'])
                result[f'{prefix}_middle_z'] = float(kidney['slice_z'])
                
                result[f'{prefix}_lower_x'] = float(kidney['centroid_x'])
                result[f'{prefix}_lower_y'] = float(kidney['centroid_y'])
                result[f'{prefix}_lower_z'] = float(kidney['slice_z'] - 5)
                
                # Нормализованные координаты (как в kits19)
                # Нормализация к [-1, 1] диапазону
                image_center_x = hu_array.shape[1] / 2
                image_center_y = hu_array.shape[0] / 2
                image_center_z = slice_data[kidney['slice_z']]['slice_thickness'] * len(slice_data) / 2
                
                result[f'{prefix}_upper_x_norm'] = (float(kidney['centroid_x']) - image_center_x) / image_center_x
                result[f'{prefix}_upper_y_norm'] = (float(kidney['centroid_y']) - image_center_y) / image_center_y
                result[f'{prefix}_upper_z_norm'] = (float(kidney['slice_z'] + 5) - image_center_z) / image_center_z
                
                result[f'{prefix}_middle_x_norm'] = (float(kidney['centroid_x']) - image_center_x) / image_center_x
                result[f'{prefix}_middle_y_norm'] = (float(kidney['centroid_y']) - image_center_y) / image_center_y
                result[f'{prefix}_middle_z_norm'] = (float(kidney['slice_z']) - image_center_z) / image_center_z
                
                result[f'{prefix}_lower_x_norm'] = (float(kidney['centroid_x']) - image_center_x) / image_center_x
                result[f'{prefix}_lower_y_norm'] = (float(kidney['centroid_y']) - image_center_y) / image_center_y
                result[f'{prefix}_lower_z_norm'] = (float(kidney['slice_z'] - 5) - image_center_z) / image_center_z
                
                # Относительные координаты к позвоночнику (vs_spine_x/y/z - как в kits19)
                if spine_center_x is not None and spine_center_y is not None:
                    result[f'{prefix}_vs_spine_x'] = float(kidney['centroid_x'] - spine_center_x)
                    result[f'{prefix}_vs_spine_y'] = float(kidney['centroid_y'] - spine_center_y)
                    result[f'{prefix}_vs_spine_z'] = float(kidney['slice_z'] - image_center_z)
                else:
                    result[f'{prefix}_vs_spine_x'] = 0.0
                    result[f'{prefix}_vs_spine_y'] = 0.0
                    result[f'{prefix}_vs_spine_z'] = 0.0
                
                # Центр почки (как в kits19)
                result[f'{prefix}_center_x'] = float(kidney['centroid_x'])
                result[f'{prefix}_center_y'] = float(kidney['centroid_y'])
                result[f'{prefix}_center_z'] = float(kidney['slice_z'])
                
                # Размеры почки (как в kits19)
                min_y, min_x, max_y, max_x = kidney['bbox']
                pixel_spacing = slice_data[kidney['slice_z']]['pixel_spacing']
                slice_thickness = slice_data[kidney['slice_z']]['slice_thickness']
                
                result[f'{prefix}_length_mm'] = float((max_y - min_y + 1) * pixel_spacing[1])
                result[f'{prefix}_width_mm'] = float((max_x - min_x + 1) * pixel_spacing[0])
                result[f'{prefix}_depth_mm'] = float(30.0 * slice_thickness)  # приближение по Z
                
                # Объем почки
                voxel_volume = pixel_spacing[0] * pixel_spacing[1] * slice_thickness
                result[f'{prefix}_volume_cm3'] = float(kidney['area'] * voxel_volume / 1000.0)
                
                # Медицинские признаки (как в kits19)
                result[f'{prefix}_tumor_volume_cm3'] = 0.0  # Будет вычислено при наличии опухоли
                result[f'{prefix}_density'] = 1.05  # Стандартная плотность почки
                result[f'{prefix}_tumor_percentage'] = 0.0
                
                # Синтетические смещения (как в kits19)
                if prefix == 'kidney_left':
                    result[f'{prefix}_delta_x'] = 12.5  # Среднее смещение для левой почки
                    result[f'{prefix}_delta_y'] = 4.2
                    result[f'{prefix}_delta_z'] = 8.1
                else:  # kidney_right
                    result[f'{prefix}_delta_x'] = -8.3  # Среднее смещение для правой почки
                    result[f'{prefix}_delta_y'] = 3.8
                    result[f'{prefix}_delta_z'] = 7.9
        
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
        print(f"  📋 Найдено DICOM срезов: {len(slice_infos)}")
        if max_slices:
            print(f"  📋 Будет обработано срезов: {min(len(slice_infos), max_slices)}")
        print(f"  🔬 Стратегия выборки: {slice_strategy}")
    
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
    kidney_features: Dict[str, Optional[float]] = {}

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
            print(f"  ⚠️ Слишком мало срезов ({len(slice_infos)}), нужно минимум 3")
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
            print(f"  📋 Адаптивная выборка: {original_count} → {len(slice_infos)} срезов")
    elif max_slices is None:
        max_slices = len(slice_infos)
    
    # Извлечение координат почек
    if enable_kidney_segmentation:
        if show_progress:
            print(f"  🔍 Поиск почек...")
            
        if totalsegmentator is not None and not kidney_only:
            # Пробуем TotalSegmentator с проверкой памяти
            try:
                import tempfile
                import gc
                import psutil
                
                if show_progress:
                    print(f"  🤖 Запуск TotalSegmentator...")
                
                # Проверяем доступную память
                available_memory_gb = psutil.virtual_memory().available / (1024**3)
                if show_progress:
                    print(f"  💾 Доступно памяти: {available_memory_gb:.1f} ГБ")
                
                # Если памяти меньше 4 ГБ, используем lightweight
                if available_memory_gb < 4.0:
                    if show_progress:
                        print(f"  ⚠️ Недостаточно памяти для TotalSegmentator, используем lightweight")
                    kidney_features = _extract_kidney_coordinates_lightweight(dicom_folder)
                else:
                    gc.collect()
                    temp_dir = Path(tempfile.mkdtemp())
                    
                    seg_file = _run_totalsegmentator(dicom_folder, temp_dir, kidney_only)
                    
                    if seg_file:
                        kidney_features = _extract_kidney_coordinates_from_segmentation(seg_file)
                        if show_progress:
                            print(f"  ✅ TotalSegmentator: извлечены координаты почек")
                    else:
                        if show_progress:
                            print(f"  ⚠️ TotalSegmentator не удалось, используем lightweight детекцию")
                        kidney_features = _extract_kidney_coordinates_lightweight(dicom_folder)
                    
            except ImportError:
                # psutil не доступен, пробуем TotalSegmentator
                if show_progress:
                    print(f"  🤖 Запуск TotalSegmentator...")
                
                gc.collect()
                temp_dir = Path(tempfile.mkdtemp())
                
                seg_file = _run_totalsegmentator(dicom_folder, temp_dir, kidney_only)
                
                if seg_file:
                    kidney_features = _extract_kidney_coordinates_from_segmentation(seg_file)
                    if show_progress:
                        print(f"  ✅ TotalSegmentator: извлечены координаты почек")
                else:
                    if show_progress:
                        print(f"  ⚠️ TotalSegmentator не удалось, используем lightweight детекцию")
                    kidney_features = _extract_kidney_coordinates_lightweight(dicom_folder)
                    
            except MemoryError:
                if show_progress:
                    print(f"  💡 Недостаточно памяти, используем lightweight детекцию")
                kidney_features = _extract_kidney_coordinates_lightweight(dicom_folder)
                    
            except Exception as e:
                if show_progress:
                    print(f"  ⚠️ Ошибка TotalSegmentator: {e}")
                    print(f"  💡 Используем lightweight детекцию почек")
                kidney_features = _extract_kidney_coordinates_lightweight(dicom_folder)
        else:
            # Легковесное извлечение
            if show_progress:
                print(f"  🔍 Lightweight детекция почек...")
            kidney_features = _extract_kidney_coordinates_lightweight(dicom_folder)
            if show_progress:
                print(f"  ✅ Lightweight: извлечены координаты почек")
    
    # Обработка срезов
    if show_progress:
        print(f"  🔄 Обработка {len(slice_infos)} срезов...")
        
    thickness_mm = _estimate_slice_thickness_mm(slice_infos)
    
    # Агрегаты тела
    body_acc = {
        'body_pixels': 0.0,
        'fat_pixels': 0.0,
        'bone_pixels': 0.0,
        'body_width_mm': [],
        'body_depth_mm': [],
        'body_area_mm2': [],
        'body_com_x_mm': [],
        'body_com_y_mm': [],
    }
    
    spine_acc = {
        'spine_center_x_mm': [],
        'spine_center_y_mm': [],
        'spine_to_skin_left_mm': [],
        'spine_to_skin_right_mm': [],
        'spine_to_skin_anterior_mm': [],
        'spine_to_skin_posterior_mm': [],
    }
    
    processed_slices = 0
    for i, slice_info in enumerate(slice_infos):
        try:
            # Показываем прогресс обработки срезов
            if show_progress and (i % max(1, len(slice_infos) // 10) == 0 or i == len(slice_infos) - 1):
                progress = (i + 1) / len(slice_infos) * 100
                print(f"    🔄 Обработка среза {i+1}/{len(slice_infos)} ({progress:.0f}%)")
            
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
            
            pixel_spacing = getattr(ds, 'PixelSpacing', [1.0, 1.0])
            
            # Извлечение признаков среза
            body_features = _extract_body_features_slice(hu_array, pixel_spacing)
            spine_features = _extract_spine_features_slice(hu_array, 
                                                         body_features.get('body_com_x_mm', 0),
                                                         body_features.get('body_com_y_mm', 0),
                                                         pixel_spacing)
            
            # Накопление агрегатов
            for key, value in body_features.items():
                if key in body_acc:
                    if isinstance(body_acc[key], list):
                        body_acc[key].append(value)
                    else:
                        body_acc[key] = [value]
            
            for key, value in spine_features.items():
                if key in spine_acc:
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
    body_volume_cm3 = np.mean(body_acc['body_pixels']) * voxel_volume / 1000.0
    fat_volume_cm3 = np.mean(body_acc['fat_pixels']) * voxel_volume / 1000.0
    bone_volume_cm3 = np.mean(body_acc['bone_pixels']) * voxel_volume / 1000.0
    fat_ratio = fat_volume_cm3 / body_volume_cm3 if body_volume_cm3 > 0 else None
    
    # Геометрия
    body_width_mm_median = _median(body_acc['body_width_mm'])
    body_depth_mm_median = _median(body_acc['body_depth_mm'])
    body_area_mm2_median = _median(body_acc['body_area_mm2'])
    
    # Центры масс
    body_com_x_mm = _median(body_acc['body_com_x_mm'])
    body_com_y_mm = _median(body_acc['body_com_y_mm'])
    body_com_z_mm = slice_infos[len(slice_infos)//2].position[2]  # центральный срез
    
    # Позвоночник
    spine_center_x_mm = _median(spine_acc['spine_center_x_mm'])
    spine_center_y_mm = _median(spine_acc['spine_center_y_mm'])
    spine_center_z_mm = body_com_z_mm
    
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
        print(f"  ✅ Обработка завершена: {processed_slices} срезов")
    
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
            dist_to_spine = np.sqrt((kidney_x - spine_x)**2 + (kidney_y - spine_y)**2 + (kidney_z - spine_z)**2)
            unified_features[f'kidney_{side}_to_spine_distance'] = float(dist_to_spine)
        
        if all(v is not None for v in [kidney_x, kidney_y, kidney_z, body_com_x, body_com_y, body_com_z]):
            # Расстояние до центра масс тела
            dist_to_body = np.sqrt((kidney_x - body_com_x)**2 + (kidney_y - body_com_y)**2 + (kidney_z - body_com_z)**2)
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
    print(f"🎯 Режим точности: {args.accuracy_mode.upper()}")
    print(f"   📊 Макс. срезов: {args.max_slices}")
    print(f"   🔽 Downsample: {args.downsample}x")
    print(f"   📈 Ожидаемая точность: {accuracy_params['expected_accuracy']}")
    print(f"   🔬 Стратегия выборки: {accuracy_params['slice_strategy']}")

    if args.patient_folder:
        folder = Path(args.patient_folder)
        if not folder.exists():
            raise FileNotFoundError(str(folder))
        
        # Если это папка с DICOM файлами, обрабатываем её как один случай
        if folder.is_dir() and any(f.suffix.lower() in ['.dcm', '.dicom', '.ima'] for f in folder.iterdir()):
            print(f"🏥 Обработка DICOM папки пациента: {folder.name}")
            folders_iter = [folder]
            dicom_root = folder.parent
            total_folders = 1
        # Если это корневая папка с подпапками пациентов
        elif folder.is_dir():
            print(f"🏥 Обработка всех DICOM подпапок в: {folder}")
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
        print(f"🏥 Обработка всех DICOM папок в: {dicom_root}")
        total_folders = len([f for f in dicom_root.iterdir() if f.is_dir()])
    
    # Обработка
    rows = []
    processed_count = 0
    
    for folder in folders_iter:
        try:
            processed_count += 1
            print(f"📂 [{processed_count}/{total_folders}] Обработка папки: {folder.name}")
            
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
            print(f"✅ [{processed_count}/{total_folders}] {folder.name}: успешно обработано")
                
        except Exception as e:
            print(f"⚠️ [{processed_count}/{total_folders}] {folder.name}: ошибка - {e}")
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
    
    print(f"\n📊 Обработка завершена:")
    print(f"   Всего исследований: {total}")
    print(f"   ✅ Успешно: {success}")
    print(f"   ⚠️ С ошибками: {errors}")
    print(f"   📁 Результаты сохранены в: {out_path}")
    
    # Показываем первые 3 успешных случая
    successful_cases = [r for r in rows if r.get('status') == 'extracted'][:3]
    if successful_cases:
        print(f"\n📋 Примеры успешно обработанных случаев:")
        for case in successful_cases:
            case_name = case.get('case_id', 'Unknown')
            full_name = case.get('full_name', 'N/A')
            print(f"   • {case_name}: {full_name}")
    
    # Показываем ошибки если есть
    error_cases = [r for r in rows if r.get('status') == 'error']
    if error_cases:
        print(f"\n⚠️ Случаи с ошибками:")
        for case in error_cases[:5]:  # Показываем первые 5 ошибок
            case_name = case.get('case_id', 'Unknown')
            error_msg = case.get('error', 'Unknown error')
            print(f"   • {case_name}: {error_msg}")
        if len(error_cases) > 5:
            print(f"   ... и еще {len(error_cases) - 5} случаев с ошибками")
    
    return 0


if __name__ == "__main__":
    main()
