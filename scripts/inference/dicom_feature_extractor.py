#!/usr/bin/env python3

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

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


@dataclass
class SliceInfo:
    path: Path
    instance_number: Optional[int]
    z: Optional[float]
    number_of_frames: Optional[int] = None


def _is_dicom_file(path: Path) -> bool:
    try:
        with open(path, 'rb') as f:
            f.seek(128)
            return f.read(4) == b'DICM'
    except Exception:
        return False


def _normalize_name(value: str) -> str:
    if value is None:
        return ''
    s = str(value).strip().lower()
    s = s.replace('ё', 'е')
    s = re.sub(r'\s+', '', s)
    s = re.sub(r'[^a-zа-я0-9]+', '', s)
    return s


def _extract_full_name_from_folder(folder_name: str) -> Optional[str]:
    words = re.findall(r'[А-Я][а-я]+', folder_name)
    if len(words) >= 2:
        return ' '.join(words[:3])
    return None


def _safe_str(value: object) -> str:
    if value is None:
        return ''
    try:
        return str(value).strip()
    except Exception:
        return ''


def _parse_patient_age_years(value: object) -> Optional[float]:
    """DICOM PatientAge is often like '045Y' or '032M'."""
    s = _safe_str(value).upper()
    if not s:
        return None
    m = re.match(r'^(\d{1,3})([YMWD])$', s)
    if not m:
        try:
            return float(s)
        except Exception:
            return None

    n = float(m.group(1))
    unit = m.group(2)
    if unit == 'Y':
        return n
    if unit == 'M':
        return n / 12.0
    if unit == 'W':
        return n / 52.0
    if unit == 'D':
        return n / 365.0
    return None


def _extract_demographics(ds) -> Dict[str, Optional[float]]:
    sex_raw = _safe_str(getattr(ds, 'PatientSex', None)).upper()
    sex = None
    if sex_raw in {'M', 'MALE'}:
        sex = 1.0
    elif sex_raw in {'F', 'FEMALE'}:
        sex = 0.0

    age = _parse_patient_age_years(getattr(ds, 'PatientAge', None))

    weight_kg = None
    height_m = None
    try:
        w = getattr(ds, 'PatientWeight', None)
        weight_kg = float(w) if w is not None else None
    except Exception:
        weight_kg = None

    try:
        h = getattr(ds, 'PatientSize', None)
        height_m = float(h) if h is not None else None
    except Exception:
        height_m = None

    bmi = None
    if weight_kg is not None and height_m is not None and height_m > 0:
        bmi = float(weight_kg) / float(height_m * height_m)

    # Вычисляем телосложение на основе ИМТ
    body_type = None
    if bmi is not None:
        if bmi < 18.5:
            body_type = 0  # худощавое
        elif bmi < 25.0:
            body_type = 1  # нормальное
        else:
            body_type = 2  # избыточное

    return {
        'sex': sex,
        'age': float(age) if age is not None else None,
        'weight_kg': float(weight_kg) if weight_kg is not None else None,
        'height_m': float(height_m) if height_m is not None else None,
        'bmi': float(bmi) if bmi is not None else None,
        'body_type': float(body_type) if body_type is not None else None,
    }


def _list_dicom_slices(dicom_folder: Path) -> List[SliceInfo]:
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

            slices.append(SliceInfo(path=fp, instance_number=inst_num, z=z, number_of_frames=n_frames_int))
        except Exception:
            continue

    if not slices:
        return []

    if any(s.z is not None for s in slices):
        slices.sort(key=lambda s: (float('inf') if s.z is None else s.z, s.instance_number or 0))
    else:
        slices.sort(key=lambda s: (s.instance_number is None, s.instance_number or 0, str(s.path)))

    return slices


def _estimate_slice_thickness_mm(slice_infos: List[SliceInfo]) -> Optional[float]:
    zs = [s.z for s in slice_infos if s.z is not None]
    if len(zs) >= 2:
        diffs = [abs(b - a) for a, b in zip(zs[:-1], zs[1:]) if a is not None and b is not None]
        diffs = [d for d in diffs if d > 1e-3]
        if diffs:
            return float(np.median(diffs))
    return None


def _extract_patient_position(ds) -> Optional[str]:
    """Извлекает позицию пациента из DICOM метаданных"""
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
    """Извлекает дату исследования из DICOM метаданных"""
    try:
        date = getattr(ds, 'StudyDate', None)
        if date:
            date_str = str(date).strip()
            # Формат DICOM: YYYYMMDD
            if len(date_str) == 8 and date_str.isdigit():
                return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            return date_str
    except Exception:
        pass
    return None


def _extract_kidney_coordinates_simple(dicom_folder: Path) -> Dict[str, Optional[float]]:
    """Простой детектор почек на основе Hounsfield Units (альтернатива TotalSegmentator)"""
    try:
        import scipy.ndimage as ndimage
        from skimage import measure, morphology
    except ImportError:
        print("  ⚠️ scipy/scikit-image не установлены - простой детектор недоступен")
        return {}
    
    try:
        # Загружаем DICOM файлы
        dicom_files = []
        for file_path in dicom_folder.glob('**/*.dcm'):
            if file_path.is_file():
                try:
                    ds = pydicom.dcmread(str(file_path))
                    if hasattr(ds, 'pixel_array'):
                        dicom_files.append(ds)
                except:
                    continue
        
        if len(dicom_files) < 10:  # Нужно достаточно слоев для 3D анализа
            return {}
        
        # Сортируем по позиции Z
        dicom_files.sort(key=lambda x: getattr(x, 'ImagePositionPatient', [0, 0, 0])[2] 
                        if hasattr(x, 'ImagePositionPatient') else 0)
        
        # Создаем 3D объем
        first_ds = dicom_files[0]
        slope = getattr(first_ds, 'RescaleSlope', 1.0)
        intercept = getattr(first_ds, 'RescaleIntercept', 0.0)
        
        # Берем только центральные слои для экономии памяти
        n_files = len(dicom_files)
        start_idx = max(0, n_files // 4)
        end_idx = min(n_files, 3 * n_files // 4)
        selected_files = dicom_files[start_idx:end_idx]
        
        volume_3d = np.zeros((len(selected_files),) + first_ds.pixel_array.shape, dtype=np.float32)
        
        for i, ds in enumerate(selected_files):
            volume_3d[i] = ds.pixel_array.astype(np.float32) * slope + intercept
        
        # Диапазон Hounsfield Units для почек (20-60 HU)
        kidney_mask = (volume_3d >= 20) & (volume_3d <= 60)
        
        # Дополнительная фильтрация
        kidney_mask = kidney_mask & (volume_3d < 150)  # Исключаем кости/контраст
        kidney_mask = kidney_mask & (volume_3d > -50)  # Исключаем воздух/жир
        
        # Морфологическая очистка
        kidney_mask = morphology.remove_small_objects(kidney_mask, min_size=100)
        kidney_mask = ndimage.binary_fill_holes(kidney_mask)
        
        # Анализ связанных компонентов
        labeled_mask = measure.label(kidney_mask)
        regions = measure.regionprops(labeled_mask)
        
        # Фильтрация по размеру и форме
        kidney_candidates = []
        for region in regions:
            volume_voxels = region.area
            if 1000 <= volume_voxels <= 50000:  # Фильтр по объему
                if region.eccentricity < 0.9:  # Фильтр по форме
                    kidney_candidates.append(region)
        
        # Выбираем два самых больших кандидата (левая и правая почки)
        kidney_candidates.sort(key=lambda x: x.area, reverse=True)
        kidney_candidates = kidney_candidates[:2]
        
        result = {}
        
        for i, region in enumerate(kidney_candidates):
            # Определяем сторону по центроиду
            centroid = region.centroid
            z, y, x = centroid
            image_center_x = volume_3d.shape[2] / 2
            side = 'left' if x < image_center_x else 'right'
            prefix = f'kidney_{side}'
            
            # Получаем границы
            min_z, min_y, min_x, max_z, max_y, max_x = region.bbox
            
            # Координаты верхней, средней и нижней точек
            z_upper = max_z
            z_lower = min_z  
            z_middle = (z_upper + z_lower) // 2
            
            center_y = (min_y + max_y) // 2
            center_x = (min_x + max_x) // 2
            
            # Сохраняем координаты
            result[f'{prefix}_upper_x'] = float(center_x)
            result[f'{prefix}_upper_y'] = float(center_y)
            result[f'{prefix}_upper_z'] = float(z_upper + start_idx)
            
            result[f'{prefix}_middle_x'] = float(center_x)
            result[f'{prefix}_middle_y'] = float(center_y)
            result[f'{prefix}_middle_z'] = float(z_middle + start_idx)
            
            result[f'{prefix}_lower_x'] = float(center_x)
            result[f'{prefix}_lower_y'] = float(center_y)
            result[f'{prefix}_lower_z'] = float(z_lower + start_idx)
            
            # Объем и длина (приблизительные)
            pixel_spacing = getattr(first_ds, 'PixelSpacing', [1.0, 1.0])
            slice_thickness = getattr(first_ds, 'SliceThickness', 1.0)
            
            volume_mm3 = region.area * pixel_spacing[0] * pixel_spacing[1] * slice_thickness
            length_mm = (max_z - min_z) * slice_thickness
            
            result[f'{prefix}_volume_cm3'] = float(volume_mm3 / 1000.0)
            result[f'{prefix}_length_mm'] = float(length_mm)
        
        print(f"  📍 Simple kidney detector: найдено {len(kidney_candidates)} почек")
        return result
        
    except Exception as e:
        print(f"  ⚠️ Error in simple kidney detection: {e}")
        return {}


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
                idx = np.random.randint(len(upper_coords[0]))  # случайная точка из верхних
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


def _read_slice_hu(path: Path) -> Tuple[np.ndarray, float, float, Optional[float]]:
    ds = pydicom.dcmread(str(path), force=True)

    def _get_pixel_spacing_mm(_ds) -> Tuple[float, float]:
        candidates = []

        ps = getattr(_ds, 'PixelSpacing', None)
        if ps is not None:
            candidates.append(ps)

        ips = getattr(_ds, 'ImagerPixelSpacing', None)
        if ips is not None:
            candidates.append(ips)

        nsps = getattr(_ds, 'NominalScannedPixelSpacing', None)
        if nsps is not None:
            candidates.append(nsps)

        def _try_add_from_fgs_item(fg_item) -> None:
            try:
                pm_seq = getattr(fg_item, 'PixelMeasuresSequence', None)
                if pm_seq is None:
                    return
                if len(pm_seq) <= 0:
                    return
                pm0 = pm_seq[0]
                candidates.append(getattr(pm0, 'PixelSpacing', None))
                candidates.append(getattr(pm0, 'ImagerPixelSpacing', None))
                candidates.append(getattr(pm0, 'NominalScannedPixelSpacing', None))
            except Exception:
                return

        sfg = getattr(_ds, 'SharedFunctionalGroupsSequence', None)
        if sfg is not None and len(sfg) > 0:
            _try_add_from_fgs_item(sfg[0])

        pfg = getattr(_ds, 'PerFrameFunctionalGroupsSequence', None)
        if pfg is not None and len(pfg) > 0:
            _try_add_from_fgs_item(pfg[0])

        for cand in candidates:
            if cand is None:
                continue
            try:
                if len(cand) >= 2:
                    spacing_y = float(cand[0])
                    spacing_x = float(cand[1])
                    if spacing_x > 0 and spacing_y > 0:
                        return spacing_x, spacing_y
            except Exception:
                continue

        recon_diam = getattr(_ds, 'ReconstructionDiameter', None)
        cols = getattr(_ds, 'Columns', None)
        rows = getattr(_ds, 'Rows', None)
        if recon_diam is not None and cols is not None and rows is not None:
            try:
                diam_mm = float(recon_diam)
                spacing_x = diam_mm / float(cols)
                spacing_y = diam_mm / float(rows)
                if spacing_x > 0 and spacing_y > 0:
                    return spacing_x, spacing_y
            except Exception:
                pass

        raise ValueError('PixelSpacing missing')

    spacing_x, spacing_y = _get_pixel_spacing_mm(ds)

    slope = getattr(ds, 'RescaleSlope', 1.0)
    intercept = getattr(ds, 'RescaleIntercept', 0.0)
    try:
        slope = float(slope)
        intercept = float(intercept)
    except Exception:
        slope = 1.0
        intercept = 0.0

    arr = ds.pixel_array.astype(np.float32, copy=False)
    hu = arr * slope + intercept

    z = None
    ipp = getattr(ds, 'ImagePositionPatient', None)
    if ipp is not None and len(ipp) >= 3:
        try:
            z = float(ipp[2])
        except Exception:
            z = None

    return hu, spacing_x, spacing_y, z


def _process_single_slice(
    hu_2d: np.ndarray,
    spacing_x: float,
    spacing_y: float,
    z_mm: float,
    downsample: int,
    body_threshold_hu: float,
    fat_hu_range: Tuple[float, float],
    bone_threshold_hu: float,
    thickness_mm: float,
    widths_mm: List[float],
    depths_mm: List[float],
    areas_mm2: List[float],
    spine_to_skin_left: List[float],
    spine_to_skin_right: List[float],
    spine_to_skin_anterior: List[float],
    spine_to_skin_posterior: List[float],
    body_com_sum: np.ndarray,
    spine_com_sum: np.ndarray,
    acc: Dict[str, float],
) -> None:
    if downsample and downsample > 1:
        hu_2d = hu_2d[::downsample, ::downsample]
        spacing_x *= float(downsample)
        spacing_y *= float(downsample)

    body_mask = hu_2d > body_threshold_hu
    if not np.any(body_mask):
        return

    fat_mask = (hu_2d >= fat_hu_range[0]) & (hu_2d <= fat_hu_range[1]) & body_mask
    bone_mask = (hu_2d >= bone_threshold_hu) & body_mask

    body_count = int(body_mask.sum())
    fat_count = int(fat_mask.sum())
    bone_count = int(bone_mask.sum())

    acc['body_voxels'] += body_count
    acc['fat_voxels'] += fat_count
    acc['bone_voxels'] += bone_count

    rows, cols = np.where(body_mask)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())

    width_mm = (c1 - c0 + 1) * spacing_x
    depth_mm = (r1 - r0 + 1) * spacing_y
    area_mm2 = body_count * spacing_x * spacing_y

    widths_mm.append(float(width_mm))
    depths_mm.append(float(depth_mm))
    areas_mm2.append(float(area_mm2))

    ys = rows.astype(np.float64) * spacing_y
    xs = cols.astype(np.float64) * spacing_x
    body_mass = float(body_count) * spacing_x * spacing_y * thickness_mm
    acc['body_mass_sum'] += body_mass
    body_com_sum[0] += float(xs.mean()) * body_mass
    body_com_sum[1] += float(ys.mean()) * body_mass
    body_com_sum[2] += float(z_mm) * body_mass

    if bone_count <= 0:
        return

    rr, cc = np.where(bone_mask)

    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    win_r = max(5, int((r1 - r0 + 1) * 0.30))
    win_c = max(5, int((c1 - c0 + 1) * 0.30))

    in_win = (np.abs(rr - center_r) <= win_r) & (np.abs(cc - center_c) <= win_c)
    if np.any(in_win):
        rr = rr[in_win]
        cc = cc[in_win]

    if rr.size <= 0:
        return

    spine_mass = float(rr.size) * spacing_x * spacing_y * thickness_mm
    acc['spine_mass_sum'] += spine_mass
    spine_com_sum[0] += float(cc.astype(np.float64).mean() * spacing_x) * spine_mass
    spine_com_sum[1] += float(rr.astype(np.float64).mean() * spacing_y) * spine_mass
    spine_com_sum[2] += float(z_mm) * spine_mass

    spine_x = float(cc.astype(np.float64).mean() * spacing_x)
    spine_y = float(rr.astype(np.float64).mean() * spacing_y)

    left_mm = (spine_x - (c0 * spacing_x))
    right_mm = ((c1 * spacing_x) - spine_x)
    posterior_mm = (spine_y - (r0 * spacing_y))
    anterior_mm = ((r1 * spacing_y) - spine_y)

    spine_to_skin_left.append(float(left_mm))
    spine_to_skin_right.append(float(right_mm))
    spine_to_skin_posterior.append(float(posterior_mm))
    spine_to_skin_anterior.append(float(anterior_mm))


def extract_features_from_dicom_folder(
    dicom_folder: Path,
    downsample: int = 2,
    body_threshold_hu: float = -300.0,
    fat_hu_range: Tuple[float, float] = (-190.0, -30.0),
    bone_threshold_hu: float = 300.0,
    max_slices: Optional[int] = None,
    debug: bool = False,
    enable_kidney_segmentation: bool = True,
    kidney_only: bool = False,
) -> Dict[str, Optional[float]]:
    if pydicom is None:
        raise RuntimeError('pydicom is not installed')

    slice_infos = _list_dicom_slices(dicom_folder)
    if not slice_infos:
        raise ValueError('No DICOM slices found')

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
    
    # Попытка извлечь координаты почек через TotalSegmentator
    if enable_kidney_segmentation and totalsegmentator is not None:
        try:
            # Создаем временную папку для сегментации
            seg_output_path = dicom_folder / "segmentation_output"
            seg_output_path.mkdir(exist_ok=True)
            
            # Запускаем сегментацию
            seg_file = _run_totalsegmentator(dicom_folder, seg_output_path, kidney_only=kidney_only)
            
            if seg_file:
                kidney_features = _extract_kidney_coordinates_from_segmentation(seg_file)
                print(f"  📍 TotalSegmentator: извлечены координаты почек для {dicom_folder.name}")
            else:
                print(f"  ⚠️ TotalSegmentator не удалось, пробуем простой детектор...")
                kidney_features = _extract_kidney_coordinates_simple(dicom_folder)
                
        except Exception as e:
            print(f"  ⚠️ Ошибка TotalSegmentator для {dicom_folder.name}: {e}")
            print(f"  🔄 Пробуем простой детектор...")
            kidney_features = _extract_kidney_coordinates_simple(dicom_folder)
    elif enable_kidney_segmentation:
        # Если TotalSegmentator не установлен, используем простой детектор
        print(f"  🔄 TotalSegmentator не установлен, используем простой детектор для {dicom_folder.name}")
        kidney_features = _extract_kidney_coordinates_simple(dicom_folder)
    else:
        kidney_features = {}

    if max_slices is not None and max_slices > 0 and len(slice_infos) > max_slices:
        idx = np.linspace(0, len(slice_infos) - 1, num=max_slices).round().astype(int)
        slice_infos = [slice_infos[i] for i in idx]

    thickness_mm = _estimate_slice_thickness_mm(slice_infos)

    acc = {
        'body_voxels': 0.0,
        'fat_voxels': 0.0,
        'bone_voxels': 0.0,
        'body_mass_sum': 0.0,
        'spine_mass_sum': 0.0,
    }

    widths_mm: List[float] = []
    depths_mm: List[float] = []
    areas_mm2: List[float] = []

    body_com_sum = np.zeros(3, dtype=np.float64)
    spine_com_sum = np.zeros(3, dtype=np.float64)

    spine_to_skin_left: List[float] = []
    spine_to_skin_right: List[float] = []
    spine_to_skin_anterior: List[float] = []
    spine_to_skin_posterior: List[float] = []

    z_positions: List[float] = []

    for i, s in enumerate(slice_infos):
        try:
            hu, spacing_x, spacing_y, z = _read_slice_hu(s.path)
        except Exception as e:
            if debug:
                try:
                    ds_dbg = pydicom.dcmread(str(s.path), stop_before_pixels=True, force=True)
                    modality = str(getattr(ds_dbg, 'Modality', ''))
                    sop = str(getattr(ds_dbg, 'SOPClassUID', ''))
                    rows = getattr(ds_dbg, 'Rows', None)
                    cols = getattr(ds_dbg, 'Columns', None)
                    ps = getattr(ds_dbg, 'PixelSpacing', None)
                    ips = getattr(ds_dbg, 'ImagerPixelSpacing', None)
                    recon = getattr(ds_dbg, 'ReconstructionDiameter', None)
                    print(
                        f"  DEBUG: {dicom_folder.name} file={s.path.name} Modality={modality} SOP={sop} Rows={rows} Cols={cols} "
                        f"PixelSpacing={ps} ImagerPixelSpacing={ips} ReconstructionDiameter={recon} err={e}"
                    )
                except Exception:
                    pass
            raise

        if thickness_mm is None:
            st = None
            try:
                ds = pydicom.dcmread(str(s.path), stop_before_pixels=True, force=True)
                st = getattr(ds, 'SpacingBetweenSlices', None)
                if st is None:
                    st = getattr(ds, 'SliceThickness', None)
                thickness_mm = float(st) if st is not None else None
            except Exception:
                thickness_mm = None

        if thickness_mm is None:
            thickness_mm = 1.0

        if hu.ndim == 3:
            n_frames = hu.shape[0]
            if max_slices is not None and max_slices > 0 and n_frames > max_slices:
                frame_idx = np.linspace(0, n_frames - 1, num=max_slices).round().astype(int)
            else:
                frame_idx = np.arange(n_frames, dtype=int)

            for fi, frame in enumerate(hu[frame_idx]):
                if z is None:
                    z_mm = float(i * len(frame_idx) + fi) * float(thickness_mm)
                else:
                    z_mm = float(z) + float(fi) * float(thickness_mm)
                z_positions.append(float(z_mm))
                _process_single_slice(
                    frame,
                    spacing_x,
                    spacing_y,
                    float(z_mm),
                    downsample,
                    body_threshold_hu,
                    fat_hu_range,
                    bone_threshold_hu,
                    float(thickness_mm),
                    widths_mm,
                    depths_mm,
                    areas_mm2,
                    spine_to_skin_left,
                    spine_to_skin_right,
                    spine_to_skin_anterior,
                    spine_to_skin_posterior,
                    body_com_sum,
                    spine_com_sum,
                    acc,
                )
            continue

        if z is None:
            z_mm = float(i) * float(thickness_mm)
        else:
            z_mm = float(z)

        z_positions.append(float(z_mm))
        _process_single_slice(
            hu,
            spacing_x,
            spacing_y,
            float(z_mm),
            downsample,
            body_threshold_hu,
            fat_hu_range,
            bone_threshold_hu,
            float(thickness_mm),
            widths_mm,
            depths_mm,
            areas_mm2,
            spine_to_skin_left,
            spine_to_skin_right,
            spine_to_skin_anterior,
            spine_to_skin_posterior,
            body_com_sum,
            spine_com_sum,
            acc,
        )

    voxel_volume_mm3 = None
    if thickness_mm is not None:
        voxel_volume_mm3 = float(thickness_mm)

    if acc['body_mass_sum'] > 0:
        body_com = body_com_sum / acc['body_mass_sum']
        body_com_x = float(body_com[0])
        body_com_y = float(body_com[1])
        body_com_z = float(body_com[2])
    else:
        body_com_x = None
        body_com_y = None
        body_com_z = None

    if acc['spine_mass_sum'] > 0:
        spine_com = spine_com_sum / acc['spine_mass_sum']
        spine_x = float(spine_com[0])
        spine_y = float(spine_com[1])
        spine_z = float(spine_com[2])
    else:
        spine_x = None
        spine_y = None
        spine_z = None

    if thickness_mm is None:
        thickness_mm = 1.0

    body_volume_cm3 = float(acc['body_voxels']) * float(thickness_mm) / 1000.0
    fat_volume_cm3 = float(acc['fat_voxels']) * float(thickness_mm) / 1000.0
    bone_volume_cm3 = float(acc['bone_voxels']) * float(thickness_mm) / 1000.0

    fat_ratio = float(acc['fat_voxels']) / float(acc['body_voxels']) if acc['body_voxels'] > 0 else None

    def _median(values: List[float]) -> Optional[float]:
        if not values:
            return None
        return float(np.median(np.array(values, dtype=np.float64)))

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
        'fat_ratio': float(fat_ratio) if fat_ratio is not None else None,
        'body_width_mm_median': _median(widths_mm),
        'body_depth_mm_median': _median(depths_mm),
        'body_area_mm2_median': _median(areas_mm2),
        'body_com_x_mm': body_com_x,
        'body_com_y_mm': body_com_y,
        'body_com_z_mm': body_com_z,
        'spine_center_x_mm': spine_x,
        'spine_center_y_mm': spine_y,
        'spine_center_z_mm': spine_z,
        'spine_to_skin_left_mm_median': _median(spine_to_skin_left),
        'spine_to_skin_right_mm_median': _median(spine_to_skin_right),
        'spine_to_skin_anterior_mm_median': _median(spine_to_skin_anterior),
        'spine_to_skin_posterior_mm_median': _median(spine_to_skin_posterior),
        'slice_count_used': float(len(slice_infos)),
        'slice_thickness_mm': float(thickness_mm),
    }

    return features


def _iter_patient_folders(dicom_root: Path) -> Iterable[Path]:
    for p in dicom_root.iterdir():
        if p.is_dir():
            yield p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('dicom_root', nargs='?', help='Папка, внутри которой лежат подпапки пациентов с DICOM (не требуется при использовании --patient-folder)')
    parser.add_argument('--patient-folder', default=None, help='Если задано, обработать только одну папку пациента (путь)')
    parser.add_argument('--output', default='dicom_features.csv')
    parser.add_argument('--downsample', type=int, default=2)
    parser.add_argument('--max-slices', type=int, default=300)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--include-failures', action='store_true')
    parser.add_argument('--disable-kidney-segmentation', action='store_true', 
                       help='Отключить сегментацию почек через TotalSegmentator')
    parser.add_argument('--kidney-only', action='store_true', 
                       help='Сегментировать только почки (экономия памяти)')
    args = parser.parse_args()

    # Проверяем, что указан либо dicom_root, либо patient-folder
    if not args.patient_folder and not args.dicom_root:
        parser.error("Необходимо указать либо dicom_root, либо --patient-folder")
    
    out_path = Path(args.output)

    if args.patient_folder:
        folder = Path(args.patient_folder)
        if not folder.exists():
            raise FileNotFoundError(str(folder))
        folders_iter: Iterable[Path] = [folder]
        dicom_root = folder.parent  # для информации
    else:
        dicom_root = Path(args.dicom_root)
        if not dicom_root.exists():
            raise FileNotFoundError(str(dicom_root))
        folders_iter = _iter_patient_folders(dicom_root)

    rows: List[Dict[str, object]] = []

    for folder in folders_iter:
        try:
            feats = extract_features_from_dicom_folder(
                folder,
                downsample=args.downsample,
                max_slices=args.max_slices,
                debug=args.debug,
                enable_kidney_segmentation=not args.disable_kidney_segmentation,
                kidney_only=args.kidney_only,
            )

            full_name = None
            patient_name = feats.get('patient_name')
            if patient_name:
                full_name = str(patient_name)
            if not full_name:
                full_name = _extract_full_name_from_folder(folder.name)
            row: Dict[str, object] = {
                'dicom_folder': folder.name,
                'full_name': full_name,
                'full_name_key': _normalize_name(full_name) if full_name else '',
                'dicom_folder_key': _normalize_name(folder.name),
            }
            row.update(feats)
            rows.append(row)
            print(f"✅ {folder.name}: extracted")
        except Exception as e:
            print(f"⚠️ {folder.name}: {e}")

            if args.include_failures:
                full_name = _extract_full_name_from_folder(folder.name)
                rows.append({
                    'dicom_folder': folder.name,
                    'full_name': full_name,
                    'full_name_key': _normalize_name(full_name) if full_name else '',
                    'dicom_folder_key': _normalize_name(folder.name),
                    'status': 'failed',
                    'error': str(e),
                })

    if not rows:
        print('No studies processed')
        return 2

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding='utf-8')
    print(f"Saved: {out_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
