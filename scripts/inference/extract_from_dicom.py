#!/usr/bin/env python3
"""
Скрипт для извлечения данных из DICOM снимков.

Извлекает:
- Метаданные (пол, возраст, вес, рост, ИМТ)
- 3D координаты почек (с помощью TotalSegmentator)

Использование:
    # Обработка одной папки с DICOM файлами
    python extract_from_dicom.py /path/to/dicom/folder
    
    # Обработка нескольких папок
    python extract_from_dicom.py /path/to/folder1 /path/to/folder2
    
    # Обработка конкретного DICOM файла
    python extract_from_dicom.py /path/to/file.dcm
    
    # Смешанный режим
    python extract_from_dicom.py /path/to/folder /path/to/file.dcm --output results.csv
    
    # Только метаданные (без сегментации)
    python extract_from_dicom.py /path/to/folder --no-segmentation
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import pydicom
except ImportError:
    print("❌ Ошибка: pydicom не установлен")
    print("Установите: pip install pydicom")
    sys.exit(1)

try:
    import nibabel as nib
except ImportError:
    print("⚠️ Предупреждение: nibabel не установлен (нужен для чтения NIfTI)")
    print("Установите: pip install nibabel")
    nib = None

try:
    import dcm2niix
    DCM2NIIX_AVAILABLE = True
except ImportError:
    print("⚠️ Предупреждение: dcm2niix не установлен")
    print("Установите: pip install dcm2niix")
    DCM2NIIX_AVAILABLE = False

# TotalSegmentator опционален - проверим позже


def find_dicom_files(folder):
    """Находит DICOM файлы в папке."""
    dicom_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            # Ищем файлы с расширением .dcm/.DCM ИЛИ файлы без расширения
            # (типичная структура DICOM с нумерованными файлами)
            if (file.endswith('.dcm') or file.endswith('.DCM') or 
                ('.' not in file and file.isdigit())):
                dicom_files.append(os.path.join(root, file))
    return sorted(dicom_files)


def find_dicom_folders(paths):
    """
    Находит папки с DICOM файлами из списка путей.
    
    Args:
        paths: Список путей (файлы или папки)
        
    Returns:
        list: Список папок, содержащих DICOM файлы
    """
    dicom_folders = set()
    
    for path in paths:
        path = Path(path)
        
        if not path.exists():
            print(f"  ⚠️ Путь не существует: {path}")
            continue
            
        if path.is_file():
            # Если это файл, проверяем DICOM ли он
            if (path.suffix.lower() in ['.dcm'] or 
                ('.' not in path.name and path.name.isdigit())):
                dicom_folders.add(path.parent)
            else:
                print(f"  ⚠️ Файл не является DICOM: {path}")
                
        elif path.is_dir():
            # Если это папка, ищем в ней DICOM файлы
            dicom_files = find_dicom_files(str(path))
            if dicom_files:
                dicom_folders.add(path)
                print(f"  📁 Найдено DICOM файлов в {path.name}: {len(dicom_files)}")
            else:
                # Если в папке нет файлов, ищем в подпапках
                found_subfolders = False
                for root, dirs, files in os.walk(str(path)):
                    if any(f.endswith('.dcm') or f.endswith('.DCM') or 
                           ('.' not in f and f.isdigit()) for f in files):
                        dicom_folders.add(Path(root))
                        found_subfolders = True
                
                if not found_subfolders:
                    print(f"  ⚠️ DICOM файлы не найдены в: {path}")
    
    return sorted(list(dicom_folders))


def extract_dicom_metadata(dicom_file):
    """
    Извлекает метаданные из DICOM файла.
    
    Args:
        dicom_file: Путь к .dcm файлу
        
    Returns:
        dict с метаданными
    """
    metadata = {}
    
    try:
        dcm = pydicom.dcmread(dicom_file)
        
        # Patient ID
        metadata['patient_id'] = str(dcm.PatientID) if 'PatientID' in dcm else None
        metadata['patient_name'] = str(dcm.PatientName) if 'PatientName' in dcm else None
        
        # Демография
        if 'PatientSex' in dcm:
            sex = dcm.PatientSex
            metadata['sex'] = 1 if sex == 'M' else 0 if sex == 'F' else None
        else:
            metadata['sex'] = None
        
        if 'PatientAge' in dcm:
            age_str = str(dcm.PatientAge).replace('Y', '').replace('y', '')
            try:
                metadata['age'] = int(age_str)
            except:
                metadata['age'] = None
        else:
            metadata['age'] = None
        
        # Антропометрия
        metadata['weight_kg'] = float(dcm.PatientWeight) if 'PatientWeight' in dcm else None
        metadata['height_m'] = float(dcm.PatientSize) if 'PatientSize' in dcm else None
        
        # ИМТ
        if metadata['weight_kg'] and metadata['height_m']:
            metadata['bmi'] = metadata['weight_kg'] / (metadata['height_m'] ** 2)
        else:
            metadata['bmi'] = None
        
        # Позиция пациента
        metadata['patient_position'] = str(dcm.PatientPosition) if 'PatientPosition' in dcm else None
        
        # Параметры сканирования
        metadata['study_date'] = str(dcm.StudyDate) if 'StudyDate' in dcm else None
        metadata['slice_thickness'] = float(dcm.SliceThickness) if 'SliceThickness' in dcm else None
        
    except Exception as e:
        print(f"  ⚠️ Ошибка чтения метаданных: {e}")
    
    return metadata


def get_kidney_coordinates_from_mask(mask_file):
    """
    Извлекает координаты 3 точек почки из маски сегментации.
    
    Args:
        mask_file: Путь к .nii.gz файлу с маской
        
    Returns:
        dict: {'upper': {'x', 'y', 'z'}, 'middle': {...}, 'lower': {...}}
    """
    if nib is None:
        print("  ⚠️ nibabel не установлен, пропускаем извлечение координат")
        return None
    
    try:
        # Загружаем маску
        kidney_img = nib.load(mask_file)
        kidney_data = kidney_img.get_fdata()
        affine = kidney_img.affine
        
        # Находим вокселы почки
        kidney_voxels = np.argwhere(kidney_data > 0)
        
        if len(kidney_voxels) == 0:
            print("  ⚠️ Маска пустая (почка не найдена)")
            return None
        
        # Находим границы по Z (краниокаудальное направление)
        z_min = kidney_voxels[:, 2].min()
        z_max = kidney_voxels[:, 2].max()
        z_middle = int((z_min + z_max) / 2)
        
        # Уровни для 3 точек
        z_upper = int(z_min + (z_max - z_min) * 0.25)   # Верхняя четверть
        z_lower = int(z_min + (z_max - z_min) * 0.75)   # Нижняя четверть
        
        coords = {}
        
        for level_name, z_level in [('upper', z_upper), ('middle', z_middle), ('lower', z_lower)]:
            # Находим вокселы на этом уровне Z
            slice_voxels = kidney_voxels[
                (kidney_voxels[:, 2] >= z_level - 2) & 
                (kidney_voxels[:, 2] <= z_level + 2)
            ]
            
            if len(slice_voxels) == 0:
                slice_voxels = kidney_voxels
            
            # Центроид в вокселях
            centroid_voxel = slice_voxels.mean(axis=0)
            
            # Переводим в мировые координаты (мм)
            centroid_world = affine @ np.append(centroid_voxel, 1)
            
            coords[level_name] = {
                'x': float(centroid_world[0]),
                'y': float(centroid_world[1]),
                'z': float(centroid_world[2])
            }
        
        return coords
        
    except Exception as e:
        print(f"  ⚠️ Ошибка извлечения координат: {e}")
        return None


def convert_dicom_to_nifti(dicom_folder, output_folder, compress=True, filename_pattern='%p_%t_%s', timeout_seconds=1800):
    """
    Конвертирует DICOM файлы в NIfTI формат с помощью dcm2niix.
    
    Args:
        dicom_folder: Папка с DICOM файлами
        output_folder: Папка для сохранения NIfTI файлов
        compress: Использовать gzip сжатие (.nii.gz)
        filename_pattern: Шаблон имени файла
        timeout_seconds: Время ожидания конвертации в секундах
        
    Returns:
        tuple: (успех, список_nifti_файлов)
    """
    if not DCM2NIIX_AVAILABLE:
        print("  ⚠️ dcm2niix недоступен, пропускаем конвертацию")
        return False, []
    
    try:
        import subprocess
        import shutil
        import sys
        import os
        
        # Создаем выходную папку
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Находим dcm2niix
        dcm2niix_cmd = shutil.which('dcm2niix')
        if not dcm2niix_cmd:
            # Ищем в Scripts папке Python
            scripts_dir = os.path.join(os.path.dirname(sys.executable), 'Scripts')
            dcm2niix_path = os.path.join(scripts_dir, 'dcm2niix.exe')
            if os.path.exists(dcm2niix_path):
                dcm2niix_cmd = dcm2niix_path
            else:
                print("  dcm2niix не найден, пропускаем конвертацию")
                return False, []
        
        # Формируем команду dcm2niix
        cmd = [
            dcm2niix_cmd,
            '-z', 'y' if compress else 'n',      # сжатие
            '-f', filename_pattern,              # шаблон имени
            '-o', output_folder,                  # выходная папка
            '-b', 'n',                            # не создавать .json файлы
            '-m', 'y',
            '-x', 'n',                            # не обрезать изображение
            dicom_folder                          # входная папка
        ]
        
        print(f"  Конвертация DICOM → NIfTI...")
        
        # Запускаем конвертацию
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
        
        if result.returncode == 0:
            # Ищем созданные NIfTI файлы
            nifti_files = [str(f) for f in Path(output_folder).glob('*.nii*')]
            if not nifti_files:
                print("  ❌ Конвертация завершилась без ошибок, но NIfTI файлы не найдены")
                return False, []
            
            print(f"  ✅ Конвертация успешна: {len(nifti_files)} NIfTI файлов")
            return True, nifti_files
        else:
            print(f"  ❌ Ошибка конвертации: {result.stderr}")
            return False, []
            
    except subprocess.TimeoutExpired:
        print(f"  ⏰ Превышено время конвертации ({timeout_seconds} секунд)")
        return False, []
    except Exception as e:
        print(f"  Ошибка конвертации: {e}")
        return False, []


def run_totalsegmentator(input_folder, output_folder, fast=False, roi_subset=None):
    """
    Запускает TotalSegmentator для сегментации почек.
    
    Args:
        input_folder: Папка с DICOM
        output_folder: Папка для сохранения масок
        fast: Использовать быстрый режим
        roi_subset: Список органов для сегментации
        
    Returns:
        True если успешно, False если ошибка
    """
    try:
        # Проверяем установлен ли TotalSegmentator
        from totalsegmentator.python_api import totalsegmentator
    except ImportError:
        print("  TotalSegmentator не установлен")
        print("  Установите: pip install TotalSegmentator")
        return False
    
    def _run(fast_mode: bool) -> None:
        print(f"  Запуск TotalSegmentator...")
        if fast_mode:
            print(f"  Используется быстрый режим")

        # Уменьшаем использование памяти/потоков для numpy/BLAS
        import os
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'

        totalsegmentator(
            input=input_folder,
            output=output_folder,
            ml=True,
            nr_thr_resamp=1,
            nr_thr_saving=1,
            roi_subset=roi_subset or ['kidney_right', 'kidney_left'],
            fast=fast_mode,
            fastest=fast_mode,
            robust_crop=True,
            quiet=True,
            device='cpu'  # Явно указываем CPU
        )

    try:
        _run(fast)
        print(f"  Сегментация завершена")
        return True
        
    except MemoryError as e:
        if not fast:
            print(f"  Ошибка памяти: {e}")
            print(f"  Пробуем повторить в режиме --fast...")
            try:
                _run(True)
                print(f"  Сегментация завершена")
                return True
            except Exception as retry_e:
                e = retry_e

        print(f"  Ошибка памяти: {e}")
        print(f"  Попробуйте:")
        print(f"     - Использовать --no-segmentation для извлечения только метаданных")
        print(f"     - Уменьшить количество DICOM файлов (первые 100-200)")
        print(f"     - Добавить больше оперативной памяти")
        return False
    except Exception as e:
        if ("Unable to allocate" in str(e)) or ("memory" in str(e).lower()):
            if not fast:
                print(f"  Ошибка памяти: {e}")
                print(f"  Пробуем повторить в режиме --fast...")
                try:
                    _run(True)
                    print(f"  Сегментация завершена")
                    return True
                except Exception as retry_e:
                    e = retry_e

            print(f"  Ошибка памяти: {e}")
            print(f"  Попробуйте:")
            print(f"     - Использовать --no-segmentation для извлечения только метаданных")
            print(f"     - Уменьшить количество DICOM файлов (первые 100-200)")
            print(f"     - Добавить больше оперативной памяти")
            print(f"  Ошибка TotalSegmentator: {e}")
        return False


def process_patient_folder(patient_folder, use_totalsegmentator=True, temp_dir='/tmp', fast=False, roi_subset=None, max_files=None, use_dcm2niix=True, compress_nifti=True, reuse_nifti=False):
    """
    Обрабатывает папку одного пациента.
    
    Args:
        patient_folder: Путь к папке с DICOM пациента
        use_totalsegmentator: Использовать ли TotalSegmentator
        temp_dir: Папка для временных файлов
        fast: Использовать быстрый режим
        roi_subset: Список органов для сегментации
        max_files: Максимальное количество файлов для обработки
        use_dcm2niix: Использовать ли dcm2niix для конвертации
        compress_nifti: Использовать сжатие NIfTI
        reuse_nifti: Использовать уже созданные NIfTI во временной папке
        
    Returns:
        dict с данными пациента или None
    """
    patient_folder = Path(patient_folder)
    patient_id = patient_folder.name
    
    print(f"\n Обработка: {patient_id}")
    
    # 1. Находим DICOM файлы
    dicom_files = find_dicom_files(patient_folder)
    
    if not dicom_files:
        print(f"  DICOM файлы не найдены")
        return None
    
    # Ограничиваем количество файлов если указано
    if max_files and len(dicom_files) > max_files:
        dicom_files = dicom_files[:max_files]
        print(f"  Ограничено до {max_files} файлов для экономии памяти")
    
    print(f"  Найдено DICOM файлов: {len(dicom_files)}")
    
    # 2. Извлекаем метаданные из первого файла
    metadata = extract_dicom_metadata(dicom_files[0])
    metadata['patient_id'] = patient_id
    metadata['dicom_folder'] = str(patient_folder)
    metadata['num_dicom_files'] = len(dicom_files)
    
    # Добавляем информацию о конвертации
    metadata['dcm2niix_used'] = use_dcm2niix and DCM2NIIX_AVAILABLE
    metadata['nifti_compressed'] = compress_nifti if use_dcm2niix else None
    metadata['reuse_nifti'] = bool(reuse_nifti)
    
    print(f"  Метаданные:")
    if metadata.get('sex') is not None:
        sex_label = 'М' if metadata['sex'] == 1 else 'Ж'
        print(f"     Пол: {sex_label}")
    if metadata.get('age'):
        print(f"     Возраст: {metadata['age']} лет")
    if metadata.get('bmi'):
        print(f"     ИМТ: {metadata['bmi']:.1f}")
    if metadata.get('patient_position'):
        print(f"     Позиция: {metadata['patient_position']}")
    
    # 3. Конвертация DICOM → NIfTI (если включено)
    nifti_folder = None
    nifti_files = []
    nifti_input_file = None
    conversion_success = False
    if use_dcm2niix and DCM2NIIX_AVAILABLE:
        candidate_folder = Path(temp_dir) / f'nifti_{patient_id}'
        existing_nifti_files = [str(f) for f in candidate_folder.glob('*.nii*')] if candidate_folder.exists() else []

        if reuse_nifti and existing_nifti_files:
            conversion_success, nifti_files = True, existing_nifti_files
        else:
            try:
                import shutil
                if candidate_folder.exists():
                    shutil.rmtree(candidate_folder)
            except Exception:
                pass

            conversion_success, nifti_files = convert_dicom_to_nifti(
                str(patient_folder),
                str(candidate_folder),
                compress=compress_nifti
            )
        
        if conversion_success and nifti_files:
            nifti_folder = candidate_folder
            try:
                nifti_input_file = max(nifti_files, key=lambda p: Path(p).stat().st_size)
            except Exception:
                nifti_input_file = nifti_files[0]

            metadata['nifti_files_count'] = len(nifti_files)
            metadata['nifti_folder'] = str(nifti_folder)
            metadata['nifti_input_file'] = str(nifti_input_file) if nifti_input_file else None
            print(f"  Создано NIfTI файлов: {len(nifti_files)}")
        else:
            print("  Конвертация NIfTI не удалась")
    
    # 4. Запускаем TotalSegmentator (если включено)
    if use_totalsegmentator:
        if not nifti_input_file:
            print("  Сегментация пропущена: TotalSegmentator ожидает NIfTI, но конвертация не выполнена")
            return metadata
        
        output_folder = Path(temp_dir) / f'segmentation_{patient_id}'
        try:
            import shutil
            if output_folder.exists():
                shutil.rmtree(output_folder)
        except Exception:
            pass
        output_folder.mkdir(parents=True, exist_ok=True)
        
        print("  Входные данные для сегментации: NIfTI")
        
        success = run_totalsegmentator(
            str(nifti_input_file),
            str(output_folder),
            fast=fast,
            roi_subset=roi_subset
        )
        
        if success:
            # 5. Извлекаем координаты из масок
            # Правая почка
            kidney_right_file = output_folder / 'kidney_right.nii.gz'
            if kidney_right_file.exists():
                coords_right = get_kidney_coordinates_from_mask(str(kidney_right_file))
                
                if coords_right:
                    print(f"  Правая почка:")
                    for level, coord in coords_right.items():
                        print(f"     {level}: X={coord['x']:.1f}, Y={coord['y']:.1f}, Z={coord['z']:.1f}")
                        metadata[f'X_{level}_right'] = coord['x']
                        metadata[f'Y_{level}_right'] = coord['y']
                        metadata[f'Z_{level}_right'] = coord['z']
            
            # Левая почка
            kidney_left_file = output_folder / 'kidney_left.nii.gz'
            if kidney_left_file.exists():
                coords_left = get_kidney_coordinates_from_mask(str(kidney_left_file))
                
                if coords_left:
                    print(f"  Левая почка:")
                    for level, coord in coords_left.items():
                        print(f"     {level}: X={coord['x']:.1f}, Y={coord['y']:.1f}, Z={coord['z']:.1f}")
                        metadata[f'X_{level}_left'] = coord['x']
                        metadata[f'Y_{level}_left'] = coord['y']
                        metadata[f'Z_{level}_left'] = coord['z']
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Извлечение данных из DICOM снимков'
    )
    parser.add_argument(
        'paths',
        nargs='+',
        help='Пути к DICOM файлам или папкам (можно указать несколько через пробел)'
    )
    parser.add_argument(
        '--output', '-o',
        default='extracted_from_dicom.csv',
        help='Путь для сохранения CSV файла (по умолчанию: extracted_from_dicom.csv)'
    )
    parser.add_argument(
        '--no-segmentation',
        action='store_true',
        help='Не запускать TotalSegmentator (только метаданные)'
    )
    parser.add_argument(
        '--temp-dir',
        default='/tmp',
        help='Папка для временных файлов сегментации'
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Использовать быстрый режим TotalSegmentator (меньше памяти, ниже качество)'
    )
    parser.add_argument(
        '--roi-subset',
        nargs='*',
        default=['kidney_right', 'kidney_left'],
        help='Список органов для сегментации (по умолчанию: kidney_right kidney_left)'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Максимальное количество DICOM файлов для обработки (по умолчанию: все)'
    )
    parser.add_argument(
        '--no-dcm2niix',
        action='store_true',
        help='Отключить конвертацию DICOM → NIfTI (использовать DICOM напрямую)'
    )
    parser.add_argument(
        '--no-compression',
        action='store_true',
        help='Не использовать сжатие NIfTI (.nii вместо .nii.gz)'
    )
    parser.add_argument(
        '--nifti-only',
        action='store_true',
        help='Только конвертировать DICOM → NIfTI без сегментации'
    )
    parser.add_argument(
        '--reuse-nifti',
        action='store_true',
        help='Использовать уже созданные NIfTI во временной папке (не запускать повторно dcm2niix)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print(" ИЗВЛЕЧЕНИЕ ДАННЫХ ИЗ DICOM")
    print("="*80)
    print(f"\n Указано путей: {len(args.paths)}")
    for path in args.paths:
        print(f"  - {path}")
    print(f" Выходной файл: {args.output}")
    print(f" TotalSegmentator: {'Выключен' if args.no_segmentation or args.nifti_only else 'Включен'}")
    print(f" dcm2niix: {'Отключен' if args.no_dcm2niix else 'Включен'}")
    print(f" Сжатие NIfTI: {'Отключено' if args.no_compression else 'Включено'}")
    print(f" Reuse NIfTI: {'Включен' if args.reuse_nifti else 'Выключен'}")
    if args.nifti_only:
        print(f" Режим: Только конвертация NIfTI")
    
    # Находим все папки с DICOM файлами
    print(f"\n Поиск DICOM файлов...")
    patient_folders = find_dicom_folders(args.paths)
    
    if not patient_folders:
        print(f"\n Ошибка: Папки с DICOM файлами не найдены")
        sys.exit(1)
    
    print(f"\nНайдено папок с DICOM: {len(patient_folders)}")
    
    # Обрабатываем каждого пациента
    all_patients_data = []

    for i, patient_folder in enumerate(patient_folders, 1):
        print(f"\n[{i}/{len(patient_folders)}]", end=" ")

        patient_data = process_patient_folder(
            patient_folder,
            use_totalsegmentator=not args.no_segmentation and not args.nifti_only,
            temp_dir=args.temp_dir,
            fast=args.fast,
            roi_subset=args.roi_subset,
            max_files=args.max_files,
            use_dcm2niix=not args.no_dcm2niix,
            compress_nifti=not args.no_compression,
            reuse_nifti=args.reuse_nifti
        )

        if patient_data:
            all_patients_data.append(patient_data)

    # Сохраняем в CSV
    if all_patients_data:
        df = pd.DataFrame(all_patients_data)
        
        print(f"\n" + "="*80)
        print(f" РЕЗУЛЬТАТЫ")
        print("="*80)
        print(f"\nУспешно обработано пациентов: {len(df)}")
        print(f"Колонок данных: {len(df.columns)}")
        
        # Показываем статистику
        print(f"\n Статистика:")
        if 'sex' in df.columns:
            males = (df['sex'] == 1).sum()
            females = (df['sex'] == 0).sum()
            print(f"  Мужчины: {males}, Женщины: {females}")
        
        if 'age' in df.columns:
            print(f"  Средний возраст: {df['age'].mean():.1f} лет")
        
        if 'bmi' in df.columns:
            print(f"  Средний ИМТ: {df['bmi'].mean():.1f}")
        
        # Проверяем наличие координат
        coord_cols = [col for col in df.columns if col.startswith(('X_', 'Y_', 'Z_'))]
        if coord_cols:
            print(f"  Координатных колонок: {len(coord_cols)}")
            complete_coords = df[coord_cols].notna().all(axis=1).sum()
            print(f"  Пациентов с полными координатами: {complete_coords}")
        
        # Сохраняем
        df.to_csv(args.output, index=False)
        print(f"\n Данные сохранены: {args.output}")
        
    else:
        print(f"\n Не удалось извлечь данные ни из одного пациента")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()