import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """Расширенный feature engineering для предсказания смещения почек"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def create_relative_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создает относительные координаты"""
        logger.info("Создание относительных координат...")
        
        # Относительно позвоночника
        if 'kidney_left_center_x_mm' in df.columns and 'spine_center_x_mm' in df.columns:
            df['kidney_left_x_rel_spine'] = df['kidney_left_center_x_mm'] - df['spine_center_x_mm']
            df['kidney_right_x_rel_spine'] = df['kidney_right_center_x_mm'] - df['spine_center_x_mm']
        
        if 'kidney_left_center_y_mm' in df.columns and 'spine_center_y_mm' in df.columns:
            df['kidney_left_y_rel_spine'] = df['kidney_left_center_y_mm'] - df['spine_center_y_mm']
            df['kidney_right_y_rel_spine'] = df['kidney_right_center_y_mm'] - df['spine_center_y_mm']
        
        # Относительно центра тела
        if 'body_com_x_mm' in df.columns:
            df['kidney_left_x_rel_body'] = df['kidney_left_center_x_mm'] - df['body_com_x_mm']
            df['kidney_right_x_rel_body'] = df['kidney_right_center_x_mm'] - df['body_com_x_mm']
        
        # Нормализация на размеры тела
        if 'body_width_mm_median' in df.columns and df['body_width_mm_median'].notna().any():
            df['kidney_left_x_norm'] = df['kidney_left_center_x_mm'] / df['body_width_mm_median']
            df['kidney_right_x_norm'] = df['kidney_right_center_x_mm'] / df['body_width_mm_median']
        
        if 'body_depth_mm_median' in df.columns and df['body_depth_mm_median'].notna().any():
            df['kidney_left_y_norm'] = df['kidney_left_center_y_mm'] / df['body_depth_mm_median']
            df['kidney_right_y_norm'] = df['kidney_right_center_y_mm'] / df['body_depth_mm_median']
        
        return df
    
    def create_anatomical_relations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создает анатомические отношения между органами"""
        logger.info("Создание анатомических отношений...")
        
        # Расстояние почка-позвоночник
        if all(col in df.columns for col in ['kidney_left_center_x_mm', 'spine_center_x_mm', 
                                        'kidney_left_center_y_mm', 'spine_center_y_mm']):
            df['kidney_left_spine_dist'] = np.sqrt(
                (df['kidney_left_center_x_mm'] - df['spine_center_x_mm'])**2 +
                (df['kidney_left_center_y_mm'] - df['spine_center_y_mm'])**2
            )
            
            df['kidney_right_spine_dist'] = np.sqrt(
                (df['kidney_right_center_x_mm'] - df['spine_center_x_mm'])**2 +
                (df['kidney_right_center_y_mm'] - df['spine_center_y_mm'])**2
            )
        
        # Расстояние до поверхности тела
        if all(col in df.columns for col in ['kidney_left_distance_to_skin_mm', 'kidney_right_distance_to_skin_mm']):
            df['kidney_left_skin_ratio'] = df['kidney_left_distance_to_skin_mm'] / df['body_depth_mm_median']
            df['kidney_right_skin_ratio'] = df['kidney_right_distance_to_skin_mm'] / df['body_depth_mm_median']
        
        # Симметрия почек
        if all(col in df.columns for col in ['kidney_left_center_x_mm', 'kidney_right_center_x_mm',
                                        'kidney_left_center_y_mm', 'kidney_right_center_y_mm']):
            df['kidneys_symmetry_x'] = df['kidney_left_center_x_mm'] + df['kidney_right_center_x_mm']
            df['kidneys_symmetry_y'] = df['kidney_left_center_y_mm'] + df['kidney_right_center_y_mm']
            df['kidneys_asymmetry_x'] = df['kidney_left_center_x_mm'] - df['kidney_right_center_x_mm']
            df['kidneys_asymmetry_y'] = df['kidney_left_center_y_mm'] - df['kidney_right_center_y_mm']
        
        return df
    
    def create_geometric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создает геометрические признаки почек"""
        logger.info("Создание геометрических признаков...")
        
        # Ось почки (вектор верх→низ)
        for kidney in ['left', 'right']:
            if all(col in df.columns for col in [f'kidney_{kidney}_upper_z_mm', 
                                            f'kidney_{kidney}_lower_z_mm']):
                df[f'kidney_{kidney}_axis_vector_z'] = (
                    df[f'kidney_{kidney}_lower_z_mm'] - df[f'kidney_{kidney}_upper_z_mm']
                )
        
        # Угол наклона относительно вертикали
        for kidney in ['left', 'right']:
            if all(col in df.columns for col in [f'kidney_{kidney}_axis_vector_z',
                                            f'kidney_{kidney}_length_mm']):
                df[f'kidney_{kidney}_tilt_angle'] = np.arctan2(
                    df[f'kidney_{kidney}_axis_vector_z'], 
                    df[f'kidney_{kidney}_length_mm']
                ) * 180 / np.pi
        
        # Отношения размеров
        for kidney in ['left', 'right']:
            if all(col in df.columns for col in [f'kidney_{kidney}_length_mm',
                                            f'kidney_{kidney}_bbox_width_mm',
                                            f'kidney_{kidney}_bbox_height_mm']):
                df[f'kidney_{kidney}_aspect_ratio'] = (
                    df[f'kidney_{kidney}_bbox_width_mm'] / df[f'kidney_{kidney}_bbox_height_mm']
                )
                df[f'kidney_{kidney}_volume_est'] = (
                    df[f'kidney_{kidney}_length_mm'] * 
                    df[f'kidney_{kidney}_bbox_width_mm'] * 
                    df[f'kidney_{kidney}_bbox_height_mm'] * np.pi / 6
                )
        
        return df
    
    def create_body_composition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создает признаки композиции тела"""
        logger.info("Создание признаков композиции тела...")
        
        # Отношение жировой ткани
        if all(col in df.columns for col in ['fat_volume_cm3', 'body_volume_cm3']):
            df['fat_ratio'] = df['fat_volume_cm3'] / df['body_volume_cm3']
        
        # Отношение костной ткани  
        if all(col in df.columns for col in ['bone_volume_cm3', 'body_volume_cm3']):
            df['bone_ratio'] = df['bone_volume_cm3'] / df['body_volume_cm3']
        
        # Индикаторы ожирения
        if 'bmi' in df.columns:
            df['obesity_class'] = pd.cut(df['bmi'], 
                                     bins=[0, 18.5, 25, 30, 100], 
                                     labels=['underweight', 'normal', 'overweight', 'obese'])
            df['obesity_class'] = df['obesity_class'].cat.codes
        
        return df
    
    def create_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создает признаки положения пациента"""
        logger.info("Создание признаков положения...")
        
        # Нормализованные антропометрические признаки
        if all(col in df.columns for col in ['age', 'weight_kg', 'height_m']):
            df['bmi_normalized'] = df['bmi'] / 25.0  # норма BMI
            df['weight_normalized'] = df['weight_kg'] / 70.0  # средний вес
            df['height_normalized'] = df['height_m'] / 1.75  # средний рост
        
        # Возрастные группы
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], 
                                   bins=[0, 30, 50, 70, 100], 
                                   labels=['young', 'middle', 'senior', 'elderly'])
            df['age_group'] = df['age_group'].cat.codes
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Основной метод feature engineering"""
        logger.info("Начало расширенного feature engineering...")
        
        df = df.copy()
        
        # Применяем все преобразования
        df = self.create_relative_coordinates(df)
        df = self.create_anatomical_relations(df)
        df = self.create_geometric_features(df)
        df = self.create_body_composition_features(df)
        df = self.create_position_features(df)
        
        # Сохраняем названия признаков
        self.feature_names = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        
        logger.info(f"Создано {len(self.feature_names)} признаков")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Возвращает названия engineered признаков"""
        return self.feature_names
    
    def prepare_for_training(self, df: pd.DataFrame, target_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Подготавливает данные для обучения"""
        # Выбираем только engineered признаки
        feature_cols = [col for col in self.feature_names if col in df.columns]
        
        X = df[feature_cols].fillna(0).values
        y = df[target_cols].fillna(0).values
        
        # Нормализация
        X = self.scaler.fit_transform(X)
        
        logger.info(f"Подготовлено X: {X.shape}, y: {y.shape}")
        
        return X, y

if __name__ == "__main__":
    # Тестирование
    from pathlib import Path
    
    # Загрузка данных
    data_path = Path("data/processed/train.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
        
        engineer = AdvancedFeatureEngineer()
        df_enhanced = engineer.engineer_features(df)
        
        print(f"Исходных признаков: {df.shape[1]}")
        print(f"Расширенных признаков: {df_enhanced.shape[1]}")
        print(f"Новые признаки: {len(engineer.get_feature_names())}")
        
        # Пример новых признаков
        new_features = [col for col in df_enhanced.columns if col not in df.columns]
        print(f"Примеры новых признаков: {new_features[:10]}")
    else:
        logger.warning("Файл train.csv не найден")
