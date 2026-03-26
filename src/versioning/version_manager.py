import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class VersionManager:
    """Версионирование модели и признаков"""
    
    def __init__(self):
        self.current_model_version = "model_v1"
        self.current_features_version = "features_v1"
        self.current_pipeline_version = "pipeline_v1"
        self.current_system_version = "system_v1"
        
        # Директории для версионированных артефактов
        self.models_dir = Path("models")
        self.features_dir = Path("data/features")
        self.pipelines_dir = Path("data/pipelines")
        self.system_dir = Path("data/system")
        
        # Создаем директории
        for dir_path in [self.models_dir, self.features_dir, self.pipelines_dir, self.system_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def save_versioned_artifact(self, artifact, artifact_type: str, version: str, 
                              metadata: Optional[Dict] = None):
        """
        Сохранение артефакта с версией
        
        Args:
            artifact: сохраняемый артефакт
            artifact_type: тип артефакта (model, features, pipeline, system)
            version: версия
            metadata: дополнительные метаданные
        """
        timestamp = datetime.now().isoformat()
        
        # Формирование имени файла
        filename = f"{artifact_type}_{version}"
        
        # Метаданные
        artifact_metadata = {
            'version': version,
            'artifact_type': artifact_type,
            'timestamp': timestamp,
            'system_version': self.current_system_version,
            'metadata': metadata or {}
        }
        
        try:
            if artifact_type == "model":
                filepath = self.models_dir / f"{filename}.pkl"
                joblib.dump(artifact, filepath)
                
                # Сохраняем метаданные
                metadata_path = self.models_dir / f"{filename}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(artifact_metadata, f, indent=2)
                
            elif artifact_type == "features":
                filepath = self.features_dir / f"{filename}.csv"
                if isinstance(artifact, pd.DataFrame):
                    artifact.to_csv(filepath, index=False)
                else:
                    # Если это не DataFrame, сохраняем как JSON
                    json_path = self.features_dir / f"{filename}.json"
                    with open(json_path, 'w') as f:
                        json.dump(artifact, f, indent=2, default=str)
                    filepath = json_path
                
                # Сохраняем метаданные
                metadata_path = self.features_dir / f"{filename}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(artifact_metadata, f, indent=2)
                
            elif artifact_type == "pipeline":
                filepath = self.pipelines_dir / f"{filename}.pkl"
                joblib.dump(artifact, filepath)
                
                # Сохраняем метаданные
                metadata_path = self.pipelines_dir / f"{filename}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(artifact_metadata, f, indent=2)
                
            elif artifact_type == "system":
                filepath = self.system_dir / f"{filename}.json"
                with open(filepath, 'w') as f:
                    json.dump(artifact, f, indent=2, default=str)
                
                # Сохраняем метаданные
                metadata_path = self.system_dir / f"{filename}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(artifact_metadata, f, indent=2)
                
            else:
                raise ValueError(f"Unsupported artifact type: {artifact_type}")
            
            logger.info(f"Артефакт {artifact_type} версии {version} сохранен: {filepath}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения артефакта {artifact_type} версии {version}: {e}")
            raise
    
    def load_versioned_artifact(self, artifact_type: str, version: str):
        """
        Загрузка артефакта версии
        
        Args:
            artifact_type: тип артефакта
            version: версия
            
        Returns:
            загруженный артефакт
        """
        filename = f"{artifact_type}_{version}"
        
        try:
            if artifact_type == "model":
                filepath = self.models_dir / f"{filename}.pkl"
                if not filepath.exists():
                    raise FileNotFoundError(f"Model file not found: {filepath}")
                artifact = joblib.load(filepath)
                
            elif artifact_type == "features":
                # Пробуем CSV, потом JSON
                csv_path = self.features_dir / f"{filename}.csv"
                json_path = self.features_dir / f"{filename}.json"
                
                if csv_path.exists():
                    artifact = pd.read_csv(csv_path)
                elif json_path.exists():
                    with open(json_path, 'r') as f:
                        artifact = json.load(f)
                else:
                    raise FileNotFoundError(f"Features file not found: {csv_path} or {json_path}")
                
            elif artifact_type == "pipeline":
                filepath = self.pipelines_dir / f"{filename}.pkl"
                if not filepath.exists():
                    raise FileNotFoundError(f"Pipeline file not found: {filepath}")
                artifact = joblib.load(filepath)
                
            elif artifact_type == "system":
                filepath = self.system_dir / f"{filename}.json"
                if not filepath.exists():
                    raise FileNotFoundError(f"System file not found: {filepath}")
                with open(filepath, 'r') as f:
                    artifact = json.load(f)
                
            else:
                raise ValueError(f"Unsupported artifact type: {artifact_type}")
            
            logger.info(f"Артефакт {artifact_type} версии {version} загружен")
            return artifact
            
        except Exception as e:
            logger.error(f"Ошибка загрузки артефакта {artifact_type} версии {version}: {e}")
            raise
    
    def get_artifact_metadata(self, artifact_type: str, version: str) -> Dict:
        """Получение метаданных артефакта"""
        filename = f"{artifact_type}_{version}_metadata.json"
        
        if artifact_type == "model":
            metadata_path = self.models_dir / filename
        elif artifact_type == "features":
            metadata_path = self.features_dir / filename
        elif artifact_type == "pipeline":
            metadata_path = self.pipelines_dir / filename
        elif artifact_type == "system":
            metadata_path = self.system_dir / filename
        else:
            raise ValueError(f"Unsupported artifact type: {artifact_type}")
        
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            logger.error(f"Ошибка загрузки метаданных: {e}")
            return {}
    
    def list_available_versions(self, artifact_type: str) -> List[str]:
        """Получение списка доступных версий для типа артефакта"""
        versions = []
        
        if artifact_type == "model":
            pattern = "model_v*.pkl"
            search_dir = self.models_dir
        elif artifact_type == "features":
            pattern = "features_v*.csv"
            search_dir = self.features_dir
        elif artifact_type == "pipeline":
            pattern = "pipeline_v*.pkl"
            search_dir = self.pipelines_dir
        elif artifact_type == "system":
            pattern = "system_v*.json"
            search_dir = self.system_dir
        else:
            raise ValueError(f"Unsupported artifact type: {artifact_type}")
        
        try:
            for filepath in search_dir.glob(pattern):
                # Извлекаем версию из имени файла
                version = filepath.stem.split('_', 1)[1]
                versions.append(version)
            
            # Сортируем по версии
            versions.sort()
            return versions
            
        except Exception as e:
            logger.error(f"Ошибка получения списка версий: {e}")
            return []
    
    def get_latest_version(self, artifact_type: str) -> Optional[str]:
        """Получение последней версии"""
        versions = self.list_available_versions(artifact_type)
        return versions[-1] if versions else None
    
    def create_version_snapshot(self, description: str = "") -> str:
        """
        Создание снепшота всех текущих версий
        
        Args:
            description: описание снепшота
            
        Returns:
            ID снепшота
        """
        snapshot_id = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        snapshot_data = {
            'snapshot_id': snapshot_id,
            'timestamp': datetime.now().isoformat(),
            'description': description,
            'versions': {
                'model': self.current_model_version,
                'features': self.current_features_version,
                'pipeline': self.current_pipeline_version,
                'system': self.current_system_version
            }
        }
        
        # Сохраняем снепшот
        snapshot_path = self.system_dir / f"{snapshot_id}.json"
        with open(snapshot_path, 'w') as f:
            json.dump(snapshot_data, f, indent=2)
        
        logger.info(f"Снепшот {snapshot_id} создан")
        return snapshot_id
    
    def list_snapshots(self) -> List[Dict]:
        """Получение списка снепшотов"""
        snapshots = []
        
        try:
            for filepath in self.system_dir.glob("snapshot_*.json"):
                with open(filepath, 'r') as f:
                    snapshot = json.load(f)
                    snapshots.append(snapshot)
            
            # Сортируем по времени
            snapshots.sort(key=lambda x: x['timestamp'], reverse=True)
            return snapshots
            
        except Exception as e:
            logger.error(f"Ошибка получения списка снепшотов: {e}")
            return []
    
    def restore_snapshot(self, snapshot_id: str) -> bool:
        """
        Восстановление версий из снепшота
        
        Args:
            snapshot_id: ID снепшота
            
        Returns:
            успешность восстановления
        """
        try:
            snapshot_path = self.system_dir / f"{snapshot_id}.json"
            
            with open(snapshot_path, 'r') as f:
                snapshot = json.load(f)
            
            # Восстанавливаем версии
            self.current_model_version = snapshot['versions']['model']
            self.current_features_version = snapshot['versions']['features']
            self.current_pipeline_version = snapshot['versions']['pipeline']
            self.current_system_version = snapshot['versions']['system']
            
            logger.info(f"Версии восстановлены из снепшота {snapshot_id}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка восстановления снепшота {snapshot_id}: {e}")
            return False

if __name__ == "__main__":
    # Тестирование версионирования
    logging.basicConfig(level=logging.INFO)
    logger.info("Тестирование системы версионирования")
    
    version_manager = VersionManager()
    
    # Тестовый артефакт
    test_model = {
        'type': 'test_model',
        'parameters': {'n_estimators': 100},
        'accuracy': 0.95
    }
    
    # Сохранение артефакта
    version_manager.save_versioned_artifact(
        test_model, 
        "model", 
        "model_v1.0",
        metadata={'description': 'Test model for versioning system'}
    )
    
    # Загрузка артефакта
    loaded_model = version_manager.load_versioned_artifact("model", "model_v1.0")
    print(f"Загруженная модель: {loaded_model}")
    
    # Получение метаданных
    metadata = version_manager.get_artifact_metadata("model", "model_v1.0")
    print(f"Метаданные: {metadata}")
    
    # Список версий
    versions = version_manager.list_available_versions("model")
    print(f"Доступные версии моделей: {versions}")
    
    # Создание снепшота
    snapshot_id = version_manager.create_version_snapshot("Test snapshot")
    print(f"Снепшот создан: {snapshot_id}")
    
    # Список снепшотов
    snapshots = version_manager.list_snapshots()
    print(f"Снепшоты: {len(snapshots)}")
    
    logger.info("Система версионирования протестирована успешно")
