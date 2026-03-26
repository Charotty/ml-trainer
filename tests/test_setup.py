import sys
from pathlib import Path

# Добавляем src в Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Теперь импортируем наши модули
from ar_system.kidney_ar_system import KidneyARSystem
from validation.data_validator import DataValidator, ClinicalMetrics, SystemLogger
from versioning.version_manager import VersionManager
from unpaired.unpaired_trainer import UnpairedDataProcessor, EnhancedModelTrainer

# Остальной код остается без изменений...
