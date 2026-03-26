import sys
from pathlib import Path

# Добавляем src в Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

# Теперь импортируем наши модули
from geometry.kidney_model import KidneyGeometryModel, create_personal_kidney_model, get_fallback_model
from coordinate_system.patient_coords import PatientCoordinateSystem, MultiLevelTransformer
from preprocessing.unified_pipeline import UnifiedPreprocessingPipeline
from reliability.confidence_constraints import (
    ConfidenceEstimator, AnatomicalConstraints, 
    FallbackHandler, TemporalSmoother
)

# Остальной код остается без изменений...
