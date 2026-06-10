"""
Feature engineering module for kidney displacement ML project.
"""

from .phase1_schema import (
    BASE_FEATURES,
    CROSS_FEATURES,
    ENGINEERED_FEATURES,
    SCHEMA_VERSION,
    TARGET_NAMES,
    align_to_feature_names,
    normalize_dataframe,
    normalize_record,
    validate_base_features,
)
from .pipeline import (
    apply_model_preprocessing,
    build_inference_matrix,
    normalize_raw_features,
    normalize_raw_record,
    predict_targets,
    print_canonical_flow,
)

__all__ = [
    "BASE_FEATURES",
    "ENGINEERED_FEATURES",
    "CROSS_FEATURES",
    "TARGET_NAMES",
    "SCHEMA_VERSION",
    "normalize_dataframe",
    "normalize_record",
    "validate_base_features",
    "align_to_feature_names",
    "normalize_raw_features",
    "normalize_raw_record",
    "build_inference_matrix",
    "apply_model_preprocessing",
    "predict_targets",
    "print_canonical_flow",
]
