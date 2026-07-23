"""Regression tests for critical AR/validation behavior (audit fixes 1.9, 1.10, 1.11)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ar_system.kidney_ar_system import KidneyARSystem
from data_validation import DataValidator, ValidationLevel
from reliability.confidence_constraints import AnatomicalConstraints, FallbackHandler


@pytest.fixture
def patient_data() -> dict:
    return {
        "age": 45,
        "bmi": 24.5,
        "sex_encoded": 1,
        "kidney_left_center_x_mm": -45.2,
        "kidney_left_center_y_mm": 18.5,
        "kidney_left_center_z_mm": 95.3,
        "kidney_right_center_x_mm": 52.1,
        "kidney_right_center_y_mm": 19.8,
        "kidney_right_center_z_mm": 96.7,
    }


@pytest.fixture
def sensor_data() -> dict:
    return {
        "position": [10.0, 5.0, 0.0],
        "orientation": [0, 0, 0, 1],
        "tilt": 15.0,
        "rotation": 5.0,
    }


@pytest.fixture
def ar_system_data() -> dict:
    return {
        "world_to_ar_matrix": np.eye(4).tolist(),
        "scale_factor": 1.0,
    }


def test_ar_system_fails_closed_without_model(
    patient_data: dict,
    sensor_data: dict,
    ar_system_data: dict,
) -> None:
    """Without loaded ML artifacts AR must not return success=True (audit 1.9)."""
    system = KidneyARSystem()
    assert system.model_ready is False

    result = system.predict_kidney_displacement(patient_data, sensor_data, ar_system_data)
    assert result["success"] is False
    assert result["confidence"] == 0.0
    assert result["left_kidney"] is None
    assert result["right_kidney"] is None


def test_fallback_handler_treats_ml_output_as_displacement() -> None:
    """apply_constraints must not subtract original_pos from a delta vector (audit 1.10)."""
    body_limits = {
        "x_min": -150,
        "x_max": 150,
        "y_min": -100,
        "y_max": 100,
        "z_min": 50,
        "z_max": 150,
    }
    constraints = AnatomicalConstraints(body_limits, np.array([0.0, 0.0, 100.0]))
    handler = FallbackHandler(None, constraints)

    original = np.array([-50.0, 20.0, 95.0])
    ml_delta = np.array([5.0, -3.0, 2.0])

    constrained_delta = handler.handle_prediction(
        features=np.zeros(4),
        ml_prediction=ml_delta,
        confidence=0.9,
        original_position=original,
    )

    assert constrained_delta.shape == (3,)
    expected_pos = original + ml_delta
    assert np.allclose(constrained_delta + original, expected_pos, atol=1e-6)


def test_ar_constraint_path_uses_displacement_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """KidneyARSystem constraint path must call apply_constraints_from_displacement."""
    body_limits = {
        "x_min": -150,
        "x_max": 150,
        "y_min": -100,
        "y_max": 100,
        "z_min": 50,
        "z_max": 150,
    }
    constraints = AnatomicalConstraints(body_limits, np.array([0.0, 0.0, 100.0]))
    calls: list[tuple[np.ndarray, np.ndarray]] = []
    real = constraints.apply_constraints_from_displacement

    def _spy(original_pos: np.ndarray, displacement: np.ndarray) -> np.ndarray:
        calls.append((np.asarray(original_pos).copy(), np.asarray(displacement).copy()))
        return real(original_pos, displacement)

    monkeypatch.setattr(constraints, "apply_constraints_from_displacement", _spy)
    handler = FallbackHandler(None, constraints)
    system = KidneyARSystem()
    system.fallback_handler = handler
    system.constraints = constraints

    patient = {
        "kidney_left_center_x_mm": -50.0,
        "kidney_left_center_y_mm": 20.0,
        "kidney_left_center_z_mm": 95.0,
        "kidney_right_center_x_mm": 52.0,
        "kidney_right_center_y_mm": 19.0,
        "kidney_right_center_z_mm": 96.0,
    }
    prediction = np.array([5.0, -3.0, 2.0, -4.0, 1.0, 0.5])
    out = system._apply_constraints_and_fallback(
        features=np.zeros(4),
        prediction=prediction,
        confidence=0.9,
        patient_data=patient,
    )

    assert len(calls) == 2
    assert np.allclose(calls[0][1], prediction[:3])
    assert np.allclose(calls[1][1], prediction[3:])
    assert out.shape == (6,)


def test_validate_processed_data_flags_out_of_range_deltas() -> None:
    """Range check must use boolean mask, not sum of absolutes (audit 1.11)."""
    validator = DataValidator()
    X = np.zeros((3, 2))
    y = np.array(
        [
            [10.0, 5.0],
            [200.0, 0.0],
            [-5.0, 300.0],
        ]
    )
    target_names = ["kidney_left_delta_x", "kidney_right_delta_y"]

    results = validator.validate_processed_data(X, y, target_names=target_names)
    warnings = [r for r in results if r.level == ValidationLevel.WARNING]

    assert len(warnings) == 2
    assert any("kidney_left_delta_x" in r.message for r in warnings)
    assert any("kidney_right_delta_y" in r.message for r in warnings)
    assert warnings[0].value is not None
    assert len(warnings[0].value) >= 1
