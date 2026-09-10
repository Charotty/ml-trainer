#!/usr/bin/env python3
"""Restore the missing categorical encoder on the current clinical_honest.pkl.

Does not retrain heads. ``FoldCategoricalEncoder.fit`` only records dummy
column names; reconstructing it from ``feature_names`` restores the serve
contract that ``train_clinical_honest.py`` used to drop on dump.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.phase1_schema import SCHEMA_VERSION  # noqa: E402
from src.models.ensemble.serializer import (  # noqa: E402
    RUNTIME_PAYLOAD_SCHEMA_VERSION,
    estimator_type_from_payload,
    preprocessing_contract_from_payload,
)
from src.models.runtime import (  # noqa: E402
    default_model_path,
    reconstruct_categorical_encoder,
    validate_runtime_payload,
)


def repair_payload(payload: dict) -> dict:
    out = dict(payload)
    names = list(out.get("feature_names") or [])
    encoder = out.get("categorical_encoder")
    if encoder is None or not bool(getattr(encoder, "fitted_", False)):
        out["categorical_encoder"] = reconstruct_categorical_encoder(names)
    out["encode_categoricals"] = True
    if out.get("z_driver_names") is None:
        out["z_driver_names"] = []
    else:
        out["z_driver_names"] = list(out["z_driver_names"])
    out.setdefault("artifact_schema_version", RUNTIME_PAYLOAD_SCHEMA_VERSION)
    out.setdefault("feature_schema_version", SCHEMA_VERSION)
    out.setdefault("estimator_type", estimator_type_from_payload(out))
    if not isinstance(out.get("preprocessing_contract"), dict):
        out["preprocessing_contract"] = preprocessing_contract_from_payload(out)
    validate_runtime_payload(out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    path = args.model or default_model_path()
    payload = joblib.load(path)
    repaired = repair_payload(payload)
    encoder = repaired["categorical_encoder"]
    print(f"model     : {path}")
    print(f"encoder   : fitted={encoder.fitted_} dummies={encoder.dummy_names}")
    print(f"missing   : {encoder.missing_names}")
    print(f"schema    : {repaired.get('artifact_schema_version')} / {repaired.get('feature_schema_version')}")
    if args.dry_run:
        print("dry-run: not writing")
        return 0
    joblib.dump(repaired, path)
    print(f"[OK] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
