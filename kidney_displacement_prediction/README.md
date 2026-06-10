# Kidney Displacement Prediction (distribution package)

Thin installable wrapper for configuration artifacts used by the main
repository implementation under `src/` and `models/`.

## Contents

- `config/*.yaml` — model, feature, and deployment defaults
- `requirements.txt` — minimal runtime dependencies for this package

## Install

From the repository root:

```bash
pip install ./kidney_displacement_prediction
```

## Note

Core training/inference code lives in the parent repository (`src/`, `models/`).
This package is intentionally lightweight and provides versioned config access
via `kidney_displacement_prediction.get_config_dir()`.
