---
trigger: manual
---
# GLOBAL RULES FOR AR KIDNEY DISPLACEMENT ML PROJECT

## Project Context
You are working on a medical ML project for predicting kidney displacement during laparoscopic surgery. The system uses machine learning to predict how the kidney moves when a patient changes position from supine (lying on back) to lateral (lying on side).

## Technology Stack
- Python 3.10+
- ML: scikit-learn, xgboost, pandas, numpy
- API: FastAPI, uvicorn, pydantic
- Visualization: matplotlib, seaborn, plotly
- Medical: pydicom, nibabel (optional)

## Code Style Guidelines

### General
- Use English for all code, comments, and documentation
- Follow PEP 8 style guide strictly
- Maximum line length: 100 characters
- Use type hints for all functions
- Write docstrings for all functions and classes (Google style)

### Naming Conventions
- Variables: snake_case (e.g., patient_data, kidney_position)
- Functions: snake_case (e.g., calculate_mae, predict_displacement)
- Classes: PascalCase (e.g., KidneyPredictor, TrocarPlanner)
- Constants: UPPER_SNAKE_CASE (e.g., MAX_ERROR_THRESHOLD, FEATURE_COLS)
- Private methods: _leading_underscore (e.g., _validate_input)

### File Organization
```
project/
├── data/
│   ├── raw/              # Original CSV/Excel files
│   ├── processed/        # Clean train/val/test datasets
│   └── README.md         # Data documentation
├── models/
│   ├── production/       # Final models for deployment
│   ├── experiments/      # Development models
│   └── README.md
├── notebooks/            # Jupyter notebooks for EDA
├── src/
│   ├── data/            # Data processing scripts
│   ├── models/          # ML model code
│   ├── features/        # Feature engineering
│   ├── visualization/   # Plotting utilities
│   └── utils/           # Helper functions
├── backend/             # FastAPI application
├── tests/               # Unit and integration tests
├── configs/             # Configuration files
└── docs/                # Documentation
```

## Critical Requirements

### Data Handling
- NEVER hardcode file paths - use Path objects from pathlib
- ALWAYS check for NaN values before training
- ALWAYS use try-except for file operations
- Save all intermediate results (don't lose work)
- Use descriptive variable names for medical data

### ML Model Development
- ALWAYS split data before any processing (avoid data leakage)
- ALWAYS save scaler/preprocessor separately
- Use random_state=42 for reproducibility
- Log all metrics (don't just print)
- Save model metadata (version, date, metrics, parameters)

### Medical Domain
- Coordinates are in millimeters (mm)
- Use anatomically correct terminology:
  - supine = lying on back
  - lateral = lying on side
  - upper_third, middle_third, lower_third of kidney
  - delta = displacement/change
- Target accuracy: MAE < 10 mm
- Success rate: > 85% of predictions within 10mm

### API Development
- Use Pydantic models for ALL input/output validation
- ALWAYS add error handling (try-except)
- Return proper HTTP status codes
- Log all requests and errors
- Add CORS middleware for cross-origin requests

### Testing
- Write unit tests for all critical functions
- Test edge cases (NaN, negative values, extreme values)
- Use pytest fixtures for test data
- Aim for >80% code coverage

## Error Messages
- Use clear, actionable error messages
- Include context (what failed, why, what to do)
- Example: "Failed to load model from {path}. File not found. Please run training first."

## Comments and Documentation
- Write comments for WHY, not WHAT
- Document medical assumptions and constraints
- Add references to medical literature where applicable
- Keep docstrings up to date

## Performance
- Prefer vectorized operations (numpy/pandas) over loops
- Use appropriate data types (float32 vs float64)
- Profile slow code sections
- Consider memory usage for large datasets

## Version Control
- Commit often with descriptive messages
- Use conventional commit format:
  - feat: new feature
  - fix: bug fix
  - docs: documentation
  - refactor: code restructuring
  - test: adding tests
  - chore: maintenance

## Dependencies
- Pin all dependency versions in requirements.txt
- Separate dev dependencies (requirements-dev.txt)
- Document why each dependency is needed

## Prohibited Practices
- ❌ NO hardcoded credentials or API keys
- ❌ NO print() for debugging in production code (use logging)
- ❌ NO training models without validation set
- ❌ NO single-letter variable names (except i, j, k in loops)
- ❌ NO ignoring warnings (fix or suppress explicitly)
