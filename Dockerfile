# CT Workbench — CPU image (API, UI, model predict, PDF).
# For TotalSegmentator GPU extraction use profile `gpu` in docker-compose.yml.

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/models/adaptive_ensemble_clinical_honest.pkl \
    CASES_ROOT=/data/cases \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        dcm2niix \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt /app/requirements-docker.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements-docker.txt

# Application code (keep layers cache-friendly: deps first, then source)
COPY config /app/config
COPY frontend/public /app/frontend/public
COPY models/phase1 /app/models/phase1
COPY scripts/inference /app/scripts/inference
COPY scripts/validation/common.py /app/scripts/validation/common.py
COPY src /app/src

# Production model is gitignored locally; compose mounts it, or COPY if present at build.
# Placeholder dir so MODEL_PATH parent always exists.
RUN mkdir -p /app/models /data/cases

EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8010/health || exit 1

CMD ["python", "-m", "uvicorn", "src.api.ct_workbench_api:app", \
     "--host", "0.0.0.0", "--port", "8010", "--workers", "1"]
