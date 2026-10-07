"""Clinical report contract (disclaimer text). Visualization must not import the API."""

from __future__ import annotations

DISCLAIMER = (
    "Исследовательский инструмент для планирования доступа. "
    "Загруженная модель — кандидат, не production-победитель. "
    "Не заменяет клинический протокол, осмотр и решение лечащего врача."
)

REPORT_SCHEMA_VERSION = "ct_workbench_report_v1"

__all__ = ["DISCLAIMER", "REPORT_SCHEMA_VERSION"]
