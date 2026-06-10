
"""
Kidney Displacement Prediction Package.

Lightweight distribution wrapper: configs + version metadata.
Implementation code lives in the parent repository under ``src/``.
"""

from pathlib import Path

__version__ = "1.0.0"
__author__ = "ML Team"
__description__ = "Kidney displacement prediction using ensemble models"


def get_config_dir() -> Path:
    """Return path to packaged YAML configuration files."""
    return Path(__file__).resolve().parent / "config"
