"""System logger for kidney displacement prediction"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

class SystemLogger:
    """Системный логгер для AR системы"""
    
    def __init__(self, name: str = "kidney_ar_system", log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Очистка существующих handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (если указан)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(console_formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Логирование INFO уровня"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Логирование WARNING уровня"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Логирование ERROR уровня"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Логирование DEBUG уровня"""
        self.logger.debug(message)
