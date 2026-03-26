#!/usr/bin/env python3
"""Тест импортов API сервера"""

def test_api_imports():
    """Проверка всех импортов для API сервера"""
    try:
        print('🔍 Проверка импортов API сервера...')
        
        # Тест базовых импортов
        print('📦 Библиотеки:')
        try:
            import fastapi
            print('  ✅ fastapi')
        except ImportError as e:
            print(f'  ❌ fastapi: {e}')
            return False
            
        try:
            import uvicorn
            print('  ✅ uvicorn')
        except ImportError as e:
            print(f'  ❌ uvicorn: {e}')
            return False
            
        try:
            import pydantic
            print('  ✅ pydantic')
        except ImportError as e:
            print(f'  ❌ pydantic: {e}')
            return False
            
        try:
            import numpy as np
            print('  ✅ numpy')
        except ImportError as e:
            print(f'  ❌ numpy: {e}')
            return False
            
        try:
            import pandas as pd
            print('  ✅ pandas')
        except ImportError as e:
            print(f'  ❌ pandas: {e}')
            return False
            
        try:
            import joblib
            print('  ✅ joblib')
        except ImportError as e:
            print(f'  ❌ joblib: {e}')
            return False
        
        # Тест импортов проекта
        print('\n🏗️  Модули проекта:')
        import sys
        import os
        from pathlib import Path
        
        # Добавляем src в путь
        src_path = Path(__file__).parent / 'src'
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
            
        try:
            from api.api_server import app
            print('  ✅ api_server')
        except ImportError as e:
            print(f'  ❌ api_server: {e}')
            return False
            
        try:
            from ar_system.kidney_ar_system import KidneyARSystem
            print('  ✅ kidney_ar_system')
        except ImportError as e:
            print(f'  ❌ kidney_ar_system: {e}')
            return False
            
        print('\n🎉 Все импорты успешны!')
        return True
        
    except Exception as e:
        print(f'❌ Критическая ошибка импортов: {e}')
        return False

if __name__ == "__main__":
    success = test_api_imports()
    if success:
        print('\n✅ API сервер готов к запуску!')
    else:
        print('\n❌ Проблемы с зависимостями - нужно установить requirements.txt')
