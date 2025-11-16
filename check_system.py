#!/usr/bin/env python3
"""
Проверка системы - все ли модули работают
"""

import sys
from pathlib import Path

def check_imports():
    """Проверка всех импортов"""
    print("=" * 60)
    print("ПРОВЕРКА ИМПОРТОВ")
    print("=" * 60)

    checks = []

    # 1. YOLO
    try:
        from ultralytics import YOLO
        print("✓ ultralytics (YOLO) - OK")
        checks.append(True)
    except Exception as e:
        print(f"✗ ultralytics (YOLO) - ОШИБКА: {e}")
        checks.append(False)

    # 2. OpenCV
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__} - OK")
        checks.append(True)
    except Exception as e:
        print(f"✗ OpenCV - ОШИБКА: {e}")
        checks.append(False)

    # 3. PDF2Image
    try:
        from pdf2image import convert_from_path
        print("✓ pdf2image - OK")
        checks.append(True)
    except Exception as e:
        print(f"✗ pdf2image - ОШИБКА: {e}")
        checks.append(False)

    # 4. QR Decoder (ВАЖНО!)
    try:
        from pyzbar import pyzbar
        print("✓ pyzbar (QR декодер) - OK")
        checks.append(True)
    except Exception as e:
        print(f"✗ pyzbar (QR декодер) - ОШИБКА: {e}")
        print("  Установите: sudo apt-get install libzbar0")
        checks.append(False)

    # 5. Streamlit
    try:
        import streamlit
        print(f"✓ Streamlit {streamlit.__version__} - OK")
        checks.append(True)
    except Exception as e:
        print(f"✗ Streamlit - ОШИБКА: {e}")
        checks.append(False)

    # 6. NumPy, Pandas
    try:
        import numpy as np
        import pandas as pd
        print(f"✓ NumPy {np.__version__}, Pandas {pd.__version__} - OK")
        checks.append(True)
    except Exception as e:
        print(f"✗ NumPy/Pandas - ОШИБКА: {e}")
        checks.append(False)

    return all(checks)

def check_project_files():
    """Проверка файлов проекта"""
    print("\n" + "=" * 60)
    print("ПРОВЕРКА ФАЙЛОВ ПРОЕКТА")
    print("=" * 60)

    files = {
        "Детекция": [
            "inference.py",
            "enhanced_inference.py",
        ],
        "Killer Features": [
            "qr_decoder.py",
            "validator.py",
            "html_reporter.py",
        ],
        "Приложение": [
            "mobile_app.py",
        ],
        "Обучение": [
            "prepare_dataset.py",
            "train_yolov8s_optimized.py",
        ],
        "Документация": [
            "README.md",
            "SETUP.md",
        ],
    }

    all_ok = True
    for category, file_list in files.items():
        print(f"\n{category}:")
        for file in file_list:
            if Path(file).exists():
                size = Path(file).stat().st_size
                print(f"  ✓ {file} ({size:,} bytes)")
            else:
                print(f"  ✗ {file} - ОТСУТСТВУЕТ!")
                all_ok = False

    return all_ok

def check_data_structure():
    """Проверка структуры данных"""
    print("\n" + "=" * 60)
    print("ПРОВЕРКА ДАННЫХ")
    print("=" * 60)

    data_dir = Path("data")
    if not data_dir.exists():
        print("✗ Папка data/ не найдена!")
        return False

    # Проверка тестовых PDF
    test_dir = data_dir / "test"
    if test_dir.exists():
        pdfs = list(test_dir.glob("*.pdf"))
        print(f"✓ Тестовые PDFs: {len(pdfs)} файлов")
        for pdf in pdfs[:3]:
            print(f"  - {pdf.name}")
        if len(pdfs) > 3:
            print(f"  ... и еще {len(pdfs) - 3}")
    else:
        print("✗ Папка data/test/ не найдена!")

    # Проверка модели
    model_paths = [
        "best.pt",
        "models/best.pt",
        "runs/detect/train2/weights/best.pt",
    ]

    model_found = False
    for path in model_paths:
        if Path(path).exists():
            size = Path(path).stat().st_size / (1024 * 1024)
            print(f"✓ Модель найдена: {path} ({size:.1f} MB)")
            model_found = True
            break

    if not model_found:
        print("⚠ Модель best.pt не найдена!")
        print("  Скачайте с Kaggle и положите в корень проекта")
        return False

    return True

def check_modules():
    """Проверка модулей проекта"""
    print("\n" + "=" * 60)
    print("ПРОВЕРКА МОДУЛЕЙ ПРОЕКТА")
    print("=" * 60)

    checks = []

    # 1. QR Decoder
    try:
        from qr_decoder import QRDecoder
        decoder = QRDecoder()
        print("✓ QRDecoder - OK (готов к использованию)")
        checks.append(True)
    except Exception as e:
        print(f"✗ QRDecoder - ОШИБКА: {e}")
        checks.append(False)

    # 2. Validator
    try:
        from validator import DocumentValidator, ContractValidator, LicenseValidator
        validator = ContractValidator()
        print("✓ DocumentValidator - OK (готов к использованию)")
        checks.append(True)
    except Exception as e:
        print(f"✗ DocumentValidator - ОШИБКА: {e}")
        checks.append(False)

    # 3. HTML Reporter
    try:
        from html_reporter import HTMLReporter
        reporter = HTMLReporter()
        print("✓ HTMLReporter - OK (готов к использованию)")
        checks.append(True)
    except Exception as e:
        print(f"✗ HTMLReporter - ОШИБКА: {e}")
        checks.append(False)

    # 4. Inference
    try:
        from inference import YOLODocumentDetector
        print("✓ YOLODocumentDetector - OK")
        checks.append(True)
    except Exception as e:
        print(f"✗ YOLODocumentDetector - ОШИБКА: {e}")
        checks.append(False)

    return all(checks)

def main():
    print("\n" + "=" * 60)
    print("ПРОВЕРКА СИСТЕМЫ ARMETA CV")
    print("=" * 60)
    print()

    results = []

    # 1. Импорты
    results.append(("Импорты библиотек", check_imports()))

    # 2. Файлы проекта
    results.append(("Файлы проекта", check_project_files()))

    # 3. Данные
    results.append(("Структура данных", check_data_structure()))

    # 4. Модули
    results.append(("Модули проекта", check_modules()))

    # Итоги
    print("\n" + "=" * 60)
    print("ИТОГИ ПРОВЕРКИ")
    print("=" * 60)

    for name, status in results:
        status_str = "✓ OK" if status else "✗ ОШИБКА"
        print(f"{name:.<40} {status_str}")

    all_ok = all(status for _, status in results)

    print("\n" + "=" * 60)
    if all_ok:
        print("✓ СИСТЕМА ГОТОВА К РАБОТЕ!")
        print("=" * 60)
        print("\nСледующие шаги:")
        print("1. Убедитесь, что модель best.pt скачана с Kaggle")
        print("2. Запустите: python enhanced_inference.py --input data/test/ТЗ-2.pdf --html")
        print("3. Или запустите: streamlit run mobile_app.py")
        return 0
    else:
        print("✗ ЕСТЬ ПРОБЛЕМЫ!")
        print("=" * 60)
        print("\nУстраните ошибки выше и запустите проверку снова")
        return 1

if __name__ == "__main__":
    sys.exit(main())
