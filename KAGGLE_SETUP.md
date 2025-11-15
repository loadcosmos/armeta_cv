# 🚀 Kaggle Setup Guide

Полная инструкция по запуску в Kaggle

---

## 📋 Подготовка (5 минут)

### 1. Создать Kaggle Dataset

1. Зайти на https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Загрузить файлы:
   ```
   armeta_cv_data/
   ├── pdf/           (45 PDF файлов)
   └── annotations/   (JSON файлы)
   ```
4. Название: `armeta-cv-documents`
5. Click "Create"

### 2. Создать Kaggle Notebook

1. https://www.kaggle.com/code
2. Click "New Notebook"
3. Settings:
   - ✅ Accelerator: **GPU T4 x2**
   - ✅ Internet: **ON**
   - ✅ Persistence: **Files only**

---

## 🔧 Установка (CELL 1)

```python
# Установить зависимости
!pip install -q ultralytics opencv-python-headless pdf2image pillow scikit-learn

# Клонировать репозиторий
!git clone https://github.com/loadcosmos/armeta_cv.git
%cd armeta_cv

# Проверить GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

---

## 📦 Подготовка датасета (CELL 2)

```python
# Подготовить YOLO датасет из PDF + JSON
!python prepare_dataset.py \
  --pdf-dir /kaggle/input/armeta-cv-documents/pdf \
  --annotations-dir /kaggle/input/armeta-cv-documents/annotations \
  --output-dir /kaggle/working/data \
  --train-split 0.8 \
  --dpi 200

# Проверить результат
!ls -lh /kaggle/working/data/
!ls /kaggle/working/data/train/images/ | wc -l
!ls /kaggle/working/data/val/images/ | wc -l
```

**Ожидаемый вывод:**
```
📊 Train images: 103
📊 Val images: 26
📝 Total annotations: 258
```

---

## 🎯 Обучение модели (CELL 3)

```python
# Обновить путь к data.yaml в train скрипте
import os

# Запустить обучение
!python train_yolov8s_optimized.py

# Это займет ~40 минут
```

**Ожидаемые результаты:**
- mAP50: 0.60-0.65
- Stamp: 0.99+
- Signature: 0.55-0.60
- QR: 0.25-0.30

---

## 🧪 Тестирование гибридного подхода (CELL 4)

```python
# Загрузить модель
from ultralytics import YOLO
model = YOLO('/kaggle/working/runs/detect/train/weights/best.pt')

# Запустить hybrid test
!python colab_hybrid_test.py
```

---

## 💾 Сохранение результатов (CELL 5)

```python
# Скопировать важные файлы в output (автосохранение Kaggle)
import shutil
from pathlib import Path

# Создать output структуру
output_dir = Path('/kaggle/working/output')
output_dir.mkdir(exist_ok=True)

# Сохранить модель
shutil.copy(
    '/kaggle/working/runs/detect/train/weights/best.pt',
    output_dir / 'best.pt'
)

# Сохранить результаты
shutil.copy(
    '/kaggle/working/runs/detect/train/results.csv',
    output_dir / 'training_results.csv'
)

# Создать архив датасета (для повторного использования)
!zip -r /kaggle/working/output/yolo_dataset.zip /kaggle/working/data/

print("✅ Результаты сохранены в /kaggle/working/output/")
print("После завершения Notebook они будут доступны во вкладке Output")
```

---

## 📊 Валидация (CELL 6)

```python
# Запустить валидацию на лучшей модели
results = model.val(data='/kaggle/working/data/data.yaml')

# Вывести метрики
print(f"\n📊 Final Validation Metrics:")
print(f"mAP50: {results.box.map50:.3f}")
print(f"mAP50-95: {results.box.map:.3f}")
print(f"Precision: {results.box.mp:.3f}")
print(f"Recall: {results.box.mr:.3f}")

# Per-class метрики
class_names = ['signature', 'stamp', 'qr']
for i, name in enumerate(class_names):
    print(f"\n{name}:")
    print(f"  mAP50: {results.box.maps[i]:.3f}")
    print(f"  Precision: {results.box.p[i]:.3f}")
    print(f"  Recall: {results.box.r[i]:.3f}")
```

---

## 🔍 Inference тест (CELL 7)

```python
# Протестировать на валидационных изображениях
!python inference_optimized.py \
  --source /kaggle/working/data/val/images \
  --model /kaggle/working/runs/detect/train/weights/best.pt \
  --output /kaggle/working/output/val_results.json

# Посмотреть результаты
import json
with open('/kaggle/working/output/val_results.json', 'r') as f:
    results = json.load(f)
    print(json.dumps(results['stats'], indent=2))
```

---

## 📥 Скачивание результатов

После завершения всех ячеек:

1. Остановить Notebook (чтобы сохранились outputs)
2. Перейти во вкладку **Output** (справа)
3. Скачать:
   - `best.pt` (модель)
   - `training_results.csv` (метрики)
   - `yolo_dataset.zip` (датасет на будущее)
   - `val_results.json` (результаты inference)

---

## ⚡ Быстрый старт (All-in-One)

```python
# ========== CELL 1: Setup ==========
!pip install -q ultralytics opencv-python-headless pdf2image pillow scikit-learn
!git clone https://github.com/loadcosmos/armeta_cv.git
%cd armeta_cv

# ========== CELL 2: Prepare Dataset ==========
!python prepare_dataset.py \
  --pdf-dir /kaggle/input/armeta-cv-documents/pdf \
  --annotations-dir /kaggle/input/armeta-cv-documents/annotations \
  --output-dir /kaggle/working/data \
  --dpi 200

# ========== CELL 3: Train ==========
!python train_yolov8s_optimized.py

# ========== CELL 4: Test Hybrid ==========
from ultralytics import YOLO
model = YOLO('/kaggle/working/runs/detect/train/weights/best.pt')
!python colab_hybrid_test.py

# ========== CELL 5: Save Results ==========
import shutil
from pathlib import Path

output_dir = Path('/kaggle/working/output')
output_dir.mkdir(exist_ok=True)

shutil.copy('/kaggle/working/runs/detect/train/weights/best.pt', output_dir / 'best.pt')
shutil.copy('/kaggle/working/runs/detect/train/results.csv', output_dir / 'training_results.csv')

print("✅ Done! Check Output tab")
```

---

## 🐛 Troubleshooting

### Проблема: "No module named 'pdf2image'"
```python
!apt-get update
!apt-get install -y poppler-utils
!pip install pdf2image
```

### Проблема: "CUDA out of memory"
```python
# Уменьшить batch size в train_yolov8s_optimized.py
# Строка 30: 'batch': 2  (вместо 4)
```

### Проблема: "File not found: data.yaml"
```python
# Обновить путь в train_yolov8s_optimized.py
# Строка 25:
'data_yaml': '/kaggle/working/data/data.yaml'
```

---

## ⏱️ Время выполнения

| Шаг | Время |
|-----|-------|
| Setup | 2 мин |
| Dataset preparation | 15 мин |
| Training | 40 мин |
| Testing | 5 мин |
| **TOTAL** | **~1 час** |

---

## ✅ Чеклист

- [ ] Kaggle Dataset создан и загружен
- [ ] Kaggle Notebook создан (GPU + Internet ON)
- [ ] Зависимости установлены
- [ ] Датасет подготовлен (103 train / 26 val)
- [ ] Модель обучена (best.pt)
- [ ] Гибридный подход протестирован
- [ ] Результаты сохранены в Output
- [ ] best.pt скачан

---

**После этого переходим к добавлению killer features! 🚀**
