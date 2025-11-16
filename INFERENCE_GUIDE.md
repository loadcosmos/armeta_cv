# 🚀 Inference Guide

Production-ready inference для детекции объектов в документах.

## Быстрый старт

### 1. Установка

```bash
pip install ultralytics opencv-python pdf2image pillow tqdm
```

### 2. Использование

#### Один PDF файл:
```bash
python inference.py --input document.pdf --output results/
```

#### С визуализацией:
```bash
python inference.py --input document.pdf --output results/ --visualize
```

#### Batch обработка папки:
```bash
python inference.py --input data/test/ --output results/ --batch --visualize
```

#### С настройками:
```bash
python inference.py \
  --input document.pdf \
  --output results/ \
  --model runs/detect/train2/weights/best.pt \
  --conf 0.2 \
  --iou 0.3 \
  --visualize
```

## Параметры

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--input, -i` | PDF файл или папка | Required |
| `--output, -o` | Папка для результатов | `inference_results` |
| `--model, -m` | Путь к модели | `runs/detect/train2/weights/best.pt` |
| `--visualize, -v` | Сохранить визуализацию | False |
| `--batch, -b` | Обработать папку | False |
| `--conf` | Порог уверенности | 0.25 |
| `--iou` | NMS IoU порог | 0.45 |
| `--dpi` | DPI для PDF | 200 |

## Формат вывода

### JSON результаты (`results.json`):
```json
{
  "pdf": "document.pdf",
  "timestamp": "2024-11-16T...",
  "total_pages": 5,
  "summary": {
    "total_detections": 12,
    "by_class": {
      "signature": 5,
      "stamp": 4,
      "qr": 3
    }
  },
  "pages": [
    {
      "page": 1,
      "image_size": {"width": 3306, "height": 4678},
      "detections": [
        {
          "class": "signature",
          "confidence": 0.95,
          "bbox": {
            "x1": 100, "y1": 200,
            "x2": 300, "y2": 350,
            "width": 200, "height": 150
          }
        }
      ],
      "summary": {"signature": 1}
    }
  ]
}
```

### Структура выходных файлов:
```
inference_results/
├── document_name/
│   ├── results.json          # Детекции в JSON
│   ├── page_1.jpg           # Визуализация (если --visualize)
│   └── page_2.jpg
└── batch_summary.json        # Общая статистика (если --batch)
```

## Примеры использования

### Python API:
```python
from inference import DocumentDetector
import cv2

# Инициализация
detector = DocumentDetector(
    model_path='best.pt',
    conf_threshold=0.25,
    iou_threshold=0.45
)

# Детекция на изображении
image = cv2.imread('document.jpg')
detections = detector.detect_image(image)

# Визуализация
img_vis = detector.visualize_detections(image, detections)
cv2.imwrite('result.jpg', img_vis)

# Детекция на PDF
results = detector.detect_pdf('document.pdf')
print(results['summary'])
```

### Интеграция в веб-сервис:
```python
from flask import Flask, request, jsonify
from inference import DocumentDetector

app = Flask(__name__)
detector = DocumentDetector()

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['pdf']
    file.save('temp.pdf')

    results = detector.detect_pdf('temp.pdf')
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Настройки для разных сценариев

### Высокая точность (меньше false positives):
```bash
python inference.py --input doc.pdf --conf 0.4 --iou 0.5
```

### Высокий recall (найти всё):
```bash
python inference.py --input doc.pdf --conf 0.15 --iou 0.3
```

### Перекрывающиеся объекты:
```bash
python inference.py --input doc.pdf --conf 0.2 --iou 0.3 --visualize
```

## Performance

- **Single PDF (10 pages)**: ~15-20 секунд
- **Batch (50 PDFs)**: ~8-12 минут
- **GPU**: 3-5x ускорение

## Troubleshooting

### Model not found:
```bash
# Укажите правильный путь к модели
python inference.py --input doc.pdf --model path/to/best.pt
```

### Low quality results:
```bash
# Увеличьте DPI
python inference.py --input doc.pdf --dpi 300
```

### Out of memory:
```bash
# Уменьшите DPI
python inference.py --input doc.pdf --dpi 150
```

## Метрики модели

| Класс | mAP50 | Precision | Recall |
|-------|-------|-----------|--------|
| **QR** | 99.5% | 98.0% | 100% |
| **Stamp** | 87.0% | 88.6% | 91.7% |
| **Signature** | 77.6% | 92.7% | 68.4% |
| **Overall** | 88.1% | 93.1% | 86.7% |

## Поддержка

Для вопросов и багов: создайте issue в репозитории
