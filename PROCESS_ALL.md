# 📊 Обработка всех 58 PDF - Пошаговая инструкция

## ✅ Шаг 1: Обработка 13 тестовых PDF (УЖЕ ГОТОВО!)

```bash
python generate_simple.py
```

**Результат:** `submission.json` (13 PDF из data/test/)

---

## 🔄 Шаг 2: Обработка 45 основных PDF

```bash
python generate_full.py
```

**Что будет:**
- Обрабатывает по 1 PDF за раз
- Сохраняет прогресс в `submission_full_progress.json`
- Итоговый результат: `submission_full.json`

**Если процесс убьет:**
```bash
# Просто запусти снова - продолжит с того места!
python generate_full.py
```

**Займет:** ~30-60 минут (зависит от железа)

---

## 🔀 Шаг 3: Объединение результатов

Когда оба файла готовы:

```bash
python merge_results.py
```

**Результат:** `submission_final.json` (все 58 PDF)

---

## 📁 Итоговые файлы

После всех шагов у тебя будет:

```
📁 Промежуточные:
├── submission.json                  # 13 PDF из data/test/
├── submission_full.json             # 45 PDF из data/pdf/
├── submission_full_progress.json    # Прогресс обработки

📄 ФИНАЛЬНЫЙ:
└── submission_final.json            # ВСЕ 58 PDF ⭐
```

---

## 💡 Советы

### Если процесс убивается:

**Вариант 1:** Запусти снова - продолжит с того места
```bash
python generate_full.py
```

**Вариант 2:** Обрабатывай по частям
```bash
# Обработай первые 20 файлов, потом продолжи
# Прогресс автоматически сохраняется!
```

**Вариант 3:** Попробуй в Windows PowerShell
```powershell
cd \\wsl$\Ubuntu\home\loadcosmos\armeta_cv
python generate_full.py
```

---

## 🎯 Ожидаемый результат

```json
{
  "metadata": {
    "total_pdfs": 58,
    "total_pages": ~200-300,
    "total_detections": ~1000-2000,
    "detections_by_class": {
      "signature": ~400-600,
      "stamp": ~300-500,
      "qr": ~200-300
    }
  },
  "results": [
    {
      "pdf": "file1.pdf",
      "total_pages": 5,
      "summary": {
        "total_detections": 12,
        "by_class": {...}
      },
      "pages": [...]
    },
    ...58 PDF
  ]
}
```

---

## ⏱️ Примерное время

- **Шаг 1:** ✅ Готово (уже сделал)
- **Шаг 2:** ~30-60 минут
- **Шаг 3:** ~10 секунд

**Итого:** ~30-60 минут до полного результата

---

## 🆘 Проблемы?

**Ошибка "Killed":**
- Не хватает памяти
- Запусти снова - продолжит
- Или попробуй в Windows

**Ошибка "Model not found":**
```bash
# Проверь наличие модели
ls -lh best.pt

# Должен быть ~21-22 MB
```

**Другие ошибки:**
- Скрипт пропускает проблемные файлы
- Смотри детали в консоли
- Результат все равно сохранится

---

## 🚀 Начинай!

```bash
# Запускай второй шаг:
python generate_full.py

# Жди ~30-60 минут

# Потом объединяй:
python merge_results.py

# Готово! 🎉
```
