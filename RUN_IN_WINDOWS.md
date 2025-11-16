# 🪟 Запуск в Windows (без WSL)

Если WSL2 убивает процессы из-за нехватки памяти, попробуй Windows PowerShell.

## Шаг 1: Установи Python в Windows

Скачай: https://www.python.org/downloads/
- Выбери "Add Python to PATH"
- Установи

## Шаг 2: Открой PowerShell в папке проекта

```powershell
# Перейди в папку (замени на свой путь)
cd C:\Users\YourName\armeta_cv

# Или через WSL путь:
cd \\wsl$\Ubuntu\home\loadcosmos\armeta_cv
```

## Шаг 3: Создай виртуальное окружение

```powershell
# Создай venv
python -m venv venv_win

# Активируй
.\venv_win\Scripts\Activate.ps1

# Если ошибка ExecutionPolicy:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Шаг 4: Установи зависимости

```powershell
pip install ultralytics opencv-python pillow pdf2image numpy tqdm

# Для pyzbar (опционально):
pip install pyzbar
```

**Важно для PDF:**
- Скачай poppler для Windows: https://github.com/oschwartz10612/poppler-windows/releases
- Распакуй в `C:\poppler`
- Добавь в PATH: `C:\poppler\Library\bin`

## Шаг 5: Запусти скрипт

```powershell
python generate_simple.py
```

---

## ⚡ Еще проще: Только 3 PDF

Если и это не работает, давай обработаем только 3 самых простых PDF:

```powershell
# Будет создан отдельный скрипт
python generate_minimal.py
```
