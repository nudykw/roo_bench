# 🚀 Roo Bench — Context & VRAM Analyzer

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Professional benchmarking tool for Ollama models with multi-language support (EN/UA)**

---

## 🌐 Documentation / Документація

| Language / Мова | Description |
|-----------------|-------------|
| 🇬🇧 [README_EN.md](README_EN.md) | Full documentation in English |
| 🇺🇦 [README_UA.md](README_UA.md) | Повна документація українською мовою |

---

## Quick Start / Швидкий старт

```bash
# Clone the repository / Клонувати репозиторій
git clone https://github.com/nudykw/roo_bench.git
cd roo_bench

# Create and activate virtual environment / Створити та активувати віртуальне середовище
python -m venv venv
source venv/bin/activate

# Install dependencies / Встановити залежності
pip install -r requirements.txt

# Run benchmark / Запустити бенчмарк
./venv/bin/python main.py
```

For detailed instructions, see:
- **English:** [README_EN.md](README_EN.md)
- **Українська:** [README_UA.md](README_UA.md)

---

## 📊 Codegraph Intelligence / Інтелект Codegraph

**Codegraph** — інструмент для аналізу кодової бази з використанням AI. Індекс автоматично оновлюється при кожному коммите.

### Ручна синхронізація

```bash
# Провірити стан індексу
make codegraph-status

# Інкрементальна синхронізація
make codegraph-sync

# Повна переіндексація
make codegraph-index
```

### Автоматична синхронізація

Pre-commit hook автоматично синхронізиує індекс при кожному комміті Python-файлів:

```
🔄 Синхронізація codegraph індексу...
synced 0 files (skipped 63), nodes=0 edges=0
✅ Codegraph індекс оновлений
```

> **Примітка:** Якщо синхронізація не вдалася, комміт всеред буде виконаний (non-fatal error).

### VS Code Watch Mode

Для автоматичного моніторингу змін у реальному часі:

```bash
# Вимагає inotify-tools: sudo pacman -S inotify-tools
./scripts/codegraph-watch.sh
```

**Через VS Code Tasks:**
1. **Ctrl+Shift+P** → **Tasks: Run Task** → **Codegraph Watch (inotify)**
2. Watch запуститься у фоновому режимі
3. **Ctrl+Shift+P** → **Tasks: Show Running Tasks** → для зупинки

> **Примітка:** inotify-tools вимагає Linux. Для macOS/Windows використовуйте pre-commit hook або ручну синхронізацію.
