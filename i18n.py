"""
Система локализации (i18n) для roo_bench.py
Поддержка Ukrainian (ua) и English (en)
"""

# Словари переводов
TRANSLATIONS = {
    "en": {
        "app_title": "Roo Code Model Benchmark",
        "scanning_models": "Scanning models and gathering data from ollama.com...",
        "available_models": "Available models:",
        "model_list_header": "[{index}] {name:<25} | {params:<5} | Size: {size_gb:4.1f} GB | MaxCtx: {max_ctx_str:>4} | Vision: {vision} | Tools: {tools} | Think: {thinking}",
        "select_models": "Enter model numbers separated by commas (or 'all'): ",
        "select_models_interactive": "Use ↑/↓ arrows to navigate, Space to select/deselect, Enter to confirm, 'a' to select all, 'Esc' to select none",
        "invalid_input": "❌ Invalid input!",
        "no_models_found": "❌ No models found locally (or filtered): ",
        "no_test_models": "❌ No available models for testing.",
        "repeated_run_header": "💡 FOR REPEATED RUN USE:",
        "testing_model": "🚀 TESTING: {model_name}",
        "model_size": "   (Size: {size_gb:.1f} GB | Declared Max Context: {max_ctx_str})",
        "warming_up": "  -> Warming up with {ctx} token context...",
        "benchmark_failed": "     ❌ Failure: {error_msg}",
        "stopping_tests": "     ⏭️  Stopping tests for '{model_name}'. Larger contexts will definitely fail.",
        "speed_status_normal": "     Speed: {tps:.2f} t/s | VRAM used: {vram} MB | Status: ✅ NORMAL",
        "speed_status_gpu": "     Speed: {tps:.2f} t/s | VRAM used: {vram} MB | Status: ⚡ FLYING (GPU)",
        "speed_status_ram": "     Speed: {tps:.2f} t/s | VRAM used: {vram} MB | Status: 🐌 SLOW (RAM/CPU)",
        "recommendations_header": "=== RECOMMENDATIONS FOR ROO CODE SETUP (TOP 3 OPTIONS) ===",
        "architect_mode": "🏗️  'Architect' Mode (Priority: Memory volume for project analysis)",
        "code_mode": "💻 'Code' Mode (Priority: Fast code generation)",
        "debug_mode": "🐛 'Debug' Mode (Priority: Balance of context and speed)",
        "variant": "    Option {i}: Context = {ctx:<6} | Speed: {tps:.1f} t/s",
        "no_successful_runs": "Model: {model_name} - ❌ No successful runs",
        "restart_ollama": "🔄 Clearing memory (restarting Ollama)...",
        "cli_help": "Roo Code Model Benchmark",
        "cli_models": "List of models separated by comma",
        "cli_capabilities": "Capabilities filter: v (Vision), T (Tools), t (Think). Example: --capabilities vT or -f vT",
        "cli_lang": "Language (en or ua)",
        "cli_restart_method": "Restart method: systemctl, docker, kill_start, manual",
        "cli_no_restart": "Disable Ollama restart",
        "cli_num_runs": "Number of benchmark runs (default: 3)",
        "restart_ollama_disabled": "Ollama restart disabled (--no-restart)",
        "error_unknown_restart_method": "Unknown restart method: {method}",
        "error_restart_command": "Restart command error '{cmd}': {stderr}",
        "restart_success": "Ollama restarted successfully",
        "error_restart_command_not_found": "Restart command not found: {cmd}",
        "error_restart_permission": "Permission error when restarting Ollama",
        "error_restart_unknown": "Unknown error during restart: {error}",
        "cli_context_sizes": "Context sizes to test (comma-separated, e.g., 8192,16384,32768)",
        "cli_context_sizes_auto": "Auto-select context sizes (geometric progression)",
        "cli_output": "Output file path",
        "cli_output_format": "Output format: json or csv (default: none)",
        "error_ollama_connection": "Failed to connect to local Ollama: {error}",
        "error_ollama_api": "Error from Ollama API: {error_msg}",
        "error_parsing_response": "Response parsing failure ({json_err}). Raw response: {raw_text}",
        "error_timeout": "Timeout: model was generating response too long (more than 300 seconds).",
        "error_crash": "Crash/OOM: Ollama process unexpectedly terminated (RAM/VRAM shortage).",
        "error_unknown": "Unknown error: {error_details}",
        "filter_applied": "🔍 Capabilities filter applied: {capabilities}",
        "no_models_match_filter": "❌ No models match the specified filters.",
        "skipping_no_contexts": "  ⏭️  Skipping: Could not determine valid context sizes for testing.",
        "output_json": "Results saved to {output_file} (JSON)",
        "output_csv": "Results saved to {output_file} (CSV)",
        "no_output": "No output file specified.",
        "avg_tps": "Average TPS: {avg_tps:.2f}",
        "min_tps": "Min TPS: {min_tps:.2f}",
        "max_tps": "Max TPS: {max_tps:.2f}",
        "std_dev": "Std Dev: {std_dev:.2f}",
        "vram_none": "VRAM: N/A (no GPU)",
        "vram_used": "VRAM: {vram} MB",
        "language": "Language",
        "model": "Model",
        "context": "Context",
        "tps": "TPS",
        "timestamp": "Timestamp",
    },
    "ua": {
        "app_title": "Roo Code Model Benchmark",
        "scanning_models": "Сканирование моделей і збір даних з ollama.com...",
        "available_models": "Доступні моделі:",
        "model_list_header": "[{index}] {name:<25} | {params:<5} | Розмір: {size_gb:4.1f} GB | MaxCtx: {max_ctx_str:>4} | Vision: {vision} | Tools: {tools} | Think: {thinking}",
        "select_models": "Введіть номери моделей через запятую (або 'all'): ",
        "select_models_interactive": "Використовуйте ↑/↓ для навігації, Пробіл для виділення/зняття, Enter для підтвердження, 'a' виділити все, 'Esc' зняти все",
        "invalid_input": "❌ Невірний ввід!",
        "no_models_found": "⚠️  Моделі не знайдено локально (або відфільтровані): ",
        "no_test_models": "❌ Немає доступних моделей для тестування.",
        "repeated_run_header": "💡 ДЛЯ ПОВТОРНОГО ЗАПУСКУ ІСПОЛЬЗУЙТЕ:",
        "testing_model": "🚀 ТЕСТИРОВАНИЕ: {model_name}",
        "model_size": "   (Розмір: {size_gb:.1f} GB | Заявлений Max Context: {max_ctx_str})",
        "warming_up": "  -> Прогрев з контекстом {ctx} токенів...",
        "benchmark_failed": "     ❌ Сбой: {error_msg}",
        "stopping_tests": "     ⏭️  Остановка тестів для '{model_name}'. Бóльші контексти гарантовано упадуть.",
        "speed_status_normal": "     Скорість: {tps:.2f} t/s | VRAM занято: {vram} МБ | Статус: ✅ НОРМА",
        "speed_status_gpu": "     Скорість: {tps:.2f} t/s | VRAM занято: {vram} МБ | Статус: ⚡ ЛЕТАЄ (GPU)",
        "speed_status_ram": "     Скорість: {tps:.2f} t/s | VRAM занято: {vram} МБ | Статус: 🐌 ТОРМОЗИТЬ (RAM/CPU)",
        "recommendations_header": "=== РЕКОМЕНДАЦІЇ ПО НАСТРОЙКЕ ROO CODE (ТОП-3 ВАРІАНТИ) ===",
        "architect_mode": "🏗️  Режим 'Architect' (Пріоритет: Обсяг пам'яті для аналізу проекту)",
        "code_mode": "💻 Режим 'Code' (Пріоритет: Швидка генерація коду)",
        "debug_mode": "🐛 Режим 'Debug' (Пріоритет: Баланс контексту і швидкості)",
        "variant": "    Варіант {i}: Context = {ctx:<6} | Скорість: {tps:.1f} t/s",
        "no_successful_runs": "Модель: {model_name} - ❌ Немає успішних прогонів",
        "restart_ollama": "🔄 Очистка пам'яті (перезапуск Ollama)...",
        "cli_help": "Roo Code Model Benchmark",
        "cli_models": "Список моделей через запятую",
        "cli_capabilities": "Фільтр можливостей: v (Vision), T (Tools), t (Think). Приклад: --capabilities vT або -f vT",
        "cli_lang": "Мова (en або ua)",
        "cli_restart_method": "Метод перезапуску: systemctl, docker, kill_start, manual",
        "cli_no_restart": "Вимкнути перезапуск Ollama",
        "restart_ollama_disabled": "Перезапуск Ollama відключено (--no-restart)",
        "error_unknown_restart_method": "Невідомий метод перезапуску: {method}",
        "error_restart_command": "Помилка команди перезапуску '{cmd}': {stderr}",
        "restart_success": "Ollama успішно перезапущено",
        "error_restart_command_not_found": "Команда перезапуску не знайдена: {cmd}",
        "error_restart_permission": "Помилка дозволів при перезапуску Ollama",
        "error_restart_unknown": "Невідома помилка при перезапуску: {error}",
        "cli_num_runs": "Кількість запусків бенчмарку (за замовчуванням: 3)",
        "cli_context_sizes": "Розміри контексту для тесту (через кому, наприклад, 8192,16384,32768)",
        "cli_context_sizes_auto": "Автоматичний вибір розмірів контексту (геометрична прогресія)",
        "cli_output": "Шлях до файлу виводу",
        "cli_output_format": "Формат виводу: json або csv (за замовчуванням: немає)",
        "error_ollama_connection": "Помилка підключення до локальної Ollama: {error}",
        "error_ollama_api": "Помилка від Ollama API: {error_msg}",
        "error_parsing_response": "Сбой парсингу відповіді ({json_err}). Сырой відповідь: {raw_text}",
        "error_timeout": "Timeout: модель генерувала відповідь занадто довго (більше 300 секунд).",
        "error_crash": "Crash/OOM: Процес Ollama несподівано завершився (нехватка RAM/VRAM).",
        "error_unknown": "Невідома помилка: {error_details}",
        "filter_applied": "🔍 Фільтр можливостей застосовано: {capabilities}",
        "no_models_match_filter": "❌ Ні одна модель не відповідає заданим фільтрам.",
        "skipping_no_contexts": "  ⏭️  Пропуск: Не вдалося визначити валідні розміри контексту для тесту.",
        "output_json": "Результати збережено в {output_file} (JSON)",
        "output_csv": "Результати збережено в {output_file} (CSV)",
        "no_output": "Файл виводу не вказано.",
        "avg_tps": "Середнє TPS: {avg_tps:.2f}",
        "min_tps": "Мін TPS: {min_tps:.2f}",
        "max_tps": "Макс TPS: {max_tps:.2f}",
        "std_dev": "Std Dev: {std_dev:.2f}",
        "vram_none": "VRAM: N/A (немає GPU)",
        "vram_used": "VRAM: {vram} МБ",
        "language": "Мова",
        "model": "Модель",
        "context": "Контекст",
        "tps": "TPS",
        "timestamp": "Час",
    }
}

# Доступні мови
AVAILABLE_LANGUAGES = ["en", "ua"]


def get_available_languages():
    """Повертає список доступних мов."""
    return AVAILABLE_LANGUAGES


def set_language(lang):
    """
    Встановлює поточну мову.
    
    Args:
        lang: Код мови ('en' або 'ua')
    
    Returns:
        bool: True якщо мова встановлена, False інакше
    """
    if lang in AVAILABLE_LANGUAGES:
        global _current_language
        _current_language = lang
        return True
    return False


def get_text(key, *args, **kwargs):
    """
    Отримує текст за ключем з поточної мови.
    
    Args:
        key: Ключ для отримання тексту
        *args: Додаткові аргументи для форматирования
        **kwargs: Додаткові іменовані аргументи для форматирования
    
    Returns:
        str: Текст або ключ якщо мова не знайдена
    """
    global _current_language
    
    if _current_language is None:
        _current_language = "en"
    
    if _current_language not in TRANSLATIONS:
        _current_language = "en"
    
    translations = TRANSLATIONS[_current_language]
    
    if key in translations:
        text = translations[key]
        # Форматуємо текст з аргументами
        try:
            return text.format(*args, **kwargs)
        except (KeyError, IndexError, AttributeError):
            return text
    return key


# Ініціалізація поточної мови
_current_language = None