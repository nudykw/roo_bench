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
        "model_list_header": "[{index}] {name:<25} | {params:<5} | Size: {size_gb:4.1f} GB | MoE: {moe_str:>8} | MaxCtx: {max_ctx_str:>4} | Vision: {vision} | Tools: {tools} | Think: {thinking}",
        "select_models": "Enter model numbers separated by commas (or 'all'): ",
        "select_models_interactive": "Use ↑/↓ arrows to navigate, Space to select/deselect, Enter to confirm, 'a' to select all, 'Esc' to select none",
        "select_model_single": "Space: select model | Enter: confirm | Esc: cancel",
        "invalid_input": "❌ Invalid input!",
        "no_models_found": "❌ No models found locally (or filtered): ",
        "no_test_models": "❌ No available models for testing.",
        "repeated_run_header": "💡 FOR REPEATED RUN USE:",
        "testing_model": "🚀 TESTING: {model_name}",
        "model_size": "   (Size: {size_gb} GB | Declared Max Context: {max_ctx_str})",
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
        "current_num_ctx": "   📊 Current num_ctx: {num_ctx}",
        "actual_num_ctx": "   📊 Actual num_ctx (from /api/ps): {num_ctx}",
        "ctx_info_only": "   Setting context to {expected}",
        "ctx_verified": "   ✅ Verified num_ctx: {actual} == {expected}",
        "ctx_mismatch": "   ⚠️  num_ctx mismatch: actual={actual}, expected={expected}",
        "ctx_fix_attempt": "   🔄 Attempting to fix num_ctx via Modelfile...",
        "ctx_fix_success": "   ✅ num_ctx fixed to {ctx}",
        "ctx_fix_failed": "   ❌ Failed to fix num_ctx",
        "ctx_skip_warning": "   ⚠️  Skipping benchmark for ctx={ctx} due to num_ctx mismatch",
        "ctx_show_info": "   🔍 via /api/show: Found context values: {values}",
        "ctx_show_match": "   ✅ /api/show confirms context >= {expected} (found {actual})",
        "ctx_show_too_small": "   ❌ /api/show context too small: {actual} < {expected}",
        "ctx_gen_test": "   🧪 Generation test: prompt_eval_count={eval_count}, estimated_tokens={tokens}",
        "ctx_gen_verified": "   ✅ Generation test passed — context window verified",
        "ctx_gen_failed": "   ⚠️  Generation test: not all tokens evaluated",
        "actual_n_ctx_during_gen": "   📊 Actual n_ctx during generation: max={max_ctx}, avg={avg_ctx:.0f}",
        "ctx_verified_during_gen": "   ✅ Context verified: n_ctx={max_ctx} == {expected}",
        "ctx_mismatch_during_gen": "   ⚠️  Context mismatch: actual={max_ctx}, expected={expected}",
        "actual_n_ctx_after_gen": "   📊 Actual n_ctx after generation: {ctx}",
        "ctx_verified_after_gen": "   ✅ Context verified: {actual} == {expected}",
        "ctx_mismatch_after_gen": "   ⚠️  Context mismatch: actual={actual}, expected={expected}",
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
        "error_restart_timeout": "SSH connection timeout",
        "error_ssh_no_host": "SSH host is not specified",
        "restart_ssh_exec": "Executing via SSH: {cmd}",
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
        "output_file_exists": "⚠️  File '{output_file}' already exists (size: {file_size}).",
        "ask_overwrite": "Overwrite existing file",
        "save_cancelled": "Save cancelled.",
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
        "temperature": "Temperature",
        "duration": "Time",
        "prompt_tokens": "Input",
        "response_tokens": "Output",
        "tps": "TPS",
        "timestamp": "Timestamp",
        "benchmark_runs_header": "=== BENCHMARK RUNS ===",
        "benchmark_summary": "=== BENCHMARK SUMMARY ===",
        "result_row": "  Context: {ctx} | Avg TPS: {avg_tps:.2f} | Min: {min_tps:.2f} | Max: {max_tps:.2f} | StdDev: {std_dev:.2f} | VRAM: {vram}",
        "results_header": "=== RESULTS FOR {model_name} ===",
        "benchmark_interrupted": "Benchmark interrupted by user.",
        "ask_save_results": "Would you like to save the results to a file?",
        "ask_save_filename": "Enter filename (default: benchmark_results.json): ",
        "save_filename_default": "benchmark_results.json",
        "ask_analyze_ai": "Would you like to send results for AI analysis?",
        "ask_select_model": "Select a model for analysis (number or name): ",
        "no_models_for_analysis": "No models available for analysis.",
        "analysis_sending": "Sending request to {model_name}...",
        "analysis_response": "=== AI ANALYSIS FROM {model_name} ===",
        "analysis_raw_response": "=== RAW RESPONSE ===",
        "analysis_translated": "=== TRANSLATED RESPONSE ===",
        "translation_unavailable": "Translation unavailable, showing raw response.",
        "analysis_complete": "Analysis complete.",
        "analysis_error": "Analysis failed: {error}",
        "select_analysis_model": "Use ↑/↓ arrows to navigate, Enter to select",
        "no_output_file_specified": "No output file was specified — results were not saved.",
        "cache_using": "✅ Using cached model metadata",
        "cache_fetching": "📡 Fetching model metadata from Ollama...",
        "cache_update_start": "Updating full model metadata cache...",
        "cache_update_complete": "✅ Cache update complete! Updated {count} models.",
        "analyze_file_not_found": "❌ File not found: {file_path}",
        "analyze_file_unknown_format": "❌ Unknown file format: {ext}. Use .json or .csv",
        "analyze_file_empty": "❌ File is empty.",
        "analyze_file_parse_error": "❌ Error parsing file: {error}",
        "analysis_model_not_found": "❌ Model '{model_name}' not found in available models.",
        "cli_analyze_file": "Analyze benchmark results from a saved JSON/CSV file",
        "cli_analysis_model": "Model name to use for analysis (used with --analyze-file)",
        "using_independent_prompts_default": "📝 Using independent prompts from prompts.md by default",
        "ask_enable_expert": "Enable Expert-Evaluator for automatic response quality assessment?",
        "expert_model_selection": "Select an expert model for response evaluation",
        "cli_temperature": "Temperature values to test (comma-separated, e.g., 0.0,0.7,1.0)",
        "error_invalid_temperature": "❌ Invalid temperature values '{values}'. Use comma-separated values between 0.0 and 2.0.",
        "cli_independent_top": "Limit the number of independent prompts per mode (e.g., --independent-top 1 runs only the first prompt per mode)",
        "benchmark_params_header": "=== BENCHMARK PARAMETERS ===",
        "param_temperatures": "Temperatures: {temperatures}",
        "param_context_sizes": "Context sizes: {context_sizes}",
        "param_prompts_file": "Prompts file: {prompts_file}",
        "param_models": "Models to test: {models}",
        "param_test_models": "Models for testing: {test_models}",
        "param_expert_model": "Expert model: {expert_model}",
    },
    "ua": {
        "app_title": "Roo Code Model Benchmark",
        "scanning_models": "Сканування моделей та збір даних з ollama.com...",
        "available_models": "Доступні моделі:",
        "model_list_header": "[{index}] {name:<25} | {params:<5} | Розмір: {size_gb:4.1f} ГБ | MoE: {moe_str:>8} | MaxCtx: {max_ctx_str:>4} | Vision: {vision} | Tools: {tools} | Think: {thinking}",
        "select_models": "Введіть номери моделей через кому (або 'all'): ",
        "select_models_interactive": "Використовуйте ↑/↓ для навігації, Пробіл для виділення/зняття, Enter для підтвердження, 'a' виділити все, 'Esc' зняти все",
        "select_model_single": "Пробіл: виділити модель | Enter: підтвердити | Esc: скасувати",
        "invalid_input": "❌ Невірний ввід!",
        "no_models_found": "❌ Моделі не знайдено локально (або відфільтровано): ",
        "no_test_models": "❌ Немає доступних моделей для тестування.",
        "repeated_run_header": "💡 ДЛЯ ПОВТОРНОГО ЗАПУСКУ ВИКОРИСТОВУЙТЕ:",
        "testing_model": "🚀 ТЕСТУВАННЯ: {model_name}",
        "model_size": "   (Розмір: {size_gb} ГБ | Заявлений Max Context: {max_ctx_str})",
        "warming_up": "  -> Прогрів з контекстом {ctx} токенів...",
        "benchmark_failed": "     ❌ Помилка: {error_msg}",
        "stopping_tests": "     ⏭️  Зупинка тестів для '{model_name}'. Більші контексти гарантовано падуть.",
        "speed_status_normal": "     Швидкість: {tps:.2f} t/s | VRAM використано: {vram} МБ | Статус: ✅ НОРМА",
        "speed_status_gpu": "     Швидкість: {tps:.2f} t/s | VRAM використано: {vram} МБ | Статус: ⚡ ЛЕТИТЬ (GPU)",
        "speed_status_ram": "     Швидкість: {tps:.2f} t/s | VRAM використано: {vram} МБ | Статус: 🐌 ГАЛІТЬ (RAM/CPU)",
        "recommendations_header": "=== РЕКОМЕНДАЦІЇ ДЛЯ НАЛАШТУВАННЯ ROO CODE (ТОП-3 ВАРІАНТИ) ===",
        "architect_mode": "🏗️  Режим 'Architect' (Пріоритет: Об'єм пам'яті для аналізу проекту)",
        "code_mode": "💻 Режим 'Code' (Пріоритет: Швидка генерація коду)",
        "debug_mode": "🐛 Режим 'Debug' (Пріоритет: Баланс контексту та швидкості)",
        "variant": "    Варіант {i}: Context = {ctx:<6} | Швидкість: {tps:.1f} t/s",
        "no_successful_runs": "Модель: {model_name} - ❌ Немає успішних запусків",
        "restart_ollama": "🔄 Очищення пам'яті (перезапуск Ollama)...",
        "current_num_ctx": "   📊 Поточний num_ctx: {num_ctx}",
        "actual_num_ctx": "   📊 Фактичний num_ctx (з /api/ps): {num_ctx}",
        "ctx_info_only": "   Встановлюємо контекст {expected}",
        "ctx_verified": "   ✅ Перевірено num_ctx: {actual} == {expected}",
        "ctx_mismatch": "   ⚠️  Неузгодженість num_ctx: фактично={actual}, потрібно={expected}",
        "ctx_fix_attempt": "   🔄 Спроба виправити num_ctx через Modelfile...",
        "ctx_fix_success": "   ✅ num_ctx виправлено на {ctx}",
        "ctx_fix_failed": "   ❌ Не вдалося виправити num_ctx",
        "ctx_skip_warning": "   ⚠️  Пропуск бенчмарку для ctx={ctx} через неузгодженість num_ctx",
        "ctx_show_info": "   🔍 через /api/show: Знайдено значення контексту: {values}",
        "ctx_show_match": "   ✅ /api/show підтверджує контекст >= {expected} (знайдено {actual})",
        "ctx_show_too_small": "   ❌ /api/show контекст замалий: {actual} < {expected}",
        "ctx_gen_test": "   🧪 Тест генерації: prompt_eval_count={eval_count}, estimated_tokens={tokens}",
        "ctx_gen_verified": "   ✅ Тест генерації пройдено — контекстне вікно підтверджено",
        "ctx_gen_failed": "   ⚠️  Тест генерації: не всі токени оброблено",
        "actual_n_ctx_during_gen": "   📊 Фактичний n_ctx під час генерації: max={max_ctx}, avg={avg_ctx:.0f}",
        "ctx_verified_during_gen": "   ✅ Контекст підтверджено: n_ctx={max_ctx} == {expected}",
        "ctx_mismatch_during_gen": "   ⚠️  Неузгодженість контексту: фактично={max_ctx}, потрібно={expected}",
        "actual_n_ctx_after_gen": "   📊 Фактичний n_ctx після генерації: {ctx}",
        "ctx_verified_after_gen": "   ✅ Контекст підтверджено: {actual} == {expected}",
        "ctx_mismatch_after_gen": "   ⚠️  Неузгодженість контексту: фактично={actual}, потрібно={expected}",
        "cli_help": "Roo Code Model Benchmark",
        "cli_models": "Список моделей через кому",
        "cli_capabilities": "Фільтр можливостей: v (Vision), T (Tools), t (Think). Приклад: --capabilities vT або -f vT",
        "cli_lang": "Мова (en або ua)",
        "cli_restart_method": "Метод перезапуску: systemctl, docker, kill_start, manual",
        "cli_no_restart": "Вимкнути перезапуск Ollama",
        "cli_num_runs": "Кількість запусків бенчмарку (за замовчуванням: 3)",
        "restart_ollama_disabled": "Перезапуск Ollama вимкнено (--no-restart)",
        "error_unknown_restart_method": "Невідомий метод перезапуску: {method}",
        "error_restart_command": "Помилка команди перезапуску '{cmd}': {stderr}",
        "restart_success": "Ollama успішно перезапущено",
        "error_restart_command_not_found": "Команду перезапуску не знайдено: {cmd}",
        "error_restart_permission": "Помилка дозволів під час перезапуску Ollama",
        "error_restart_unknown": "Невідома помилка під час перезапуску: {error}",
        "error_restart_timeout": "Тайм-аут підключення SSH",
        "error_ssh_no_host": "Хост SSH не вказано",
        "restart_ssh_exec": "Виконання через SSH: {cmd}",
        "cli_context_sizes": "Розміри контексту для тесту (через кому, наприклад, 8192,16384,32768)",
        "cli_context_sizes_auto": "Автоматичний вибір розмірів контексту (геометрична прогресія)",
        "cli_output": "Шлях до файлу виводу",
        "cli_output_format": "Формат виводу: json або csv (за замовчуванням: немає)",
        "error_ollama_connection": "Не вдалося підключитися до локальної Ollama: {error}",
        "error_ollama_api": "Помилка від Ollama API: {error_msg}",
        "error_parsing_response": "Помилка парсингу відповіді ({json_err}). Сире повідомлення: {raw_text}",
        "error_timeout": "Timeout: модель генерувала відповідь занадто довго (більше 300 секунд).",
        "error_crash": "Crash/OOM: Процес Ollama несподівано завершився (не вистачає RAM/VRAM).",
        "error_unknown": "Невідома помилка: {error_details}",
        "filter_applied": "🔍 Фільтр можливостей застосовано: {capabilities}",
        "no_models_match_filter": "❌ Жодна модель не відповідає вказанним фільтрам.",
        "skipping_no_contexts": "  ⏭️  Пропуск: Не вдалося визначити валідні розміри контексту для тесту.",
        "output_json": "Результати збережено до {output_file} (JSON)",
        "output_csv": "Результати збережено до {output_file} (CSV)",
        "output_file_exists": "⚠️  Файл '{output_file}' вже існує (розмір: {file_size}).",
        "ask_overwrite": "Перезаписати існуючий файл",
        "save_cancelled": "Збереження скасовано.",
        "no_output": "Файл виводу не вказано.",
        "avg_tps": "Середнє TPS: {avg_tps:.2f}",
        "min_tps": "Мін TPS: {min_tps:.2f}",
        "max_tps": "Макс TPS: {max_tps:.2f}",
        "std_dev": "СтД: {std_dev:.2f}",
        "vram_none": "VRAM: Н/З (немає GPU)",
        "vram_used": "VRAM: {vram} МБ",
        "language": "Мова",
        "model": "Модель",
        "context": "Контекст",
        "temperature": "Температура",
        "duration": "Час",
        "prompt_tokens": "Вхідні",
        "response_tokens": "Вихідні",
        "tps": "TPS",
        "timestamp": "Час",
        "benchmark_runs_header": "=== ЗАПУСКИ БЕНЧМАРКУ ===",
        "benchmark_summary": "=== ПІДСУМОК БЕНЧМАРКУ ===",
        "result_row": "  Контекст: {ctx} | Середнє TPS: {avg_tps:.2f} | Мін: {min_tps:.2f} | Макс: {max_tps:.2f} | СтД: {std_dev:.2f} | VRAM: {vram}",
        "results_header": "=== РЕЗУЛЬТАТИ ДЛЯ {model_name} ===",
        "benchmark_interrupted": "Тестування перервано користувачем.",
        "ask_save_results": "Бажаєте зберегти результати у файл?",
        "ask_save_filename": "Введіть ім'я файлу (за замовчуванням: benchmark_results.json): ",
        "save_filename_default": "benchmark_results.json",
        "ask_analyze_ai": "Бажаєте надіслати результати для AI аналізу?",
        "ask_select_model": "Оберіть модель для аналізу (номер або назва): ",
        "no_models_for_analysis": "Немає доступних моделей для аналізу.",
        "analysis_sending": "Відправка запиту до {model_name}...",
        "analysis_response": "=== AI АНАЛІЗ ВІД {model_name} ===",
        "analysis_raw_response": "=== СИРА ВІДПОВІДЬ ===",
        "analysis_translated": "=== ПЕРЕКЛАДЕНА ВІДПОВІДЬ ===",
        "translation_unavailable": "Переклад недоступний, показано сирій відповідь.",
        "analysis_complete": "Аналіз завершено.",
        "analysis_error": "Помилка аналізу: {error}",
        "select_analysis_model": "Використовуйте ↑/↓ для навігації, Enter для вибору",
        "no_output_file_specified": "Файл виводу не вказано — результати не збережено.",
        "cache_using": "✅ Використовується закешована метаданих моделей",
        "cache_fetching": "📡 Отримання метаданих моделей з Ollama...",
        "cache_update_start": "Оновлення повного кешу метаданих моделей...",
        "cache_update_complete": "✅ Кеш оновлено! Оновлено {count} моделей.",
        "analyze_file_not_found": "❌ Файл не знайдено: {file_path}",
        "analyze_file_unknown_format": "❌ Невідомий формат файлу: {ext}. Використовуйте .json або .csv",
        "analyze_file_empty": "❌ Файл порожній.",
        "analyze_file_parse_error": "❌ Помилка парсингу файлу: {error}",
        "analysis_model_not_found": "❌ Модель '{model_name}' не знайдено в доступних моделях.",
        "cli_analyze_file": "Аналізувати результати бенчмарку з збереженого JSON/CSV файлу",
        "cli_analysis_model": "Назва моделі для аналізу (використовується з --analyze-file)",
        "using_independent_prompts_default": "📝 Використовуються незалежні промти з prompts.md за замовчуванням",
        "cli_no_thinking": "Вимкнути режим міркувань для запобігання зацикленням (за замовчуванням: увімкнено)",
        "cli_thinking": "Увімкнути режим міркувань на моделях, що його підтримують",
        "ask_enable_expert": "Увімкнути Експерт-оцінювач для автоматичної оцінки якості відповідей?",
        "expert_model_selection": "Оберіть модель-експерт для оцінки відповідей",
        "cli_temperature": "Значення температури для тестування (через кому, наприклад, 0.0,0.7,1.0)",
        "error_invalid_temperature": "❌ Невірні значення температури '{values}'. Використовуйте значення через кому від 0.0 до 2.0.",
        "cli_independent_top": "Обмежити кількість незалежних промтів на режим (напр., --independent-top 1 запускає лише перший промт на режим)",
        "benchmark_params_header": "=== ПАРАМЕТРИ ТЕСТУВАННЯ ===",
        "param_temperatures": "Температури: {temperatures}",
        "param_context_sizes": "Розміри контексту: {context_sizes}",
        "param_prompts_file": "Файл промтів: {prompts_file}",
        "param_models": "Моделі для тестування: {models}",
        "param_test_models": "Моделі для тестування: {test_models}",
        "param_expert_model": "Модель експерта: {expert_model}",
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