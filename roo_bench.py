import requests
import subprocess
import time
import urllib.parse
import argparse
import sys
import os
import json
import csv
from datetime import datetime
from enum import Enum
from bs4 import BeautifulSoup
from i18n import get_text, set_language, get_available_languages

OLLAMA_URL = "http://localhost:11434"
CONTEXT_SIZES = [8192, 16384, 32768, 65536, 131072, 262144]

def get_context_sizes(args):
    """Получение списка контекстов из аргументов CLI.
    
    Args:
        args: Пarsed arguments from argparse
        
    Returns:
        list: Список размеров контекста
    """
    # Если указан --context-sizes-auto, генерируем геометрическую прогрессию
    if args.context_sizes_auto:
        sizes = []
        current = 8192
        while current <= 262144:
            sizes.append(current)
            current *= 2
        return sizes
    
    # Если указан --context-sizes, парсим из строки
    if args.context_sizes:
        try:
            sizes = [int(x.strip()) for x in args.context_sizes.split(',')]
            if all(s > 0 for s in sizes):
                return sorted(sizes)
            else:
                print(get_text("error_invalid_context_size", sizes=args.context_sizes))
                return CONTEXT_SIZES
        except ValueError:
            print(get_text("error_invalid_context_size", sizes=args.context_sizes))
            return CONTEXT_SIZES
    
    # Иначе используем дефолтные
    return CONTEXT_SIZES


class RestartMethod(Enum):
    """Методы перезапуска Ollama"""
    SYSTEMCTL = "systemctl"
    DOCKER = "docker"
    KILL_START = "kill_start"
    MANUAL = "manual"


def check_gpu_available():
    """Проверка наличия GPU.
    
    Returns:
        bool: True если GPU доступен, False иначе
    """
    # Проверка наличия nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            return True
    except Exception:
        pass
    
    # Проверка наличия /proc/driver/nvidia/gpus/
    try:
        import os
        return os.path.exists('/proc/driver/nvidia/gpus/')
    except Exception:
        pass
    
    return False


def get_vram_usage():
    """Получение использования VRAM с fallback.
    
    Returns:
        int или None: Использование VRAM в байтах, или None если GPU недоступен
    """
    if not check_gpu_available():
        return None
    
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
    except Exception:
        pass
    
    # Fallback: чтение из /proc/driver/nvidia/gpus/0/mem_used
    try:
        import os
        gpu_path = '/proc/driver/nvidia/gpus/0/mem_used'
        if os.path.exists(gpu_path):
            with open(gpu_path, 'r') as f:
                content = f.read().strip()
                # Формат: "used: 4500MiB" или "4500 MiB"
                if ':' in content:
                    value = content.split(':')[1].strip()
                else:
                    value = content
                # Парсинг значения с MiB/GiB
                value = value.strip()
                if value.endswith('GiB'):
                    return int(value[:-3]) * 1024 * 1024 * 1024
                elif value.endswith('MiB'):
                    return int(value[:-3]) * 1024 * 1024
                elif value.endswith('GB'):
                    return int(value[:-2]) * 1024 * 1024 * 1024
                elif value.endswith('MB'):
                    return int(value[:-2]) * 1024 * 1024
                else:
                    return int(value)
    except Exception:
        pass
    
    return None

def restart_ollama(method: RestartMethod = RestartMethod.SYSTEMCTL, no_restart: bool = False):
    """Перезапуск Ollama с указанным методом.
    
    Args:
        method: Метод перезапуска (SYSTEMCTL, DOCKER, KILL_START, MANUAL)
        no_restart: Если True, перезапуск не выполняется
    """
    if no_restart:
        print(get_text("restart_ollama_disabled"))
        return
    
    print(get_text("restart_ollama"))
    
    try:
        if method == RestartMethod.SYSTEMCTL:
            cmd = ['sudo', 'systemctl', 'restart', 'ollama']
        elif method == RestartMethod.DOCKER:
            cmd = ['docker', 'restart', 'ollama']
        elif method == RestartMethod.KILL_START:
            subprocess.run(['sudo', 'systemctl', 'stop', 'ollama'], check=False)
            time.sleep(1)
            cmd = ['sudo', 'systemctl', 'start', 'ollama']
        elif method == RestartMethod.MANUAL:
            cmd = ['ollama', 'restart']
        else:
            print(get_text("error_unknown_restart_method", method=method.value))
            return
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(get_text("error_restart_command", cmd=' '.join(cmd), stderr=result.stderr))
        else:
            print(get_text("restart_success"))
        
        time.sleep(4)
    except FileNotFoundError:
        fallback_cmd = ['ollama', 'restart'] if method == RestartMethod.MANUAL else ['systemctl', 'restart', 'ollama']
        print(get_text("error_restart_command_not_found", cmd=' '.join(fallback_cmd)))
    except PermissionError:
        print(get_text("error_restart_permission"))
    except Exception as e:
        print(get_text("error_restart_unknown", error=str(e)))

def get_capabilities_from_ollama_site(model_name):
    """Получение capabilities модели из Ollama API или HTML парсинга.
    
    Args:
        model_name: Имя модели (например, "llama3.2" или "dev-qwen2")
        
    Returns:
        tuple: (vision, tools, thinking) - статусы возможностей
    """
    base_name = model_name.split(':')[0]
    if base_name.startswith('dev-'):
        base_name = base_name[4:]

    # Попытка 1: Использование Ollama API
    try:
        api_url = f"https://ollama.com/api/library/{base_name}"
        response = requests.get(api_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        
        if response.status_code == 200:
            # API возвращает JSON с информацией о модели
            data = response.json()
            model_info = data.get("model", {})
            
            # Ищем capabilities в JSON ответе
            # API может возвращать capabilities в разных форматах
            capabilities = model_info.get("capabilities", {})
            
            vision = "✅" if capabilities.get("vision") else "❌"
            tools = "✅" if capabilities.get("tools") else "❌"
            thinking = "✅" if capabilities.get("thinking") else "❌"
            
            if vision != "❌" or tools != "❌" or thinking != "❌":
                return vision, tools, thinking
            
            # Если capabilities нет в JSON, пробуем поискать в других полях
            model_str = str(model_info).lower()
            if "vision" in model_str:
                vision = "✅"
            if "tools" in model_str or "tool use" in model_str:
                tools = "✅"
            if "thinking" in model_str or "reasoning" in model_str:
                thinking = "✅"
            
            if vision != "❓" or tools != "❓" or thinking != "❓":
                return vision, tools, thinking
        
        # Если API вернул ошибку или не содержит нужной информации, пробуем HTML парсинг
        if response.status_code != 200:
            # Пробуем поиск через search API
            search_url = f"https://ollama.com/api/search?q={urllib.parse.quote(base_name)}"
            search_resp = requests.get(search_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            if search_resp.status_code == 200:
                search_data = search_resp.json()
                for item in search_data.get("results", []):
                    if item.get("name") == base_name:
                        # Получаем capabilities из search results
                        caps = item.get("capabilities", {})
                        vision = "✅" if caps.get("vision") else "❌"
                        tools = "✅" if caps.get("tools") else "❌"
                        thinking = "✅" if caps.get("thinking") else "❌"
                        return vision, tools, thinking
                return "❓", "❓", "❌"
    
    except requests.exceptions.Timeout:
        print(f"⚠️  Timeout при запросе к Ollama API для {base_name}, используем HTML парсинг")
    except requests.exceptions.ConnectionError:
        print(f"⚠️  ConnectionError при запросе к Ollama API для {base_name}, используем HTML парсинг")
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code in [404, 500]:
            print(f"⚠️  HTTP {e.response.status_code} от Ollama API для {base_name}, используем HTML парсинг")
        else:
            raise
    except Exception as e:
        print(f"⚠️  Ошибка при запросе к Ollama API для {base_name}: {e}, используем HTML парсинг")

    # Fallback: HTML парсинг с более надёжными селекторами
    try:
        url = f"https://ollama.com/library/{base_name}"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        
        if response.status_code != 200:
            # Пробуем поиск через HTML
            search_url = f"https://ollama.com/search?q={urllib.parse.quote(base_name)}"
            search_resp = requests.get(search_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            if search_resp.status_code == 200:
                soup = BeautifulSoup(search_resp.text, 'html.parser')
                link = soup.find('a', href=lambda href: href and '/library/' in href)
                if link:
                    url = "https://ollama.com" + link['href']
                    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Более надёжный парсинг: ищем конкретные элементы
            # Ищем capabilities section или description с ключевыми словами
            
            # Метод 1: Поиск через data-атрибуты или специфичные классы
            capabilities_section = soup.find('div', class_=lambda c: c and 'capabilities' in c.lower())
            if capabilities_section:
                caps_text = capabilities_section.get_text().lower()
                vision = "✅" if "vision" in caps_text else "❌"
                tools = "✅" if "tools" in caps_text or "tool use" in caps_text else "❌"
                thinking = "✅" if "thinking" in caps_text or "reasoning" in caps_text else "❌"
                return vision, tools, thinking
            
            # Метод 2: Поиск через description/summary
            description = soup.find('div', class_='description') or soup.find('div', class_='summary')
            if description:
                desc_text = description.get_text().lower()
                vision = "✅" if "vision" in desc_text or "multimodal" in desc_text else "❌"
                tools = "✅" if "tools" in desc_text or "tool use" in desc_text else "❌"
                thinking = "✅" if "thinking" in desc_text or "reasoning" in desc_text else "❌"
                return vision, tools, thinking
            
            # Метод 3: Поиск через JSON-LD (если есть)
            json_ld = soup.find('script', type='application/ld+json')
            if json_ld:
                try:
                    import json
                    ld_data = json.loads(json_ld.string)
                    if isinstance(ld_data, dict):
                        capabilities = ld_data.get('capability', {})
                        if isinstance(capabilities, dict):
                            vision = "✅" if capabilities.get('vision') else "❌"
                            tools = "✅" if capabilities.get('tools') else "❌"
                            thinking = "✅" if capabilities.get('thinking') else "❌"
                            return vision, tools, thinking
                except json.JSONDecodeError:
                    pass
            
            # Метод 4: Поиск через alt-текст изображений (иконки capabilities)
            vision_icon = soup.find('img', alt=lambda alt: alt and 'vision' in alt.lower())
            tools_icon = soup.find('img', alt=lambda alt: alt and 'tools' in alt.lower())
            thinking_icon = soup.find('img', alt=lambda alt: alt and 'thinking' in alt.lower() or 'reasoning' in alt.lower())
            
            vision = "✅" if vision_icon else "❌"
            tools = "✅" if tools_icon else "❌"
            thinking = "✅" if thinking_icon else "❌"
            
            if vision != "❓" or tools != "❓" or thinking != "❓":
                return vision, tools, thinking
            
            # Метод 5: Финальный fallback - поиск в тексте страницы
            page_text = soup.get_text().lower()
            vision = "✅" if "vision" in page_text else "❌"
            tools = "✅" if "tools" in page_text or "tool use" in page_text else "❌"
            thinking = "✅" if "thinking" in page_text or "reasoning" in page_text or "deepseek" in base_name.lower() else "❌"
            
            return vision, tools, thinking
            
    except requests.exceptions.Timeout:
        print(f"⚠️  Timeout при HTML парсинге для {base_name}")
    except requests.exceptions.ConnectionError:
        print(f"⚠️  ConnectionError при HTML парсинге для {base_name}")
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code in [404, 500]:
            print(f"⚠️  HTTP {e.response.status_code} при HTML парсинге для {base_name}")
        else:
            raise
    except Exception as e:
        print(f"⚠️  Ошибка при HTML парсинге для {base_name}: {e}")

    return "❓", "❓", "❓"

def get_models():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        models = []
        for m in response.json().get("models",[]):
            details = m.get("details", {})
            size_gb = m.get("size", 0) / (1024 ** 3)
            
            # Получаем максимальный контекст модели через API show
            max_ctx = 32768  # Значение по умолчанию, если не удалось найти
            show_error = None
            try:
                show_resp = requests.post(f"{OLLAMA_URL}/api/show", json={"name": m["name"]})
                if show_resp.status_code == 200:
                    model_info = show_resp.json().get("model_info", {})
                    # Ищем ключ, содержащий 'context_length' (например 'llama.context_length' или 'qwen2.context_length')
                    for key, val in model_info.items():
                        if 'context_length' in key:
                            max_ctx = int(val)
                            break
            except Exception as e:
                show_error = f"Ошибка при получении max_ctx для {m['name']}: {e}"
                print(f"⚠️  {show_error}")

            models.append({
                "name": m["name"],
                "params": details.get("parameter_size", "N/A"),
                "quant": details.get("quantization_level", "N/A"),
                "size_gb": size_gb,
                "max_ctx": max_ctx
            })
        return models
    except Exception as e:
        print(get_text("error_ollama_connection", error=str(e)))
        return []

import math

def run_benchmark(model_name, context_size, num_runs=3):
    """Run benchmark for a model with multiple runs for averaging.
    
    Args:
        model_name: Model name
        context_size: Context size
        num_runs: Number of runs for averaging (default: 3)
        
    Returns:
        tuple: (avg_tps, vram, tps_list, error)
            avg_tps: Average TPS (float)
            vram: VRAM usage in bytes (int or None if GPU unavailable)
            tps_list: List of results for each run (list of dict)
            error: Error message (str or None)
    """
    prompt = "Write a comprehensive Python script that implements a multithreaded web server. Explain every line in extreme detail."
    
    tps_list = []
    vram = None
    error = None
    
    for run_num in range(num_runs):
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 100,
                "num_ctx": context_size
            }
        }

        response = None
        
        try:
            response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=300)
            
            if response.status_code != 200:
                try:
                    err_msg = response.json().get("error", f"HTTP {response.status_code}")
                except:
                    err_msg = f"HTTP {response.status_code}: {response.text[:100]}"
                error = get_text("error_ollama_api", error_msg=err_msg)
                break
                
            try:
                data = response.json()
            except Exception as json_err:
                raw_text = response.text[:200].replace('\n', ' ')
                error = get_text("error_parsing_response", json_err=json_err, raw_text=raw_text)
                break
                
            vram_after = get_vram_usage()
            total_duration = data.get("total_duration", 0) / 1e9
            eval_count = data.get("eval_count", 0)
            tps = eval_count / total_duration if total_duration > 0 else 0
            tps_list.append({"run": run_num + 1, "tps": tps, "vram": vram_after})
            
        except requests.exceptions.Timeout:
            error = get_text("error_timeout")
            break
        except requests.exceptions.ConnectionError:
            error = get_text("error_crash")
            break
        except Exception as e:
            err_details = get_text("error_unknown", error_details=str(e))
            if response is not None:
                err_details += f" | Сырой ответ: {response.text[:200]}"
            error = err_details
            break
    
    # Вычисляем среднее TPS
    if tps_list:
        avg_tps = sum(r['tps'] for r in tps_list) / len(tps_list)
        # Вычисляем стандартное отклонение
        if len(tps_list) > 1:
            mean = avg_tps
            variance = sum((r['tps'] - mean) ** 2 for r in tps_list) / len(tps_list)
            std_dev = math.sqrt(variance)
        else:
            std_dev = 0.0
        
        # Возвращаем VRAM из последнего успешного запуска
        vram = tps_list[-1]['vram'] if tps_list else None
        
        return avg_tps, vram, tps_list, None
    else:
        return 0.0, None, [], error

def get_top_options(runs, min_tps):
    valid = [r for r in runs if r['tps'] >= min_tps]
    if not valid:
        valid = [r for r in runs if r['tps'] > 0]
    valid.sort(key=lambda x: (x['ctx'], x['tps']), reverse=True)
    return valid[:3]


def save_results(results, output_file, output_format, model_names, test_models, args):
    """Сохранение результатов в JSON или CSV.
    
    Args:
        results: Словарь результатов по моделям
        output_file: Путь к файлу вывода
        output_format: Формат вывода ('json' или 'csv')
        model_names: Список названий протестированных моделей
        test_models: Список объектов моделей
        args: Пarsed arguments
    """
    if not output_file or not output_format:
        return
    
    # Подготовка данных для экспорта
    export_data = []
    current_language = _current_language if '_current_language' in globals() else "en"
    
    for model_name in model_names:
        if model_name not in results:
            continue
        
        model_obj = next((m for m in test_models if m['name'] == model_name), None)
        if not model_obj:
            continue
        
        model_info = {
            'model_name': model_name,
            'params': model_obj.get('params', 'N/A'),
            'quant': model_obj.get('quant', 'N/A'),
            'size_gb': model_obj.get('size_gb', 0),
            'max_ctx': model_obj.get('max_ctx', 0),
            'vision': model_obj.get('vision', '❓'),
            'tools': model_obj.get('tools', '❓'),
            'thinking': model_obj.get('thinking', '❌'),
            'language': current_language,
            'timestamp': datetime.now().isoformat()
        }
        
        # Добавляем результаты для каждого контекста
        for run in results.get(model_name, []):
            run_data = {
                'model_name': model_name,
                'ctx': run['ctx'],
                'ctx_str': f"{run['ctx'] // 1024}K" if run['ctx'] >= 1024 else str(run['ctx']),
                'avg_tps': round(run['avg_tps'], 2),
                'min_tps': round(run['min_tps'], 2),
                'max_tps': round(run['max_tps'], 2),
                'std_dev': round(run['std_dev'], 2),
                'vram': run['vram'] if run['vram'] else None,
                'vram_str': f"{run['vram'] / 1024 / 1024:.1f} MiB" if run['vram'] else None,
                **model_info
            }
            export_data.append(run_data)
    
    # Сохранение в зависимости от формата
    if output_format == 'json':
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print(get_text("output_json", output_file=output_file))
        except Exception as e:
            print(get_text("error_unknown", error_details=f"JSON export failed: {e}"))
    elif output_format == 'csv':
        try:
            fieldnames = ['model_name', 'ctx', 'ctx_str', 'avg_tps', 'min_tps', 'max_tps',
                         'std_dev', 'vram', 'vram_str', 'params', 'quant', 'size_gb',
                         'max_ctx', 'vision', 'tools', 'thinking', 'language', 'timestamp']
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(export_data)
            print(get_text("output_csv", output_file=output_file))
        except Exception as e:
            print(get_text("error_unknown", error_details=f"CSV export failed: {e}"))

def main():
    global _current_language
    
    # Инициализация языка по умолчанию
    if _current_language is None:
        _current_language = "en"
    
    parser = argparse.ArgumentParser(description=get_text("cli_help"))
    parser.add_argument('--models', type=str, help=get_text("cli_models"))
    parser.add_argument('--of', type=str, help=get_text("cli_of"))
    parser.add_argument('--lang', type=str, choices=get_available_languages(), default=_current_language, help=get_text("cli_lang"))
    parser.add_argument('--restart-method', type=str, default='systemctl', choices=['systemctl', 'docker', 'kill_start', 'manual'], help=get_text("cli_restart_method"))
    parser.add_argument('--no-restart', action='store_true', help=get_text("cli_no_restart"))
    parser.add_argument('--num-runs', type=int, default=3, help=get_text("cli_num_runs"))
    parser.add_argument('--context-sizes', type=str, help=get_text("cli_context_sizes"))
    parser.add_argument('--context-sizes-auto', action='store_true', help=get_text("cli_context_sizes_auto"))
    parser.add_argument('--output', type=str, help=get_text("cli_output"))
    parser.add_argument('--output-format', type=str, choices=['json', 'csv'], help=get_text("cli_output_format"))
    args = parser.parse_args()
    
    # Установка языка
    set_language(args.lang)
    
    print(get_text("app_title") + " (Context & VRAM Analyzer)\n")
    print(get_text("scanning_models"))
    models = get_models()
    
    if not models:
        return

    for m in models:
        vision, tools, thinking = get_capabilities_from_ollama_site(m["name"])
        m["vision"] = vision
        m["tools"] = tools
        m["thinking"] = thinking

    if args.of:
        print(get_text("filter_applied", of=args.of))
        filtered=[]
        for m in models:
            keep = True
            if 'v' in args.of and m["vision"] != "✅": keep = False
            if 'T' in args.of and m["tools"] != "✅": keep = False
            if 't' in args.of and m["thinking"] != "✅": keep = False
            if keep: filtered.append(m)
        models = filtered
        if not models:
            print(get_text("no_models_match_filter"))
            return

    test_models=[]

    if args.models:
        target_names = [name.strip() for name in args.models.split(',')]
        test_models = [m for m in models if m['name'] in target_names]
        
        found_names = [m['name'] for m in test_models]
        not_found = [name for name in target_names if name not in found_names]
        if not_found:
            print(get_text("no_models_found", models=', '.join(not_found)))
        if not test_models:
            print(get_text("no_test_models"))
            return
    else:
        print(get_text("available_models"))
        for i, m in enumerate(models):
            # Переводим макс контекст в читаемый формат 'K'
            max_ctx_str = f"{m['max_ctx'] // 1024}K" if m['max_ctx'] >= 1024 else str(m['max_ctx'])
            
            print(get_text("model_list_header", 
                index=i, 
                name=m['name'], 
                params=m['params'], 
                size_gb=m['size_gb'], 
                max_ctx_str=max_ctx_str,
                vision=m['vision'], 
                tools=m['tools'], 
                thinking=m['thinking']))
        
        selected_idx = input(get_text("select_models") + "\n")
        if selected_idx.strip().lower() == 'all':
            test_models = models
        else:
            try:
                indices = [int(x.strip()) for x in selected_idx.split(",")]
                test_models = [models[i] for i in indices]
            except (ValueError, IndexError):
                print(get_text("invalid_input"))
                return

    model_names_for_cmd = ",".join([m["name"] for m in test_models])
    # Use sys.executable for portability and absolute path to script
    script_name = os.path.basename(sys.argv[0])
    cmd_str = f"sudo {sys.executable} {script_name} --models {model_names_for_cmd}"  # Use sys.executable for portability
    if args.of:
        cmd_str += f" --of {args.of}"
    
    print("\n" + "="*60)
    print(get_text("repeated_run_header"))
    print(f"   {cmd_str}")
    print("="*60 + "\n")

    results = {}

    # Получаем список контекстов из аргументов
    context_sizes = get_context_sizes(args)
    
    for m in test_models:
        model_name = m["name"]
        max_ctx = m["max_ctx"]
        max_ctx_str = f"{max_ctx // 1024}K" if max_ctx >= 1024 else str(max_ctx)
        
        print(get_text("testing_model", model_name=model_name))
        print(get_text("model_size", size_gb=m['size_gb'], max_ctx_str=max_ctx_str))
        results[model_name] = []
        
        # Фильтруем контексты: берем только те, что МЕНЬШЕ ИЛИ РАВНЫ максимальному
        valid_contexts = [c for c in context_sizes if c <= max_ctx]
        
        # Если модель поддерживает какой-то "нестандартный" макс. контекст (например 128000),
        # которого нет в нашем списке, но он меньше 256K, можем добавить его для теста
        if max_ctx not in valid_contexts and max_ctx > 0 and max_ctx <= 262144:
            valid_contexts.append(max_ctx)
            valid_contexts.sort()

        if not valid_contexts:
            print(get_text("skipping_no_contexts"))
            continue

        for ctx in valid_contexts:
            restart_ollama(args.restart_method, args.no_restart)
            
            print(get_text("warming_up", ctx=ctx))
            avg_tps, vram, tps_list, error_msg = run_benchmark(model_name, ctx, args.num_runs)
            
            if error_msg:
                print(get_text("benchmark_failed", error_msg=error_msg))
                print(get_text("stopping_tests", model_name=model_name))
                break
                
            # Показываем результаты каждого запуска
            print(get_text("benchmark_runs_header"))
            for run in tps_list:
                run_num = run['run']
                tps = run['tps']
                vram_run = run['vram']
                if vram_run:
                    vram_str = f"{vram_run / 1024 / 1024:.1f} MiB"
                else:
                    vram_str = "N/A"
                print(f"   Запуск {run_num}: {tps:.2f} TPS (VRAM: {vram_str})")
            
            # Показываем сводку
            print(get_text("benchmark_summary"))
            print(f"   Среднее: {avg_tps:.2f} TPS")
            print(f"   Min: {min(r['tps'] for r in tps_list):.2f} TPS")
            print(f"   Max: {max(r['tps'] for r in tps_list):.2f} TPS")
            
            # Вычисляем std dev
            mean = avg_tps
            variance = sum((r['tps'] - mean) ** 2 for r in tps_list) / len(tps_list)
            std_dev = math.sqrt(variance)
            print(f"   Std Dev: {std_dev:.2f} TPS")
            
            # Сохраняем усреднённые результаты
            results[model_name].append({
                "ctx": ctx,
                "avg_tps": avg_tps,
                "min_tps": min(r['tps'] for r in tps_list),
                "max_tps": max(r['tps'] for r in tps_list),
                "std_dev": std_dev,
                "vram": vram
            })

    print("\n" + "="*60)
    print(get_text("recommendations_header"))
    print("="*60)
    
    for model_name, runs in results.items():
        if not runs:
            print(get_text("no_successful_runs", model_name=model_name))
            continue
        
        # Вывод результатов для каждой модели
        print("\n" + "="*60)
        print(get_text("results_header", model_name=model_name))
        print("="*60)
        
        for run in runs:
            ctx = run['ctx']
            ctx_str = f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)
            avg_tps = run['avg_tps']
            min_tps = run['min_tps']
            max_tps = run['max_tps']
            std_dev = run['std_dev']
            vram = run['vram']
            vram_str = f"{vram / 1024 / 1024:.1f} MiB" if vram else "N/A"
            
            print(get_text("result_row",
                ctx=ctx_str,
                avg_tps=avg_tps,
                min_tps=min_tps,
                max_tps=max_tps,
                std_dev=std_dev,
                vram=vram_str))
    
    # Сохранение результатов в файл
    save_results(results, args.output, args.output_format,
                 [m['name'] for m in test_models], test_models, args)

if __name__ == "__main__":
    main()
