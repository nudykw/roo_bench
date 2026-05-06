import requests
import subprocess
import time
import urllib.parse
import argparse
from enum import Enum
from bs4 import BeautifulSoup
from i18n import get_text, set_language, get_available_languages

OLLAMA_URL = "http://localhost:11434"
CONTEXT_SIZES = [8192, 16384, 32768, 65536, 131072, 262144]


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
    base_name = model_name.split(':')[0]
    if base_name.startswith('dev-'):
        base_name = base_name[4:]

    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        url = f"https://ollama.com/library/{base_name}"
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code != 200:
            search_url = f"https://ollama.com/search?q={urllib.parse.quote(base_name)}"
            search_resp = requests.get(search_url, headers=headers, timeout=5)
            if search_resp.status_code == 200:
                soup = BeautifulSoup(search_resp.text, 'html.parser')
                link = soup.find('a', href=lambda href: href and '/library/' in href)
                if link:
                    url = "https://ollama.com" + link['href']
                    response = requests.get(url, headers=headers, timeout=5)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            page_text = soup.get_text().lower()
            
            vision = "✅" if "vision" in page_text else "❌"
            tools = "✅" if "tools" in page_text or "tool use" in page_text else "❌"
            thinking = "✅" if "thinking" in page_text or "reasoning" in page_text or "deepseek" in base_name else "❌"
            return vision, tools, thinking
            
    except Exception:
        pass
    return "❓", "❓", "❓"

def get_models():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        models = []
        for m in response.json().get("models",[]):
            details = m.get("details", {})
            size_gb = m.get("size", 0) / (1024 ** 3)
            
            # Получаем максимальный контекст модели через API show
            max_ctx = 8192 # Значение по умолчанию, если не удалось найти
            try:
                show_resp = requests.post(f"{OLLAMA_URL}/api/show", json={"name": m["name"]})
                if show_resp.status_code == 200:
                    model_info = show_resp.json().get("model_info", {})
                    # Ищем ключ, содержащий 'context_length' (например 'llama.context_length' или 'qwen2.context_length')
                    for key, val in model_info.items():
                        if 'context_length' in key:
                            max_ctx = int(val)
                            break
            except Exception:
                pass

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
        return[]

def run_benchmark(model_name, context_size):
    """Запуск бенчмарка для модели.
    
    Args:
        model_name: Имя модели
        context_size: Размер контекста
        
    Returns:
        tuple: (tps, vram, error)
            tps: токлов в секунду (float)
            vram: использование VRAM в байтах (int или None если GPU недоступен)
            error: сообщение об ошибке (str или None)
    """
    prompt = "Write a comprehensive Python script that implements a multithreaded web server. Explain every line in extreme detail."
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 100,
            "num_ctx": context_size
        }
    }

    vram_before = get_vram_usage()
    response = None
    
    try:
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=300)
        
        if response.status_code != 200:
            try:
                err_msg = response.json().get("error", f"HTTP {response.status_code}")
            except:
                err_msg = f"HTTP {response.status_code}: {response.text[:100]}"
            return 0, None, get_text("error_ollama_api", error_msg=err_msg)
            
        try:
            data = response.json()
        except Exception as json_err:
            raw_text = response.text[:200].replace('\n', ' ')
            return 0, None, get_text("error_parsing_response", json_err=json_err, raw_text=raw_text)
            
        vram_after = get_vram_usage()
        total_duration = data.get("total_duration", 0) / 1e9
        eval_count = data.get("eval_count", 0)
        tps = eval_count / total_duration if total_duration > 0 else 0
        return tps, vram_after, None
        
    except requests.exceptions.Timeout:
        return 0, None, get_text("error_timeout")
    except requests.exceptions.ConnectionError:
        return 0, None, get_text("error_crash")
    except Exception as e:
        err_details = get_text("error_unknown", error_details=str(e))
        if response is not None:
            err_details += f" | Сырой ответ: {response.text[:200]}"
        return 0, None, err_details

def get_top_options(runs, min_tps):
    valid = [r for r in runs if r['tps'] >= min_tps]
    if not valid:
        valid = [r for r in runs if r['tps'] > 0]
    valid.sort(key=lambda x: (x['ctx'], x['tps']), reverse=True)
    return valid[:3]

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
    parser.add_argument('--num-runs', type=int, default=1, help=get_text("cli_num_runs"))
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
    cmd_str = f"sudo ./venv/bin/python roo_bench.py --models {model_names_for_cmd}"
    if args.of:
        cmd_str += f" --of {args.of}"
    
    print("\n" + "="*60)
    print(get_text("repeated_run_header"))
    print(f"   {cmd_str}")
    print("="*60 + "\n")

    results = {}

    for m in test_models:
        model_name = m["name"]
        max_ctx = m["max_ctx"]
        max_ctx_str = f"{max_ctx // 1024}K" if max_ctx >= 1024 else str(max_ctx)
        
        print(get_text("testing_model", model_name=model_name))
        print(get_text("model_size", size_gb=m['size_gb'], max_ctx_str=max_ctx_str))
        results[model_name] = []
        
        # Фильтруем контексты: берем только те, что МЕНЬШЕ ИЛИ РАВНЫ максимальному
        valid_contexts = [c for c in CONTEXT_SIZES if c <= max_ctx]
        
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
            tps, vram, error_msg = run_benchmark(model_name, ctx)
            
            if tps == 0:
                print(get_text("benchmark_failed", error_msg=error_msg))
                print(get_text("stopping_tests", model_name=model_name))
                break 
                
            if tps > 15 and vram < 23500:
                status = "⚡ FLYING (GPU)"
                print(get_text("speed_status_gpu", tps=tps, vram=vram))
            elif tps < 8:
                status = "🐌 SLOW (RAM/CPU)"
                print(get_text("speed_status_ram", tps=tps, vram=vram))
            else:
                status = "✅ NORMAL"
                print(get_text("speed_status_normal", tps=tps, vram=vram))
            
            results[model_name].append({
                "ctx": ctx,
                "tps": tps,
                "vram": vram
            })

    print("\n" + "="*60)
    print(get_text("recommendations_header"))
    print("="*60)
    
    for model_name, runs in results.items():
        if not runs:
            print(get_text("no_successful_runs", model_name=model_name))
            continue
            
        print(get_text("architect_mode"))
        arch_options = get_top_options(runs, min_tps=1.5)
        for i, opt in enumerate(arch_options, 1):
            print(get_text("variant", i=i, ctx=opt['ctx'], tps=opt['tps']))

        print(get_text("code_mode"))
        code_options = get_top_options(runs, min_tps=12.0)
        for i, opt in enumerate(code_options, 1):
            warn = " (⚠️ Below recommended speed)" if opt['tps'] < 12.0 else ""
            print(get_text("variant", i=i, ctx=opt['ctx'], tps=opt['tps']) + warn)

        print(get_text("debug_mode"))
        debug_options = get_top_options(runs, min_tps=7.0)
        for i, opt in enumerate(debug_options, 1):
            print(get_text("variant", i=i, ctx=opt['ctx'], tps=opt['tps']))

if __name__ == "__main__":
    main()
