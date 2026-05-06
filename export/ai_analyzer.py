"""AI-powered analysis module for benchmark results."""

import json
import os
import time
from typing import Optional, Dict, List

import requests
from i18n import get_text, _current_language


class AIAnalyzer:
    """Handles AI analysis of benchmark results via Ollama."""

    PROMPTS_FILE = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'analysis_prompt.json')

    def __init__(self, base_url: str, headers: dict = None, timeout: int = 300):
        """Initialize AI analyzer.

        Args:
            base_url: Ollama API base URL
            headers: HTTP headers (e.g., authentication)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.headers = headers or {}
        self.timeout = timeout
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> dict:
        """Load prompt templates from prompts/analysis_prompt.json."""
        try:
            with open(self.PROMPTS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load prompts file: {e}. Using fallback.")
            return self._get_fallback_prompts()

    def _get_fallback_prompts(self) -> dict:
        """Return hardcoded fallback prompts if file not found."""
        return {
            "system_prompt": (
                "You are a performance analysis assistant specialized in evaluating "
                "LLM benchmark results for Roo Code workflow optimization."
            ),
            "user_prompt_template": (
                "Analyze these benchmark results and provide recommendations for "
                "the three main Roo Code modes.\n\n{results}"
            ),
            "translation_prompt_template": (
                "Translate the following text to {target_lang}. Output ONLY the "
                "translated text:\n\n{text}"
            )
        }

    def get_available_models(self) -> list:
        """Get list of models available for analysis.

        Returns:
            list: List of model dictionaries from Ollama API
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=self.timeout,
                headers=self.headers
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("models", [])
            return []
        except Exception:
            return []

    def format_results_for_prompt(self, all_results: dict, test_models: list) -> str:
        """Format benchmark results for inclusion in the prompt.

        Args:
            all_results: Dictionary of results per model
            test_models: List of model objects

        Returns:
            str: Formatted results string
        """
        lines = []
        for model_name, runs in all_results.items():
            lines.append(f"\n### {model_name}")
            model_info = next((m for m in test_models if m['name'] == model_name), {})
            lines.append(f"- Parameters: {model_info.get('params', 'N/A')}")
            lines.append(f"- Size: {model_info.get('size_gb', 'N/A')} GB")
            lines.append(f"- Vision: {model_info.get('vision', 'N/A')}")
            lines.append(f"- Tools: {model_info.get('tools', 'N/A')}")
            lines.append(f"- Thinking: {model_info.get('thinking', 'N/A')}")
            lines.append("- Results:")
            for run in runs:
                ctx_str = f"{run['ctx'] // 1024}K" if run['ctx'] >= 1024 else str(run['ctx'])
                lines.append(
                    f"  - Context: {ctx_str} | "
                    f"Avg: {run['avg_tps']:.2f} TPS | "
                    f"Min: {run['min_tps']:.2f} TPS | "
                    f"Max: {run['max_tps']:.2f} TPS | "
                    f"VRAM: {run.get('vram_str', 'N/A')}"
                )
        return "\n".join(lines)

    def generate_prompt(self, all_results: dict, test_models: list) -> str:
        """Generate the analysis prompt with benchmark results.

        Args:
            all_results: Dictionary of results per model
            test_models: List of model objects

        Returns:
            str: Complete prompt string
        """
        results_text = self.format_results_for_prompt(all_results, test_models)
        user_prompt = self.prompts['user_prompt_template'].format(results=results_text)
        return user_prompt

    def analyze(self, model_name: str, all_results: dict, test_models: list) -> str:
        """Send request to Ollama and get response.

        Args:
            model_name: Name of the Ollama model to use
            all_results: Dictionary of results per model
            test_models: List of model objects

        Returns:
            str: Model's analysis response

        Raises:
            Exception: If the API request fails
        """
        prompt = self.generate_prompt(all_results, test_models)
        
        # Calculate required context size (rough estimate: 1 token ~ 4 chars)
        prompt_chars = len(prompt)
        prompt_tokens_estimate = prompt_chars // 4 + 200  # Add larger buffer
        # Ensure minimum context of 8192 and round up to nearest 1024
        required_ctx = max(8192, ((prompt_tokens_estimate + 1023) // 1024) * 1024)
        
        # Print debug info about the request
        print(f"\n   📊 Debug info:")
        print(f"      Model: {model_name}")
        print(f"      Prompt size: {prompt_chars} chars (~{prompt_tokens_estimate} tokens)")
        print(f"      num_ctx: {required_ctx}")
        print(f"      num_predict: 4096")

        # Try using /api/chat endpoint first (better for complex prompts)
        try:
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a concise performance analysis assistant. Provide brief, structured recommendations. Use bullet points and keep responses under 1500 tokens. Be direct and avoid lengthy introductions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False,
                "options": {
                    "num_predict": 4096,
                    "temperature": 0.7,
                    "num_ctx": required_ctx,
                    "repeat_penalty": 1.1
                }
            }

            print(f"   📤 Sending request to /api/chat...")
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                response_text = data.get("message", {}).get("content", "")
                if response_text:
                    return response_text
                
                # If chat endpoint returns empty, show debug info
                print(f"   ⚠️  Chat endpoint returned empty response.")
                print(f"   🔍 Debug: Full response keys: {list(data.keys())}")
                print(f"   🔍 Debug: 'done' field: {data.get('done', 'N/A')}")
                print(f"   🔍 Debug: 'done_reason' field: {data.get('done_reason', 'N/A')}")
                print(f"   🔍 Debug: prompt_eval_count: {data.get('prompt_eval_count', 'N/A')}")
                print(f"   🔍 Debug: eval_count: {data.get('eval_count', 'N/A')}")
                print(f"   🔍 Debug: Full response: {json.dumps(data, indent=2)[:800]}")
                print(f"   🔄 Trying generate endpoint...")
        except Exception as e:
            print(f"   ⚠️  Chat endpoint failed: {e}, trying generate endpoint...")

        # Fallback to /api/generate endpoint
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 4096,
                "temperature": 0.7,
                "num_ctx": required_ctx,
                "repeat_penalty": 1.1
            }
        }

        try:
            print(f"   📤 Sending request to /api/generate...")
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                response_text = data.get("response", "")
                if response_text:
                    return response_text
                
                # If still empty, show debug info
                print(f"   ⚠️  Generate endpoint also returned empty response.")
                print(f"   🔍 Debug: Full response keys: {list(data.keys())}")
                print(f"   🔍 Debug: 'done' field: {data.get('done', 'N/A')}")
                print(f"   🔍 Debug: 'done_reason' field: {data.get('done_reason', 'N/A')}")
                print(f"   🔍 Debug: prompt_eval_count: {data.get('prompt_eval_count', 'N/A')}")
                print(f"   🔍 Debug: eval_count: {data.get('eval_count', 'N/A')}")
                print(f"   🔍 Debug: Full response: {json.dumps(data, indent=2)[:800]}")
                return response_text
            else:
                error_msg = response.json().get("error", f"HTTP {response.status_code}")
                raise Exception(str(error_msg))
        except Exception as e:
            raise Exception(f"Analysis request failed: {e}")

    def translate(self, text: str, target_lang: str, model_name: str = None) -> Optional[str]:
        """Translate text using Ollama model.

        Args:
            text: Text to translate
            target_lang: Target language code ('en' or 'ua')
            model_name: Optional model name to use for translation

        Returns:
            Optional[str]: Translated text or None if translation fails
        """
        if target_lang == 'en':
            return text

        lang_name = 'Ukrainian' if target_lang == 'ua' else target_lang
        prompt = self.prompts['translation_prompt_template'].format(
            target_lang=lang_name,
            text=text
        )

        use_model = model_name or "llama3.2"

        payload = {
            "model": use_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 2000
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("response", "").strip()
            else:
                return None
        except Exception:
            return None


def prompt_user(question: str) -> bool:
    """Ask yes/no question, return True if user agrees.

    Args:
        question: The question to ask

    Returns:
        bool: True if user confirms, False otherwise
    """
    while True:
        try:
            response = input(f"\n{question} (y/n): ").strip().lower()
            if response in ('y', 'yes', 'так', 'т', 'да', 'д'):
                return True
            elif response in ('n', 'no', 'н', 'ні', 'не', 'нет', 'н'):
                return False
        except (EOFError, KeyboardInterrupt):
            return False


def prompt_filename(default: str = "benchmark_results.json") -> str:
    """Ask for filename with default value.

    Args:
        default: Default filename to use if user presses Enter

    Returns:
        str: The filename entered by user or default
    """
    try:
        response = input(f"\n{get_text('ask_save_filename')} ").strip()
        return response if response else default
    except (EOFError, KeyboardInterrupt):
        return default


def save_results_interactive(all_results: dict, test_models: list, base_url: str, headers: dict) -> Optional[str]:
    """Interactive workflow: ask to save, get filename, save results.

    Args:
        all_results: Dictionary of results per model
        test_models: List of model objects
        base_url: Ollama API base URL
        headers: HTTP headers

    Returns:
        str: The filename that was saved to, or None if user declined
    """
    if not prompt_user(get_text("ask_save_results")):
        return None

    default_filename = get_text("save_filename_default")
    filename = prompt_filename(default_filename)

    # Import and save
    from export.result_saver import save_results
    save_results(
        all_results,
        filename,
        'json',
        [m['name'] for m in test_models],
        test_models
    )
    return filename


def analyze_results_interactive(analyzer: AIAnalyzer, all_results: dict, test_models: list, current_lang: str,
                                 restart_method: str = 'manual', no_restart: bool = False,
                                 ssh_client = None):
    """Interactive workflow: list models, let user select, run analysis, show results.

    Args:
        analyzer: AIAnalyzer instance
        all_results: Dictionary of results per model
        test_models: List of model objects
        current_lang: Current language code
        restart_method: Restart method ('systemctl', 'ssh', etc.)
        no_restart: If True, skip restart
        ssh_client: SSHClient instance for remote restart
    """
    models = analyzer.get_available_models()
    if not models:
        print(get_text("no_models_for_analysis"))
        return

    # Show model list
    print("\n" + "=" * 60)
    print(get_text("ask_select_model"))
    print("=" * 60)
    for i, model in enumerate(models, 1):
        name = model.get('name', 'unknown')
        size = model.get('size', 0) / (1024**3)
        details = model.get('details', {})
        param_size = details.get('parameter_size', 'N/A') if details else 'N/A'
        print(f"  {i}. {name} ({param_size}, {size:.1f} GB)")
    print(f"  0. Cancel")
    print("=" * 60)

    # Get user selection
    try:
        selection = input("> ").strip()
        idx = int(selection) - 1
        if idx < 0 or idx >= len(models):
            print("Cancelled.")
            return
        selected_model = models[idx]
    except ValueError:
        # Try matching by name
        name_match = next((m for m in models if m.get('name') == selection), None)
        if name_match:
            selected_model = name_match
        else:
            print("Invalid selection.")
            return

    model_name = selected_model.get('name', 'unknown')
    print(get_text("analysis_sending", model_name=model_name))

    # Restart Ollama before analysis to clear model from memory (same as benchmark)
    print("   🔄 Restarting Ollama before analysis...")
    try:
        from system.restart_manager import restart_ollama, RestartMethod
        
        # Convert string method to RestartMethod enum
        method_map = {
            'systemctl': RestartMethod.SYSTEMCTL,
            'docker': RestartMethod.DOCKER,
            'kill_start': RestartMethod.KILL_START,
            'manual': RestartMethod.MANUAL,
            'ssh': RestartMethod.SSH
        }
        restart_method_enum = method_map.get(restart_method, RestartMethod.MANUAL)
        
        restart_ollama(
            method=restart_method_enum,
            no_restart=no_restart,
            ssh_client=ssh_client
        )
    except Exception as e:
        print(f"   ⚠️  Warning: Could not restart Ollama: {e}")

    try:
        response = analyzer.analyze(model_name, all_results, test_models)

        if not response:
            print("\n⚠️  Model returned an empty response.")
            print("\n🔍 Possible causes:")
            print("  1. Prompt is too large for the model's context window")
            print("  2. The model doesn't support the /api/chat or /api/generate format")
            print("  3. The model encountered an internal error (check Ollama logs)")
            print("  4. The model's num_ctx setting is too small")
            print("\n💡 Troubleshooting steps:")
            print("  - Check Ollama logs: journalctl -u ollama --follow")
            print("  - Try a larger model (e.g., qwen2.5:32b or larger)")
            print("  - Try reducing the benchmark results (fewer models/tests)")
            print("  - Ensure the model supports the chat format (use models with 'chat' in name)")
            print("\n📋 You can also try running the analysis with --no-interactive flag")
            print("   and specify an output file to skip this step.")
            return

        print("\n" + "=" * 60)
        print(get_text("analysis_response", model_name=model_name))
        print("=" * 60)
        print(response)

        # Unload model from VRAM after analysis is complete (BEFORE translation)
        print(f"\n   🧹 Unloading model '{model_name}' from VRAM...")
        try:
            from api.factory import ApiClientFactory
            analysis_client = ApiClientFactory.create_client(
                base_url=analyzer.base_url,
                headers=analyzer.headers,
                timeout=analyzer.timeout,
                ssh_host=getattr(ssh_client, 'host', None) if ssh_client else None,
                ssh_user=getattr(ssh_client, 'user', None) if ssh_client else None,
                ssh_port=getattr(ssh_client, 'port', 22) if ssh_client else 22,
                ssh_key=getattr(ssh_client, 'key_path', None) if ssh_client else None
            )
            analysis_client.unload_model(model_name)
        except Exception as e:
            print(f"   ⚠️  Warning: Could not unload model: {e}")
            print(f"   💡 Tip: Run 'ollama restart' to free VRAM.")
        
        # Translate if needed (AFTER unload to avoid conflicts)
        if current_lang != 'en':
            print(f"\n" + "=" * 60)
            print(get_text("analysis_raw_response"))
            print("=" * 60)
            print(response)

            print(f"\n" + "=" * 60)
            print(get_text("analysis_translated"))
            print("=" * 60)
            translated = analyzer.translate(response, current_lang, model_name)
            if translated:
                print(translated)
            else:
                print(get_text("translation_unavailable"))
    except Exception as e:
        print(get_text("analysis_error", error=str(e)))
