"""AI-powered analysis module for benchmark results."""

import json
import logging
import os
import time
from typing import Optional, Dict, List

import requests
from i18n import get_text, _current_language
from ui.markdown_renderer import display_markdown, stream_markdown

logger = logging.getLogger('roo_bench')


class AIAnalyzer:
    """Handles AI analysis of benchmark results via Ollama."""

    PROMPTS_FILE = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'prompts', 'analysis_prompt.json'))

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

    def analyze(self, model_name: str, all_results: dict, test_models: list, ollama_client = None, stream: bool = True) -> str:
        """Send request to Ollama and get response.

        Args:
            model_name: Name of the Ollama model to use
            all_results: Dictionary of results per model
            test_models: List of model objects
            ollama_client: BaseApiClient instance for unload_model (same as benchmark runner)
            stream: If True, use streaming mode with progress display (default: True)

        Returns:
            str: Model's analysis response

        Raises:
            Exception: If the API request fails
        """
        prompt = self.generate_prompt(all_results, test_models)
        
        # Calculate required context size (rough estimate: 1 token ~ 3 chars for mixed content)
        # Add 50% buffer for tokenization overhead and response tokens
        prompt_chars = len(prompt)
        prompt_tokens_estimate = int(prompt_chars / 3 * 1.5)
        # Round up to nearest 1024, use model's max_ctx or calculated value (whichever is larger)
        required_ctx = max(32768, ((prompt_tokens_estimate + 1023) // 1024) * 1024)
        
        # Debug info about the request (only shown with -v/-vv)
        logger.debug("Analysis request details:")
        logger.debug("  Model: %s", model_name)
        logger.debug("  Prompt size: %d chars (~%d tokens)", prompt_chars, prompt_tokens_estimate)
        logger.debug("  num_ctx: %d", required_ctx)
        logger.debug("  num_predict: -1 (unlimited)")
        logger.debug("  temperature: 0.1 (strict/deterministic)")
        logger.debug("  think: False (disabled)")
        logger.debug("  stream: %s", stream)

        # Try using /api/chat endpoint first (better for complex prompts)
        try:
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a concise performance analysis assistant. Provide brief, structured recommendations. Use bullet points. Be direct and avoid lengthy introductions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": stream,
                "think": False,  # Top-level API param to disable thinking mode
                "options": {
                    "num_predict": -1,
                    "temperature": 0.1,  # Low temperature for strict, deterministic output
                    "num_ctx": required_ctx,
                    "repeat_penalty": 1.1
                }
            }

            logger.debug("Sending request to /api/chat...")
            if stream:
                # Stream mode
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    headers=self.headers,
                    stream=True,
                    timeout=self.timeout
                )

                def chunk_generator():
                    for line in response.iter_lines():
                        if line:
                            chunk = json.loads(line)
                            if not chunk.get("done", False):
                                yield chunk.get("message", {}).get("content", "")
                            else:
                                # Last chunk might have content even with done=True
                                content = chunk.get("message", {}).get("content", "")
                                if content:
                                    yield content

                return stream_markdown(chunk_generator())
            else:
                # Non-stream mode
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
                    logger.warning("Chat endpoint returned empty response.")
                    logger.debug("Full response keys: %s", list(data.keys()))
                    logger.debug("'done' field: %s", data.get('done', 'N/A'))
                    logger.debug("'done_reason' field: %s", data.get('done_reason', 'N/A'))
                    logger.debug("prompt_eval_count: %s", data.get('prompt_eval_count', 'N/A'))
                    logger.debug("eval_count: %s", data.get('eval_count', 'N/A'))
                    logger.debug("Full response: %s", json.dumps(data, indent=2)[:800])
                    logger.debug("Trying generate endpoint...")
        except Exception as e:
            logger.warning("Chat endpoint failed: %s, trying generate endpoint...", str(e))

        # Fallback to /api/generate endpoint
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": stream,
            "think": False,  # Top-level API param to disable thinking mode
            "options": {
                "num_predict": -1,
                "temperature": 0.1,  # Low temperature for strict, deterministic output
                "num_ctx": required_ctx,
                "repeat_penalty": 1.1
            }
        }

        try:
            logger.debug("Sending request to /api/generate...")
            if stream:
                # Stream mode for generate endpoint
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    headers=self.headers,
                    stream=True,
                    timeout=self.timeout
                )

                def chunk_generator():
                    for line in response.iter_lines():
                        if line:
                            chunk = json.loads(line)
                            if not chunk.get("done", False):
                                yield chunk.get("response", "")
                            else:
                                # Last chunk might have content even with done=True
                                content = chunk.get("response", "")
                                if content:
                                    yield content

                return stream_markdown(chunk_generator())
            else:
                # Non-stream mode
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
                    logger.warning("Generate endpoint also returned empty response.")
                    logger.debug("Full response keys: %s", list(data.keys()))
                    logger.debug("'done' field: %s", data.get('done', 'N/A'))
                    logger.debug("'done_reason' field: %s", data.get('done_reason', 'N/A'))
                    logger.debug("prompt_eval_count: %s", data.get('prompt_eval_count', 'N/A'))
                    logger.debug("eval_count: %s", data.get('eval_count', 'N/A'))
                    logger.debug("Full response: %s", json.dumps(data, indent=2)[:800])
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
        # Debug: Log input parameters (only at DEBUG level)
        logger.debug("translate() called:")
        logger.debug("  target_lang: %s", target_lang)
        logger.debug("  model_name: %s", model_name or 'default (llama3.2)')
        logger.debug("  text length: %d chars", len(text))
        logger.debug("  base_url: %s", self.base_url)
        
        if target_lang == 'en':
            logger.debug("Target language is 'en', returning text as-is.")
            return text

        lang_name = 'Ukrainian' if target_lang == 'ua' else target_lang
        prompt = self.prompts['translation_prompt_template'].format(
            target_lang=lang_name,
            text=text
        )
        
        logger.debug("prompt constructed: %d chars", len(prompt))

        use_model = model_name or "llama3.2"
        logger.debug("using model: %s", use_model)

        # Use /api/chat with system prompt to disable thinking mode
        # Note: "think" is a top-level API parameter, NOT inside "options"
        # See: https://github.com/ollama/ollama/blob/main/docs/api.md
        payload = {
            "model": use_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a translator. Translate the following text. Output ONLY the translated text without any additional commentary, thinking, or explanations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "think": False  # Disable reasoning mode to prevent token waste (top-level API param)
        }

        try:
            logger.debug("Sending translation request to %s/api/chat...", self.base_url)
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            
            logger.debug("response status code: %d", response.status_code)

            if response.status_code == 200:
                data = response.json()
                logger.debug("response keys: %s", list(data.keys()))
                logger.debug("response 'done': %s", data.get('done'))
                logger.debug("response 'done_reason': %s", data.get('done_reason'))
                logger.debug("response 'total_duration': %s", data.get('total_duration'))
                logger.debug("response 'eval_count': %s", data.get('eval_count'))
                
                # /api/chat returns response in message.content, /api/generate in response
                if "message" in data:
                    translated = data["message"].get("content", "").strip()
                    # Fallback: if content is empty, try reading from thinking
                    if not translated:
                        translated = data["message"].get("thinking", "").strip()
                else:
                    translated = data.get("response", "").strip()
                
                logger.debug("translated text length: %d chars", len(translated))
                
                if translated:
                    logger.debug("Translation successful.")
                    return translated
                else:
                    logger.warning("Translation returned empty response.")
                    logger.debug("full response: %s", json.dumps(data, indent=2)[:500])
                    return None
            else:
                error_msg = response.json().get("error", f"HTTP {response.status_code}")
                logger.warning("Translation failed with HTTP %d: %s", response.status_code, error_msg)
                logger.debug("response text: %s", response.text[:500])
                return None
        except requests.exceptions.Timeout as e:
            logger.warning("Translation request timed out after %d seconds.", self.timeout)
            logger.debug("timeout error: %s", e)
            return None
        except requests.exceptions.ConnectionError as e:
            logger.warning("Translation request connection error.")
            logger.debug("connection error: %s", e)
            logger.debug("Tip: Check if Ollama is running at %s", self.base_url)
            return None
        except Exception as e:
            logger.warning("Translation error: %s", e)
            logger.debug("exception type: %s", type(e).__name__)
            logger.debug("exception str: %s", str(e))
            return None

    def analyze_from_file(self, file_path: str, model_name: str, target_lang: str = 'en', ollama_client = None, stream: bool = True) -> bool:
        """Analyze benchmark results from a saved file.

        Args:
            file_path: Path to the saved results JSON/CSV file
            model_name: Name of the Ollama model to use for analysis
            target_lang: Target language code for translation ('en' or 'ua')
            ollama_client: BaseApiClient instance for unload_model (same as benchmark runner)
            stream: If True, use streaming mode with progress display (default: True)

        Returns:
            bool: True if analysis was successful, False otherwise
        """
        from export.result_saver import load_results_from_file
        
        print(f"\n   \U0001f4da Loading results from: {file_path}")
        all_results, test_models = load_results_from_file(file_path)
        
        if all_results is None or test_models is None:
            return False
        
        print(f"   \u2705 Loaded {len(test_models)} models with results")
        
        # Get available models and let user select
        models = self.get_available_models()
        if not models:
            print(get_text("no_models_for_analysis"))
            return False
        
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
                return False
            selected_model = models[idx]
        except ValueError:
            # Try matching by name
            name_match = next((m for m in models if model_name in m.get('name', '')), None)
            if name_match:
                selected_model = name_match
            else:
                print(f"Model '{model_name}' not found.")
                return False
        
        actual_model_name = selected_model.get('name', 'unknown')
        print(f"   \U0001f4e4 Using model: {actual_model_name}")
        
        # Check if model is already loaded in VRAM
        try:
            running_models = self.get_available_models()
            model_already_loaded = any(actual_model_name in m.get('name', '') for m in running_models)
            if model_already_loaded:
                print(f"   ✅ Model '{actual_model_name}' is already loaded in VRAM")
            else:
                print(f"   ℹ️  Model '{actual_model_name}' is not loaded - will be loaded on first request")
        except Exception as e:
            logger.debug("Could not check model availability: %s", e)
        
        try:
            print(f"   \U0001f4a1 Sending analysis request...")
            response = self.analyze(actual_model_name, all_results, test_models, ollama_client, stream=stream)
            
            if not response:
                print("\n\u26a0\ufe0f  Model returned an empty response.")
                return False
            
            print("\n" + "=" * 60)
            print(get_text("analysis_response", model_name=actual_model_name))
            print("=" * 60)
            display_markdown(response)
            
            # Translate if needed (BEFORE unloading to avoid model reload)
            if target_lang != 'en':
                print(f"\n" + "=" * 60)
                print(get_text("analysis_translated"))
                print("=" * 60)
                
                # Check if model is still loaded
                try:
                    running_models = self.get_available_models()
                    model_still_loaded = any(actual_model_name in m.get('name', '') for m in running_models)
                    if model_still_loaded:
                        print(f"   ✅ Model '{actual_model_name}' is still loaded - translating without reload")
                    else:
                        print(f"   ⚠️  Model '{actual_model_name}' is NOT loaded - will be reloaded for translation")
                except Exception as e:
                    logger.debug("Could not check model availability: %s", e)
                
                print(f"   🌐 Starting translation...")
                translated = self.translate(response, target_lang, actual_model_name)
                if translated:
                    display_markdown(translated)
                    print(f"   ✅ Translation completed")
                else:
                    print(get_text("translation_unavailable"))
         
        except Exception as e:
            print(get_text("analysis_error", error=str(e)))
            return False
        
        # Unload model after analysis and translation are complete
        # Same method as benchmark runner: ollama_client.unload_model()
        print(f"\n   🤹 Unloading model '{actual_model_name}' from VRAM...")
        try:
            if ollama_client:
                ollama_client.unload_model(actual_model_name)
            else:
                print(f"   ⚠️  Warning: ollama_client not available, skipping unload")
            print(f"   ✅ Model '{actual_model_name}' unloaded successfully")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not unload model: {e}")
            print(f"   💡 Tip: Run 'ollama stop {actual_model_name}' manually to free VRAM.")
        
        return True
        

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
            if response in ('y', 'yes', '\u0442\u0430\u043a', '\u0442', '\u0434\u0430', '\u0434'):
                return True
            elif response in ('n', 'no', '\u043d', '\u043d\u0456', '\u043d\u0435', '\u043d\u0435\u0442', '\u043d'):
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
    from prompts.loader import PromptLoader
    from config import OllamaConfig
    
    # Create config and prompt loader
    config = OllamaConfig()
    prompt_loader = PromptLoader(config.prompts_file)
    
    save_results(
        all_results,
        filename,
        'json',
        [m['name'] for m in test_models],
        test_models,
        prompts_config=prompt_loader.data
    )
    return filename


def analyze_results_interactive(analyzer: AIAnalyzer, all_results: dict, test_models: list, current_lang: str,
                               restart_method: str = 'manual', no_restart: bool = False,
                               ssh_client = None, ollama_client = None):
    """Interactive workflow: list models, let user select, run analysis, show results.

    Args:
        analyzer: AIAnalyzer instance
        all_results: Dictionary of results per model
        test_models: List of model objects
        current_lang: Current language code
        restart_method: Restart method ('systemctl', 'ssh', etc.)
        no_restart: If True, skip restart
        ssh_client: SSHClient instance for remote restart
        ollama_client: BaseApiClient instance for unload_model (preferred over restart_ollama)
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
    
    # Check if model is already loaded
    try:
        running_models = analyzer.get_available_models()
        model_already_loaded = any(model_name in m.get('name', '') for m in running_models)
        if model_already_loaded:
            print(f"   ✅ Model '{model_name}' is already loaded in VRAM")
        else:
            print(f"   ℹ️  Model '{model_name}' is not loaded - will be loaded on first request")
    except Exception as e:
        logger.debug("Could not check model availability: %s", e)
    
    try:
        response = analyzer.analyze(model_name, all_results, test_models, ollama_client, stream=True)

        if not response:
            print("\n\u26a0\ufe0f  Model returned an empty response.")
            print("\n\U0001f50d Possible causes:")
            print("  1. Prompt is too large for the model's context window")
            print("  2. The model doesn't support the /api/chat or /api/generate format")
            print("  3. The model encountered an internal error (check Ollama logs)")
            print("  4. The model's num_ctx setting is too small")
            print("\n\U0001f4a1 Troubleshooting steps:")
            print("  - Check Ollama logs: journalctl -u ollama --follow")
            print("  - Try a larger model (e.g., qwen2.5:32b or larger)")
            print("  - Try reducing the benchmark results (fewer models/tests)")
            print("  - Ensure the model supports the chat format (use models with 'chat' in name)")
            print("\n\U0001f4cb You can also try running the analysis with --no-interactive flag")
            print("   and specify an output file to skip this step.")
            return

        print("\n" + "=" * 60)
        print(get_text("analysis_response", model_name=model_name))
        print("=" * 60)
        display_markdown(response)

        # Translate if needed (BEFORE unload to avoid model reload)
        if current_lang != 'en':
            print(f"\n" + "=" * 60)
            print(get_text("analysis_translated"))
            print("=" * 60)
            
            # Debug: Check model is still loaded before translation
            try:
                running_models = analyzer.get_available_models()
                model_in_list = any(model_name in m.get('name', '') for m in running_models)
                logger.debug("model '%s' in available models: %s", model_name, model_in_list)
            except Exception as e:
                logger.debug("could not check model availability: %s", e)
            
            translated = analyzer.translate(response, current_lang, model_name)
            if translated:
                display_markdown(translated)
            else:
                print(get_text("translation_unavailable"))
    
        # Unload model after analysis and translation are complete
        # Same method as benchmark runner: ollama_client.unload_model()
        print(f"\n   \U0001f939 Unloading model '{model_name}' from VRAM...")
        try:
            if ollama_client:
                ollama_client.unload_model(model_name)
            else:
                print(f"   \u26a0\ufe0f  Warning: ollama_client not available, skipping unload")
            print(f"   \u2705 Model '{model_name}' unloaded successfully")
        except Exception as e:
            print(f"   \u26a0\ufe0f  Warning: Could not unload model: {e}")
            print(f"   \U0001f4a1 Tip: Run 'ollama stop {model_name}' manually to free VRAM.")
    except Exception as e:
        print(get_text("analysis_error", error=str(e)))
