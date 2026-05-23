"""AI-powered analysis module for benchmark results."""

import json
import logging
from typing import Any, cast

from benchmark.result import BenchmarkResult
from i18n import get_text
from prompts.analysis_prompt_loader import AnalysisPromptLoader
from ui.input_validator import InputValidator
from ui.markdown_renderer import display_markdown

logger = logging.getLogger(__name__)


class AIAnalyzer:
    """Handles AI analysis of benchmark results via Ollama."""

    def __init__(self, base_url: str, headers: dict[str, Any] | None = None, timeout: int = 300,
                 analysis_prompt_file: str | None = None):
        """Initialize AIAnalyzer with Ollama connection details.
        
        Args:
            base_url: Ollama API base URL.
            headers: Optional headers for API requests.
            timeout: Request timeout in seconds.
            analysis_prompt_file: Optional path to analysis prompts file.
                                 If None, uses default .md or .jsonc.
        """
        self.base_url = base_url
        self.headers = headers or {}
        self.timeout = timeout
        self._prompt_loader = AnalysisPromptLoader(analysis_prompt_file)
        self.prompts = self._load_prompts()
        logger.info("[AIAnalyzer] Loaded analysis prompts from: %s", self._prompt_loader.file_path)
        print(f"{get_text('using_analysis_prompts')}: {self._prompt_loader.file_path}")

    def _load_prompts(self) -> dict[str, Any]:
        """Load prompt templates from AnalysisPromptLoader."""
        try:
            data = self._prompt_loader.data
            return data
        except Exception as e:
            logger.warning("Could not load prompts file: %s", e)
            return self._get_fallback_prompts()

    def _get_fallback_prompts(self) -> dict[str, str]:
        """Return fallback prompts if file loading fails."""
        return {
            "analyze": "Analyze these benchmark results and provide recommendations.",
            "translate": "Translate the following text to {target_lang}:"
        }

    def get_available_models(self) -> list[dict[str, Any]]:
        """Get list of models available for analysis."""
        import requests
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return [m for m in response.json().get('models', [])]
        except Exception as e:
            logger.error("Failed to get available models: %s", e)
            return []

    def format_results_for_prompt(self, all_results: list[BenchmarkResult]) -> str:
        """Format benchmark results for AI prompt."""
        lines = []
        for result in all_results:
            model_name = result.model.name
            lines.append(f"Model: {model_name}")
            for metric in result.results:
                lines.append(f"  Context: {metric.ctx}, TPS: {metric.avg_tps:.2f}")
        return "\n".join(lines)

    def generate_prompt(self, all_results: list[BenchmarkResult]) -> str:
        """Generate analysis prompt from results."""
        base_prompt = self.prompts.get('analyze', 'Analyze these results:')
        formatted = self.format_results_for_prompt(all_results)
        return f"{base_prompt}\n\n{formatted}"

    def analyze(self, model_name: str, all_results: list[BenchmarkResult], ollama_client: Any = None, stream: bool = True) -> str:
        """Send request to Ollama and get response."""
        import requests

        prompt = self.generate_prompt(all_results)

        try:
            if stream:
                response_text = ""
                with requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "stream": True
                    },
                    headers=self.headers,
                    timeout=self.timeout,
                    stream=True
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            chunk = json.loads(line)
                            if 'response' in chunk:
                                print(chunk['response'], end='', flush=True)
                                response_text += chunk['response']
                print()
                return response_text
            else:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "stream": False
                    },
                    headers=self.headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return cast(str, response.json().get('response', ''))
        except Exception as e:
            logger.error("Analysis failed: %s", e)
            raise

    def translate(self, text: str, target_lang: str, model_name: str | None = None) -> str | None:
        """Translate text using Ollama model."""
        import requests

        prompt = self.prompts.get('translate', 'Translate to {lang}: {text}').format(
            lang=target_lang, text=text
        )
        model = model_name or self.get_available_models()[0]['name'] if self.get_available_models() else None

        if not model:
            return None

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return cast(str, response.json().get('response', ''))
        except Exception as e:
            logger.error("Translation failed: %s", e)
            return None

    def analyze_from_file(self, file_path: str, model_name: str, target_lang: str = 'en', ollama_client: Any = None, stream: bool = True) -> bool:
        """Analyze benchmark results from a saved file."""
        from export.merge_utils import load_results_file

        results, _ = load_results_file(file_path)
        if not results:
            print("No results found in file.")
            return False

        return True


def prompt_filename(default: str = "benchmark_results.json") -> str:
    """Ask for filename with default value."""
    try:
        response = input(f"\n{get_text('ask_save_filename')} ").strip()
        return response if response else default
    except (EOFError, KeyboardInterrupt):
        return default


def save_results_interactive(all_results: list[BenchmarkResult], base_url: str, headers: dict[str, Any], filename: str | None = None) -> str | None:
    """Interactive workflow: ask to save, get filename, save results.

    Args:
        all_results: List of BenchmarkResult objects
        base_url: Ollama API base URL
        headers: HTTP headers
        filename: Optional pre-determined filename (if provided, will save directly without prompting)

    Returns:
        str: The filename that was saved to, or None if user declined
    """
    # If filename is provided, save directly without prompting
    if filename:
        final_filename = filename
    else:
        # Ask user if they want to save
        if not InputValidator.prompt_yes_no(get_text("ask_save_results")):
            return None
        default_filename = get_text("save_filename_default")
        final_filename = prompt_filename(default_filename)

    # Import and save
    from config import OllamaConfig
    from export.result_saver import save_results
    from prompts.loader import PromptLoader

    # Create config and prompt loader
    config = OllamaConfig()
    prompt_loader = PromptLoader(config.prompts_file)

    save_results(
        all_results,
        final_filename,
        'json',
        prompts_config=prompt_loader.data
    )
    return final_filename


def analyze_results_interactive(analyzer: AIAnalyzer, all_results: list[BenchmarkResult], current_lang: str,
                               restart_method: str = 'manual', no_restart: bool = False,
                               ssh_client: Any = None, ollama_client: Any = None) -> None:
    """Interactive workflow: list models, let user select, run analysis, show results.

    Args:
        analyzer: AIAnalyzer instance
        all_results: List of BenchmarkResult objects
        current_lang: Current language code
        restart_method: Restart method ('systemctl', 'ssh', etc.)
        no_restart: If True, skip restart
        ssh_client: SSHClient instance for remote restart
        ollama_client: BaseApiClient instance for unload_model (preferred over restart_ollama)
    """
    from i18n import get_text

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
    print("  0. Cancel")
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
        response = analyzer.analyze(model_name, all_results, ollama_client, stream=True)

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
        display_markdown(response)

        # Translate if needed (BEFORE unload to avoid model reload)
        if current_lang != 'en':
            print("\n" + "=" * 60)
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
        print(f"\n   🥹 Unloading model '{model_name}' from VRAM...")
        try:
            if ollama_client:
                ollama_client.unload_model(model_name)
            else:
                print("   ⚠️  Warning: ollama_client not available, skipping unload")
            print(f"   ✅ Model '{model_name}' unloaded successfully")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not unload model: {e}")
            print(f"   💡 Tip: Run 'ollama stop {model_name}' manually to free VRAM.")
    except Exception as e:
        print(get_text("analysis_error", error=str(e)))
