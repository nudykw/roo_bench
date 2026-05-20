"""Main entry point for roo_bench - orchestration of all modules."""

import logging
import os
import signal
import sys
from argparse import Namespace

from cli import parse_args
from config import OllamaConfig
from i18n import get_text, set_language
from main_workflow import _run_benchmark_workflow_impl, signal_handler

# Setup logging
logger = logging.getLogger('roo_bench')

# Default output file for benchmark results
DEFAULT_OUTPUT_FILE = "benchmark_results.json"


SAVE_MODE_DISABLED = "disabled"
SAVE_MODE_NEW = "new"
SAVE_MODE_OVERWRITE = "overwrite"
SAVE_MODE_MERGE = "merge"


def validate_expert_prompts() -> bool:
    """Check if expert evaluation prompts are available.

    Returns:
        bool: True if prompts file exists and is readable, False otherwise.
    """
    if os.path.exists('prompts/analysis_prompt.jsonc'):
        return True
    logger.warning(
        "Expert evaluation prompts not found. Expert-Evaluator will use default templates."
    )
    return False


def prompt_user(message: str) -> bool:
    """Prompt user for yes/no confirmation.

    Args:
        message: Message to display

    Returns:
        bool: True if user confirms, False otherwise
    """
    response = input(f"{message} (y/n): ").strip().lower()
    return response in ['y', 'yes']


def prompt_output_filename(default: str = DEFAULT_OUTPUT_FILE) -> str:
    """Prompt user for output filename with default value.

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


def get_ollama_config(args: Namespace) -> OllamaConfig:
    """Get Ollama configuration from CLI arguments.

    Args:
        args: Parsed arguments from argparse

    Returns:
        OllamaConfig: Configuration object
    """
    cli_config = {
        'ollama_url': getattr(args, 'ollama_url', None),
        'ollama_port': getattr(args, 'ollama_port', None),
        'ollama_api_key': getattr(args, 'ollama_api_key', None),
        'ollama_timeout': getattr(args, 'ollama_timeout', None),
        'backend_type': getattr(args, 'backend', 'ollama'),
        'config': getattr(args, 'config', None)
    }
    return OllamaConfig(cli_config)


def run_benchmark_workflow(config: OllamaConfig, args: Namespace) -> None:
    """Main benchmark workflow orchestration.

    Args:
        config: OllamaConfig object
        args: Parsed arguments
    """
    from i18n import get_text

    try:
        _run_benchmark_workflow_impl(config, args)
    except KeyboardInterrupt:
        print("\n" + get_text("benchmark_interrupted"))
        sys.exit(0)


def _post_benchmark_workflow(results_file, all_results, args, base_url) -> None:
    """Post-benchmark workflow: ask about AI analysis and model selection."""
    from export.ai_analyzer import AIAnalyzer, analyze_results_interactive

    if not all_results or not results_file:
        return

    if prompt_user(get_text("ask_ai_analysis")):
        try:
            analysis_prompt_file = getattr(args, 'analysis_prompt_file', None)
            analyzer = AIAnalyzer(base_url=base_url, analysis_prompt_file=analysis_prompt_file)
            analyze_results_interactive(
                analyzer, all_results,
                getattr(args, 'language', 'en'),
                getattr(args, 'restart_method', 'manual'),
                getattr(args, 'no_restart', False)
            )
        except Exception as e:
            print(f"⚠️  AI analysis failed: {e}")


def _main_impl() -> None:
    """Main implementation with language setup."""
    from cli import setup_logging

    args = parse_args()

    # Handle --generate-md flag
    if getattr(args, 'generate_md', False):
        from prompts.generate_md import generate_all_markdown
        success = generate_all_markdown()
        sys.exit(0 if success else 1)

    # Setup logging
    setup_logging(getattr(args, 'verbose', 0))

    # Set language
    lang = getattr(args, 'language', 'en')
    set_language(lang)

    # Get configuration
    config = get_ollama_config(args)

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Run benchmark workflow
    run_benchmark_workflow(config, args)


def main() -> None:
    """Main entry point."""
    try:
        _main_impl()
    except KeyboardInterrupt:
        print("\n" + get_text("benchmark_interrupted"))
        sys.exit(0)


if __name__ == "__main__":
    main()
