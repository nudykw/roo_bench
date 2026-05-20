"""Command-line interface argument parsing and validation."""

import argparse
import logging
from argparse import Namespace

from constants import CONTEXT_SIZES
from i18n import _current_language, get_available_languages, get_text

logger = logging.getLogger('roo_bench')


def parse_context_size(size_str: str) -> int:
    """Parse a context size string into an integer number of tokens.
    
    Supports formats:
        - Plain number: "131072", "2048"
        - K suffix: "8K", "16K", "128K", "2048K"
        - M suffix: "1M" (1048576)
    
    Args:
        size_str: String representation of context size
        
    Returns:
        int: Number of tokens
        
    Raises:
        ValueError: If the string cannot be parsed
    """
    size_str = size_str.strip().upper()
    
    # Check for K suffix (kilobytes)
    if size_str.endswith('K'):
        num = float(size_str[:-1])
        return int(num * 1024)
    
    # Check for M suffix (megabytes)
    if size_str.endswith('M'):
        num = float(size_str[:-1])
        return int(num * 1024 * 1024)
    
    # Plain number
    return int(size_str)


def setup_logging(verbose_level: int = 0) -> None:
    """Configure logging based on verbose level.
    
    Args:
        verbose_level: 0 = warnings only, 1 = INFO, 2 = DEBUG, 3 = DEBUG with timestamps
    """
    if verbose_level == 0:
        level = logging.WARNING
    elif verbose_level == 1:
        level = logging.INFO
    elif verbose_level == 2:
        level = logging.DEBUG
    else:
        level = logging.DEBUG
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    return logger


def parse_args() -> Namespace:
    """Parse and validate command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description=get_text("cli_help"))
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help=get_text("cli_verbose"))
    parser.add_argument('--models', type=str, help=get_text("cli_models"))
    parser.add_argument('--capabilities', '-f', type=str, help=get_text("cli_capabilities"))
    parser.add_argument('--lang', type=str, dest='language', choices=get_available_languages(),
                        default=_current_language if _current_language else "en",
                        help=get_text("cli_lang"))
    parser.add_argument('--restart-method', type=str, default='manual',
                        choices=['systemctl', 'docker', 'kill_start', 'manual', 'ssh'],
                        help=get_text("cli_restart_method"))
    parser.add_argument('--ssh-host', type=str, help=get_text("cli_ssh_host"))
    parser.add_argument('--ssh-user', type=str, help=get_text("cli_ssh_user"))
    parser.add_argument('--ssh-port', type=int, default=22, help=get_text("cli_ssh_port"))
    parser.add_argument('--ssh-key', type=str, help=get_text("cli_ssh_key"))
    parser.add_argument('--no-restart', action='store_true', help=get_text("cli_no_restart"))
    parser.add_argument('--num-runs', type=int, default=3, help=get_text("cli_num_runs"))
    parser.add_argument('--context-sizes', type=str, help=get_text("cli_context_sizes"))
    parser.add_argument('--context-sizes-auto', action='store_true', help=get_text("cli_context_sizes_auto"))
    parser.add_argument('--output', type=str, help=get_text("cli_output"))
    parser.add_argument('--output-format', type=str, choices=['json', 'csv'],
                        help=get_text("cli_output_format"))
    parser.add_argument('--ollama-url', type=str, help=get_text("cli_ollama_url"))
    parser.add_argument('--ollama-port', type=int, help=get_text("cli_ollama_port"))
    parser.add_argument('--ollama-api-key', type=str, help=get_text("cli_ollama_api_key"))
    parser.add_argument('--ollama-timeout', type=int, help=get_text("cli_ollama_timeout"))
    parser.add_argument('--config', type=str, help=get_text("cli_config"))
    parser.add_argument('--update-cache', action='store_true',
                        help=get_text("cli_update_cache"))
    parser.add_argument('--no-interactive', action='store_true',
                        help=get_text("cli_no_interactive"))
    parser.add_argument('--analyze-file', type=str, metavar='FILE',
                        help=get_text('cli_analyze_file'))
    parser.add_argument('--analysis-model', type=str, default=None,
                        help=get_text('cli_analysis_model'))
    parser.add_argument('--no-stream', action='store_true', default=False,
                        help=get_text("cli_no_stream"))
    parser.add_argument('--no-thinking', action='store_true', default=True,
                        help=get_text("cli_no_thinking"))
    parser.add_argument('--thinking', action='store_false', dest='no_thinking',
                        help=get_text("cli_thinking"))
    parser.add_argument('--independent', action='store_true',
                        help=get_text("cli_independent"))
    parser.add_argument('--chains', action='store_true',
                        help=get_text("cli_chains"))
    parser.add_argument('--chain', type=str, metavar='CHAIN_ID',
                        help=get_text("cli_chain"))
    parser.add_argument('--prompts-file', type=str, metavar='FILE',
                        help=get_text("cli_prompts_file"))
    parser.add_argument('--list-chains', action='store_true',
                        help=get_text("cli_list_chains"))
    parser.add_argument('--list-independent', action='store_true',
                        help=get_text("cli_list_independent"))
    parser.add_argument('--independent-top', type=int, default=None, metavar='N',
                        help=get_text("cli_independent_top"))
    parser.add_argument('--generate-md', action='store_true',
                        help=get_text("cli_generate_md"))
    parser.add_argument('--analysis-prompt-file', type=str, default=None, metavar='FILE',
                        help=get_text("cli_analysis_prompt_file"))
    parser.add_argument('--num-predict', type=int, default=12000,
                        help=get_text("cli_num_predict"))
    parser.add_argument('--temperature', type=str, default=None,
                        help=get_text("cli_temperature"))
    return parser.parse_args()


def get_context_sizes(args: Namespace) -> list[int]:
    """Get list of context sizes from CLI arguments.

    Args:
        args: Parsed arguments from argparse

    Returns:
        list: List of context sizes
    """
    # If --context-sizes-auto is specified, generate a geometric progression
    if args.context_sizes_auto:
        sizes = []
        current = 8192
        while current <= 262144:
            sizes.append(current)
            current *= 2
        return sizes

    # If --context-sizes is specified, parse from string
    if args.context_sizes:
        try:
            sizes = [parse_context_size(x.strip()) for x in args.context_sizes.split(',')]
            if all(s > 0 for s in sizes):
                return sorted(sizes)
            else:
                print(get_text("error_invalid_context_size", sizes=args.context_sizes))
                return CONTEXT_SIZES
        except ValueError:
            print(get_text("error_invalid_context_size", sizes=args.context_sizes))
            return CONTEXT_SIZES

    # Otherwise use defaults
    return CONTEXT_SIZES


def get_temperature_test_values(args: Namespace) -> list[float]:
    """Get list of temperature test values from CLI arguments.

    Args:
        args: Parsed arguments from argparse

    Returns:
        list: List of temperature values
    """
    DEFAULT_TEMPERATURES = [0.0, 0.66, 1.0]
    
    # If --temperature is specified, parse from string
    if args.temperature:
        try:
            values = [float(x.strip()) for x in args.temperature.split(',')]
            if all(0.0 <= v <= 2.0 for v in values):
                return sorted(values)
            else:
                print(get_text("error_invalid_temperature", values=args.temperature))
                return DEFAULT_TEMPERATURES
        except ValueError:
            print(get_text("error_invalid_temperature", values=args.temperature))
            return DEFAULT_TEMPERATURES

    # Otherwise use defaults
    return DEFAULT_TEMPERATURES
