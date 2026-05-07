"""Command-line interface argument parsing and validation."""

import argparse
import sys
import os
import logging
from i18n import get_text, get_available_languages, _current_language
from constants import CONTEXT_SIZES


logger = logging.getLogger('roo_bench')


def setup_logging(verbose_level: int = 0):
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


def parse_args():
    """Parse and validate command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description=get_text("cli_help"))
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Increase verbosity level (use -v, -vv, -vvv for more debug output)')
    parser.add_argument('--models', type=str, help=get_text("cli_models"))
    parser.add_argument('--capabilities', '-f', type=str, help=get_text("cli_capabilities"))
    parser.add_argument('--lang', type=str, choices=get_available_languages(),
                        default=_current_language if _current_language else "en",
                        help=get_text("cli_lang"))
    parser.add_argument('--restart-method', type=str, default='manual',
                        choices=['systemctl', 'docker', 'kill_start', 'manual', 'ssh'],
                        help=get_text("cli_restart_method"))
    parser.add_argument('--ssh-host', type=str, help='SSH host for remote restart')
    parser.add_argument('--ssh-user', type=str, help='SSH user for remote restart')
    parser.add_argument('--ssh-port', type=int, default=22, help='SSH port for remote restart')
    parser.add_argument('--ssh-key', type=str, help='Path to SSH private key')
    parser.add_argument('--no-restart', action='store_true', help=get_text("cli_no_restart"))
    parser.add_argument('--num-runs', type=int, default=3, help=get_text("cli_num_runs"))
    parser.add_argument('--context-sizes', type=str, help=get_text("cli_context_sizes"))
    parser.add_argument('--context-sizes-auto', action='store_true', help=get_text("cli_context_sizes_auto"))
    parser.add_argument('--output', type=str, help=get_text("cli_output"))
    parser.add_argument('--output-format', type=str, choices=['json', 'csv'],
                        help=get_text("cli_output_format"))
    parser.add_argument('--ollama-url', type=str, help='Ollama server URL')
    parser.add_argument('--ollama-port', type=int, help='Ollama server port')
    parser.add_argument('--ollama-api-key', type=str, help='API key for authentication')
    parser.add_argument('--ollama-timeout', type=int, help='Connection timeout')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--update-cache', action='store_true',
                        help='Force update capabilities cache from Ollama API')
    parser.add_argument('--no-interactive', action='store_true',
                        help='Disable interactive post-benchmark prompts')
    parser.add_argument('--analyze-file', type=str, metavar='FILE',
                        help=get_text('cli_analyze_file'))
    parser.add_argument('--analysis-model', type=str, default=None,
                        help=get_text('cli_analysis_model'))
    parser.add_argument('--no-stream', action='store_true', default=False,
                        help='Disable streaming mode for AI analysis output (default: enabled)')
    parser.add_argument('--no-thinking', action='store_true', default=True,
                        help='Disable thinking mode on all models to prevent reasoning loops (default: enabled)')
    parser.add_argument('--thinking', action='store_false', dest='no_thinking',
                        help='Enable thinking mode on thinking-capable models')
    parser.add_argument('--independent', action='store_true',
                        help='Run only independent prompts test mode')
    parser.add_argument('--chain', type=str, metavar='CHAIN_ID',
                        help='Run only the specified prompt chain (e.g., chain_rest_api)')
    parser.add_argument('--prompts-file', type=str, metavar='FILE',
                        help='Path to prompts.jsonc configuration file')
    parser.add_argument('--list-chains', action='store_true',
                        help='List available prompt chains and exit')
    parser.add_argument('--list-independent', action='store_true',
                        help='List available independent prompts and exit')
    return parser.parse_args()


def get_context_sizes(args) -> list:
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
            sizes = [int(x.strip()) for x in args.context_sizes.split(',')]
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
