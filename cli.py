"""Command-line interface argument parsing and validation."""

import argparse
import sys
import os
from i18n import get_text, get_available_languages, _current_language
from constants import CONTEXT_SIZES


def parse_args():
    """Parse and validate command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description=get_text("cli_help"))
    parser.add_argument('--models', type=str, help=get_text("cli_models"))
    parser.add_argument('--capabilities', '-f', type=str, help=get_text("cli_capabilities"))
    parser.add_argument('--lang', type=str, choices=get_available_languages(),
                        default=_current_language if _current_language else "en",
                        help=get_text("cli_lang"))
    parser.add_argument('--restart-method', type=str, default='systemctl',
                        choices=['systemctl', 'docker', 'kill_start', 'manual'],
                        help=get_text("cli_restart_method"))
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
