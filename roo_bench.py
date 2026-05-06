#!/usr/bin/env python3
"""roo_bench - Context & VRAM Analyzer for Ollama models.

This module is a thin wrapper for backward compatibility.
The main entry point is now `main.py` which orchestrates all refactored modules.

Module structure:
    - cli.py              : Command-line argument parsing
    - config.py           : Ollama configuration
    - constants.py        : Configuration constants
    - i18n.py             : Internationalization
    - api/                : Ollama API client and capabilities fetcher
    - benchmark/          : Benchmark execution and results
    - system/             : GPU monitoring and restart management
    - ui/                 : Interactive and output UI
    - export/             : Results export (JSON/CSV)
"""

from main import main

if __name__ == "__main__":
    main()
