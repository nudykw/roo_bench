"""Centralized configuration constants for roo_bench."""

# Default context sizes for benchmarking
CONTEXT_SIZES = [8192, 16384, 32768, 65536, 131072, 262144]

# Default Ollama API URL
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Default API timeout in seconds
DEFAULT_TIMEOUT = 300

# Default number of benchmark runs per configuration
DEFAULT_NUM_RUNS = 3
