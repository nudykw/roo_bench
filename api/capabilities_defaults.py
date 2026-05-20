"""Default capabilities and patterns for CapabilitiesFetcher."""

import re

# Default built-in capabilities for well-known models
DEFAULT_CAPABILITIES: dict[str, dict[str, bool]] = {
    # Vision models
    'llava': {'vision': True, 'tools': False, 'thinking': False},
    'llama3.2': {'vision': True, 'tools': True, 'thinking': False},  # 3.2 has both vision and tools
    'nemotron': {'vision': False, 'tools': True, 'thinking': False},
    'phi3-v': {'vision': True, 'tools': False, 'thinking': False},
    'moondream': {'vision': True, 'tools': False, 'thinking': False},
    'stable-beluga': {'vision': False, 'tools': False, 'thinking': False},
    'granite3.2-v': {'vision': True, 'tools': False, 'thinking': False},
    'granite3.2-v:2b': {'vision': True, 'tools': False, 'thinking': False},
    'granite3.2-v:8b': {'vision': True, 'tools': False, 'thinking': False},
    
    # Reasoning/Thinking models
    'deepseek-r1': {'vision': False, 'tools': True, 'thinking': True},
    'deepseek:70b': {'vision': False, 'tools': True, 'thinking': True},
    'deepseek:671b': {'vision': False, 'tools': True, 'thinking': True},
    'qwq': {'vision': False, 'tools': True, 'thinking': True},
    'qwq-preview': {'vision': False, 'tools': True, 'thinking': True},
    'qwen2.5:72b': {'vision': False, 'tools': True, 'thinking': False},
    'qwen3': {'vision': False, 'tools': True, 'thinking': True},
    'qwen3:0.6b': {'vision': False, 'tools': True, 'thinking': True},
    'qwen3:1.7b': {'vision': False, 'tools': True, 'thinking': True},
    'qwen3:4b': {'vision': False, 'tools': True, 'thinking': True},
    'qwen3:8b': {'vision': False, 'tools': True, 'thinking': True},
    'qwen3:14b': {'vision': False, 'tools': True, 'thinking': True},
    'qwen3:30b': {'vision': False, 'tools': True, 'thinking': True},
    'qwen3:32b': {'vision': False, 'tools': True, 'thinking': True},
    'qwen3:235b': {'vision': False, 'tools': True, 'thinking': True},
    
    # Tool-use models
    'qwen2.5': {'vision': False, 'tools': True, 'thinking': False},
    'qwen2.5-coder': {'vision': False, 'tools': True, 'thinking': False},
    'llama3.1': {'vision': False, 'tools': True, 'thinking': False},
    'llama3': {'vision': False, 'tools': True, 'thinking': False},
    'mistral': {'vision': False, 'tools': True, 'thinking': False},
    'mixtral': {'vision': False, 'tools': True, 'thinking': False},
    'codestral': {'vision': False, 'tools': True, 'thinking': False},
    'codex': {'vision': False, 'tools': True, 'thinking': False},
    'phi4': {'vision': False, 'tools': True, 'thinking': False},
    'granite3.2': {'vision': False, 'tools': True, 'thinking': False},
    'granite3.2:2b': {'vision': False, 'tools': True, 'thinking': False},
    'granite3.2:8b': {'vision': False, 'tools': True, 'thinking': False},
    'dolphin-mistral': {'vision': False, 'tools': True, 'thinking': False},
    'neural-chat': {'vision': False, 'tools': True, 'thinking': False},
    'zephyr': {'vision': False, 'tools': True, 'thinking': False},
    'codetrans': {'vision': False, 'tools': True, 'thinking': False},
    'gemma3': {'vision': True, 'tools': True, 'thinking': False},
    'gemma3:4b': {'vision': True, 'tools': True, 'thinking': False},
    'gemma3:12b': {'vision': True, 'tools': True, 'thinking': False},
    'gemma3:27b': {'vision': True, 'tools': True, 'thinking': False},
    
    # Qwen 3.5 family (vision + tools + thinking)
    'qwen3.5': {'vision': True, 'tools': True, 'thinking': True},
    'qwen3.5:0.8b': {'vision': True, 'tools': True, 'thinking': True},
    'qwen3.5:2b': {'vision': True, 'tools': True, 'thinking': True},
    'qwen3.5:4b': {'vision': True, 'tools': True, 'thinking': True},
    'qwen3.5:9b': {'vision': True, 'tools': True, 'thinking': True},
    'qwen3.5:27b': {'vision': True, 'tools': True, 'thinking': True},
    'qwen3.5:35b': {'vision': True, 'tools': True, 'thinking': True},
    'qwen3.5:122b': {'vision': True, 'tools': True, 'thinking': True},
    
    # Qwen 3.6 family (vision + tools + thinking)
    'qwen3.6': {'vision': True, 'tools': True, 'thinking': True},
    'qwen3.6-35b': {'vision': True, 'tools': True, 'thinking': True},
    'qwen3.6:27b': {'vision': True, 'tools': True, 'thinking': True},
    'qwen3.6:35b': {'vision': True, 'tools': True, 'thinking': True},
    
    # Gemma 4 family (vision + tools + thinking + audio)
    'gemma4': {'vision': True, 'tools': True, 'thinking': True},
    'gemma4:e2b': {'vision': True, 'tools': True, 'thinking': True},
    'gemma4:e4b': {'vision': True, 'tools': True, 'thinking': True},
    'gemma4:26b': {'vision': True, 'tools': True, 'thinking': True},
    'gemma4:31b': {'vision': True, 'tools': True, 'thinking': True},
}

# Size patterns - model sizes look like "1.5b", "7b", "70b", "70b-q4_K_M"
SIZE_PATTERN = re.compile(r'^[\d.]+[bk]m?$', re.IGNORECASE)

# Color class patterns (used for size badges, not capabilities)
COLOR_PATTERN = re.compile(r'bg-\[#?[a-z0-9]+\]')

# Known MoE architectures for detection
MOE_ARCHITECTURES = ['mixtral', 'qwen2.5-moe', 'deepseek-v3', 'deepseek-v2', 'jamba']

# Known dense (non-MoE) architectures
DENSE_ARCHITECTURES = ['llama', 'gemma', 'qwen', 'mistral', 'phi', 'granite']

# Known MoE model patterns (for name-based detection)
MOE_PATTERNS = ['mixtral', 'qwen2.5-moe', 'deepseek-v3', 'deepseek-v2', 'jamba']

# Known dense model patterns (for name-based detection)
DENSE_PATTERNS = ['llama', 'gemma', 'qwen3', 'qwen2.5', 'phi', 'granite']

# MoE-related keywords for key-based detection
MOE_KEYWORDS = ['moe', 'mixture', 'expert', 'router']

# Vision model patterns
VISION_MODELS = ['llava', 'moondream', 'neural-chat', 'all-minilm']

# Models with both vision and tools
VISION_TOOLS_MODELS = ['llama3.2']

# Reasoning models
REASONING_MODELS = ['deepseek-r1', 'deepscaler', 'qwq', 'qwq-preview', 'o1', 'o3']

# Models with tools
TOOL_MODELS = ['qwen2.5', 'qwen2', 'llama3.1', 'llama3', 'mistral', 'mixtral',
               'codestral', 'codex', 'dolphin', 'neural-chat', 'zephyr']
