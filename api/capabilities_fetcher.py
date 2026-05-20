"""Model capabilities and metadata fetching from ollama.com.

This module provides comprehensive model metadata caching including:
- Capabilities (vision, tools, thinking, audio)
- Model metadata (size, params, quantization, max_ctx)
- MoE (Mixture of Experts) detection
"""

import urllib.parse
from typing import Any

from api.capabilities_api import CapabilitiesApi
from api.capabilities_html import CapabilitiesHtml


class CapabilitiesFetcher(CapabilitiesApi, CapabilitiesHtml):
    """Fetches and caches model capabilities and metadata from Ollama API and HTML."""

    CACHE_VERSION = 1
    CACHE_TTL_HOURS = 24

    def get_capabilities_from_model_db(self, base_name: str) -> tuple[str, str, str] | None:
        """Get capabilities from the built-in model knowledge base or cache.
        
        Returns:
            tuple: (vision, tools, thinking) or None if not found
        """
        # Direct lookup
        if base_name in self.MODEL_CAPABILITIES:
            caps = self.MODEL_CAPABILITIES[base_name]
            return (
                "\u2705" if caps['vision'] else "\u274c",
                "\u2705" if caps['tools'] else "\u274c",
                "\u2705" if caps['thinking'] else "\u274c"
            )
        
        # Pattern matching for variants
        clean_name = base_name
        for prefix in ['dev-', '+']:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]
                break
        
        if clean_name != base_name and clean_name in self.MODEL_CAPABILITIES:
            caps = self.MODEL_CAPABILITIES[clean_name]
            return (
                "\u2705" if caps['vision'] else "\u274c",
                "\u2705" if caps['tools'] else "\u274c",
                "\u2705" if caps['thinking'] else "\u274c"
            )
        
        # Check for deepseek variants (they have thinking)
        if 'deepseek' in base_name.lower():
            if 'r1' in base_name.lower() or 'qwq' in base_name.lower():
                return "\u274c", "\u2705", "\u2705"
            return "\u274c", "\u2705", "\u274c"
        
        # Check for qwen2.5/qwen3 variants with thinking (qwq has thinking)
        if 'qwq' in base_name.lower():
            return "\u274c", "\u2705", "\u2705"
        
        # Check for vision models by name patterns
        if any(kw in base_name.lower() for kw in ['llava', 'moondream', 'phi.3.v', 'phi.3-v']):
            return "\u2705", "\u274c", "\u274c"
        
        return None

    def get_capabilities(self, model_name: str) -> tuple[str, str, str]:
        """Get model capabilities from multiple sources.
        
        Args:
            model_name: Model name (e.g., "llama3.2" or "dev-qwen2.5")
        
        Returns:
            tuple: (vision, tools, thinking) - capability statuses
        """
        base_name = model_name.split(':')[0]
        if base_name.startswith('dev-'):
            base_name = base_name[4:]

        # Method 1: Check built-in model database + cache (fastest, most reliable)
        result = self.get_capabilities_from_model_db(base_name)
        if result is not None:
            return result

        # Method 2: Try HTML parsing from ollama.com
        try:
            result = self.get_capabilities_from_html(base_name)
            if result is not None:
                # Cache the result for future use
                self.MODEL_CAPABILITIES[base_name] = {
                    'vision': result[0] == "\u2705",
                    'tools': result[1] == "\u2705",
                    'thinking': result[2] == "\u2705"
                }
                return result
        except Exception as e:
            print(f"\u26a0\ufe0f  Error parsing HTML for {base_name}: {e}")

        # Method 3: Fallback - try search API
        try:
            search_url = f"https://ollama.com/api/search?q={urllib.parse.quote(base_name)}"
            import requests
            search_resp = requests.get(search_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            if search_resp.status_code == 200:
                search_data = search_resp.json()
                for item in search_data.get("results", []):
                    if item.get("name") == base_name:
                        caps = item.get("capabilities", {})
                        vision = "\u2705" if caps.get("vision") else "\u274c"
                        tools = "\u2705" if caps.get("tools") else "\u274c"
                        thinking = "\u2705" if caps.get("thinking") else "\u274c"
                        # Cache the result
                        self.MODEL_CAPABILITIES[base_name] = {
                            'vision': vision == "\u2705",
                            'tools': tools == "\u2705",
                            'thinking': thinking == "\u2705"
                        }
                        return vision, tools, thinking
        except Exception as e:
            print(f"\u26a0\ufe0f  Error searching for {base_name}: {e}")

        # Final fallback: heuristic based on model name patterns
        return self._heuristic_capabilities(base_name)

    def _heuristic_capabilities(self, base_name: str) -> tuple[str, str, str]:
        """Determine capabilities using heuristics based on model name patterns.
        
        This is the last resort when no other method succeeds.
        """
        name_lower = base_name.lower()
        
        # Vision models
        if any(kw in name_lower for kw in ['llava', 'moondream', 'neural-chat', 'all-minilm']):
            return "\u2705", "\u274c", "\u274c"
        
        # Models with both vision and tools (llama3.2 family)
        if 'llama3.2' in name_lower:
            return "\u2705", "\u2705", "\u274c"
        
        # Reasoning models
        if any(kw in name_lower for kw in ['deepseek-r1', 'deepscaler', 'qwq', 'qwq-preview', 'o1', 'o3']):
            return "\u274c", "\u2705", "\u2705"
        
        # Models with tools (most modern models)
        if any(kw in name_lower for kw in ['qwen2.5', 'qwen2', 'llama3.1', 'llama3', 'mistral', 'mixtral',
                        'codestral', 'codex', 'dolphin', 'neural-chat', 'zephyr']):
            return "\u274c", "\u2705", "\u274c"
        
        # Older models or specific variants without tools
        return "\u274c", "\u274c", "\u274c"
