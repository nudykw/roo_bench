"""Model capabilities and metadata fetching from ollama.com.

This module provides comprehensive model metadata caching including:
- Capabilities (vision, tools, thinking, audio)
- Model metadata (size, params, quantization, max_ctx)
- MoE (Mixture of Experts) detection
"""

import json
import os
import re
import requests
import urllib.parse
from bs4 import BeautifulSoup
from datetime import datetime, timezone


class CapabilitiesFetcher:
    """Fetches and caches model capabilities and metadata from Ollama API and HTML."""

    # Path to the cache files (relative to this module)
    CACHE_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'capabilities_cache.json')
    MODEL_CACHE_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'model_cache.json')
    
    # Cache version (increment to force cache refresh)
    CACHE_VERSION = 1
    
    # Cache TTL in hours (refresh after this period)
    CACHE_TTL_HOURS = 24
    
    # Default built-in capabilities for well-known models
    DEFAULT_CAPABILITIES = {
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

    def __init__(self, cache_file: str = None, model_cache_file: str = None):
        """Initialize CapabilitiesFetcher with cache.
        
        Args:
            cache_file: Path to capabilities JSON cache file. Defaults to data/capabilities_cache.json
            model_cache_file: Path to model metadata JSON cache file. Defaults to data/model_cache.json
        """
        self.cache_file = cache_file or self.CACHE_FILE
        self.model_cache_file = model_cache_file or self.MODEL_CACHE_FILE
        # Start with default capabilities, then merge with cached data
        self.MODEL_CAPABILITIES = dict(self.DEFAULT_CAPABILITIES)
        self._load_cache()
        # Model metadata cache (full data from /api/show)
        self.model_metadata = {}
        self._load_model_cache()
    
    def is_cache_fresh(self) -> bool:
        """Check if the model cache is fresh (not expired).
        
        Returns:
            bool: True if cache exists and was fetched within CACHE_TTL_HOURS
        """
        if not os.path.exists(self.model_cache_file):
            return False
        
        try:
            with open(self.model_cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            cache_timestamp = cache_data.get('cache_timestamp', '')
            if not cache_timestamp:
                return False
            
            # Parse timestamp and check TTL
            fetched_at = datetime.fromisoformat(cache_timestamp.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            age_hours = (now - fetched_at).total_seconds() / 3600
            
            return age_hours < self.CACHE_TTL_HOURS
        except (json.JSONDecodeError, IOError, ValueError):
            return False
    
    def _load_model_cache(self):
        """Load cached model metadata from JSON file."""
        if not os.path.exists(self.model_cache_file):
            return
        
        try:
            with open(self.model_cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Validate cache version
            version = cache_data.get('cache_version', 0)
            if version < self.CACHE_VERSION:
                print(f"⚠️  Cache version {version} is outdated (need {self.CACHE_VERSION})")
                return
            
            self.model_metadata = cache_data.get('models', {})
            timestamp = cache_data.get('cache_timestamp', 'unknown')
            print(f"✅ Loaded model cache: {len(self.model_metadata)} models from {timestamp} ({self.model_cache_file})")
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  Error loading model cache: {e}")

    def _load_cache(self):
        """Load cached capabilities from JSON file."""
        if not os.path.exists(self.cache_file):
            return
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Merge cached data with defaults (cache takes precedence)
            if isinstance(cached_data, dict):
                self.MODEL_CAPABILITIES.update(cached_data)
                print(f"✅ Loaded cache: {len(cached_data)} models from {self.cache_file}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  Error loading cache: {e}")

    def add_model_from_api(self, model_name: str, capabilities: dict):
        """Add a model's capabilities to the cache.
        
        Args:
            model_name: Full model name (e.g., 'qwen3.6-35b:q4_0')
            capabilities: Dict with 'vision', 'tools', 'thinking', 'audio' booleans
        """
        # Extract base name (without version/quant tags)
        base_name = model_name.split(':')[0]
        
        # Store with both full name and base name
        self.MODEL_CAPABILITIES[base_name] = {
            'vision': capabilities.get('vision', False),
            'tools': capabilities.get('tools', False),
            'thinking': capabilities.get('thinking', False)
        }
        # Also store full name if it differs
        if base_name != model_name:
            self.MODEL_CAPABILITIES[model_name] = {
                'vision': capabilities.get('vision', False),
                'tools': capabilities.get('tools', False),
                'thinking': capabilities.get('thinking', False)
            }
    
    def save_cache(self):
        """Save current MODEL_CAPABILITIES to JSON file.
        
        Saves only models that were discovered via HTML parsing or search API
        (i.e., not the default built-in models). This keeps the cache file small.
        """
        try:
            # Ensure directory exists
            cache_dir = os.path.dirname(self.cache_file)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            
            # Only save models that are NOT in DEFAULT_CAPABILITIES
            # This keeps the cache file small and only stores discovered models
            cached_only = {
                k: v for k, v in self.MODEL_CAPABILITIES.items()
                if k not in self.DEFAULT_CAPABILITIES
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cached_only, f, indent=2, ensure_ascii=False)
            
            if cached_only:
                print(f"💾 Saved cache: {len(cached_only)} new models to {self.cache_file}")
            else:
                print(f"ℹ️  No new models to save (all models are in default database)")
        except IOError as e:
            print(f"⚠️  Error saving cache: {e}")
    
    # ========================================================================
    # Model Metadata Cache Methods
    # ========================================================================
    
    def save_model_cache(self):
        """Save model metadata to JSON cache file.
        
        Saves all discovered model metadata including capabilities, MoE info,
        size, params, quantization, and context length.
        """
        try:
            # Ensure directory exists
            cache_dir = os.path.dirname(self.model_cache_file)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            
            cache_data = {
                'cache_version': self.CACHE_VERSION,
                'cache_timestamp': datetime.now(timezone.utc).isoformat(),
                'models': self.model_metadata
            }
            
            with open(self.model_cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved model cache: {len(self.model_metadata)} models to {self.model_cache_file}")
        except IOError as e:
            print(f"⚠️  Error saving model cache: {e}")
    
    def add_model_metadata(self, model_name: str, metadata: dict):
        """Add or update model metadata in the cache.
        
        Args:
            model_name: Full model name (e.g., 'qwen3.6-35b:q4_0')
            metadata: Dict containing model metadata from /api/show
        """
        base_name = model_name.split(':')[0]
        
        # Extract and normalize all relevant fields
        cached_entry = {
            'name': model_name,
            'base_name': base_name,
            'size': metadata.get('size', 0),
            'size_gb': round(metadata.get('size', 0) / (1024**3), 1) if metadata.get('size', 0) > 0 else "N/A",
            'params': self._extract_params(metadata),
            'total_params': self._extract_total_params(metadata),
            'active_params': self._extract_active_params(metadata),
            'quant': self._extract_quant(metadata),
            'max_ctx': self._extract_max_ctx(metadata),
            'capabilities': self._extract_capabilities(metadata),
            'moe': self._detect_moe(metadata, base_name),
            'fetched_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Store with both full name and base name
        self.model_metadata[model_name] = cached_entry
        if base_name != model_name:
            # Also store with base name (but don't overwrite if full name already exists)
            if base_name not in self.model_metadata:
                base_entry = dict(cached_entry)
                base_entry['name'] = base_name
                self.model_metadata[base_name] = base_entry
    
    def get_model_from_cache(self, model_name: str) -> dict | None:
        """Get cached model metadata by name.
        
        Args:
            model_name: Model name to look up (full or base name)
            
        Returns:
            dict: Cached model metadata, or None if not found
        """
        # Direct lookup
        if model_name in self.model_metadata:
            return self.model_metadata[model_name]
        
        # Try base name
        base_name = model_name.split(':')[0]
        if base_name in self.model_metadata:
            return self.model_metadata[base_name]
        
        # Try dev-* prefix removal
        for prefix in ['dev-', '+']:
            if base_name.startswith(prefix):
                clean_name = base_name[len(prefix):]
                if clean_name in self.model_metadata:
                    return self.model_metadata[clean_name]
        
        return None
    
    def _extract_params(self, metadata: dict) -> str:
        """Extract human-readable params string from model metadata."""
        details = metadata.get('details', {})
        if details:
            return details.get('parameter_size', 'N/A')
        return 'N/A'
    
    def _extract_total_params(self, metadata: dict) -> str:
        """Extract total parameters count."""
        details = metadata.get('details', {})
        if details:
            return str(details.get('total_parameters', 'N/A'))
        return 'N/A'
    
    def _extract_active_params(self, metadata: dict) -> str:
        """Extract active parameters count (for MoE models)."""
        details = metadata.get('details', {})
        if details:
            # For MoE models, this might be available
            moe_info = metadata.get('moe', {})
            if moe_info:
                num_experts = moe_info.get('num_experts', 0)
                experts_per_token = moe_info.get('experts_per_token', 0)
                total = details.get('total_parameters', 0)
                if num_experts > 0 and experts_per_token > 0 and total > 0:
                    # Active params = total * experts_per_token / num_experts
                    active = total * experts_per_token / num_experts
                    return self._format_params(active)
            return str(details.get('total_parameters', 'N/A'))
        return 'N/A'
    
    def _extract_quant(self, metadata: dict) -> str:
        """Extract quantization format."""
        details = metadata.get('details', {})
        if details:
            return details.get('quantization_format', 'N/A')
        return 'N/A'
    
    def _extract_max_ctx(self, metadata: dict) -> int:
        """Extract maximum context length."""
        max_ctx = 131072  # default
        
        # Check parameters field
        parameters = metadata.get('parameters', '')
        if parameters:
            for line in parameters.split('\n'):
                line = line.strip()
                if line.startswith('num_ctx:'):
                    try:
                        max_ctx = int(line.split(':')[1].strip())
                        return max_ctx
                    except (ValueError, IndexError):
                        pass
        
        # Check model_info keys
        model_info = metadata.get('model_info', {})
        for key, val in model_info.items():
            if 'context_length' in key.lower() or 'num_ctx' in key.lower():
                try:
                    max_ctx = int(val)
                    return max_ctx
                except (ValueError, TypeError):
                    pass
        
        return max_ctx
    
    def _extract_capabilities(self, metadata: dict) -> dict:
        """Extract capabilities from model metadata."""
        # Check top-level capabilities field (new Ollama API format)
        caps_list = metadata.get('capabilities', [])
        if isinstance(caps_list, list) and len(caps_list) > 0:
            caps_str = ' '.join(caps_list).lower()
            return {
                'vision': 'vision' in caps_str or 'image' in caps_str,
                'tools': 'tools' in caps_str or 'function' in caps_str,
                'thinking': 'thinking' in caps_str or 'reason' in caps_str,
                'audio': 'audio' in caps_str or 'whisper' in caps_str
            }
        
        # Fallback: extract from model_info
        model_info = metadata.get('model_info', {})
        from api.base_client import BaseApiClient
        return BaseApiClient.get_capabilities_from_model_info(model_info)
    
    # ========================================================================
    # MoE (Mixture of Experts) Detection
    # ========================================================================
    
    def _detect_moe(self, metadata: dict, base_name: str) -> dict | bool | None:
        """Detect if model is MoE and extract MoE details.
        
        Returns:
            dict: MoE details if confirmed MoE (e.g., {'is_moe': True, 'num_experts': 64, ...})
            bool: False if confirmed NOT MoE
            None: Unknown (insufficient data)
        """
        model_info = metadata.get('model_info', {})
        details = metadata.get('details', {})
        
        # Method 1: Direct MoE fields in model_info
        moe_fields = {}
        for key, val in model_info.items():
            if key.startswith('moe.'):
                moe_fields[key] = val
        
        if moe_fields:
            result = {'is_moe': True}
            if 'moe.num_experts' in moe_fields:
                try:
                    result['num_experts'] = int(moe_fields['moe.num_experts'])
                except (ValueError, TypeError):
                    pass
            if 'moe.experts_per_token' in moe_fields:
                try:
                    result['experts_per_token'] = int(moe_fields['moe.experts_per_token'])
                except (ValueError, TypeError):
                    pass
            if 'moe.router' in moe_fields:
                result['router'] = moe_fields['moe.router']
            
            # Calculate active parameters if we have total params
            total_params = details.get('total_parameters', 0)
            if total_params and result.get('num_experts', 0) > 0:
                experts_per_token = result.get('experts_per_token', 1)
                active = total_params * experts_per_token / result['num_experts']
                result['active_params'] = self._format_params(active)
                result['total_params'] = self._format_params(total_params)
            
            return result
        
        # Method 2: Architecture-based detection
        architecture = model_info.get('general.architecture', '').lower()
        basename = model_info.get('general.basename', '').lower()
        combined = f"{architecture} {basename}".lower()
        
        # Known MoE architectures
        moe_architectures = ['mixtral', 'qwen2.5-moe', 'deepseek-v3', 'deepseek-v2', 'jamba']
        for arch in moe_architectures:
            if arch in combined:
                return {'is_moe': True, 'architecture': architecture}
        
        # Known dense (non-MoE) architectures
        dense_architectures = ['llama', 'gemma', 'qwen', 'mistral', 'phi', 'granite']
        for arch in dense_architectures:
            if arch in combined and 'moe' not in basename:
                return False
        
        # Method 3: Name-based heuristics
        name_lower = base_name.lower()
        
        # Known MoE model patterns
        moe_patterns = ['mixtral', 'qwen2.5-moe', 'deepseek-v3', 'deepseek-v2', 'jamba']
        for pattern in moe_patterns:
            if pattern in name_lower:
                return {'is_moe': True, 'detected_by': 'name_pattern'}
        
        # Known dense model patterns
        dense_patterns = ['llama', 'gemma', 'qwen3', 'qwen2.5', 'phi', 'granite']
        for pattern in dense_patterns:
            if pattern in name_lower and 'moe' not in name_lower:
                return False
        
        # Method 4: Check for MoE-related keys in model_info
        moe_keywords = ['moe', 'mixture', 'expert', 'router']
        for key in model_info.keys():
            if any(kw in key.lower() for kw in moe_keywords):
                return {'is_moe': True, 'detected_by': 'key_pattern', 'key': key}
        
        # Unknown - return None
        return None
    
    def _format_params(self, num_params: int | float) -> str:
        """Format parameter count for display."""
        if num_params >= 1e12:
            return f"{num_params / 1e12:.1f}T"
        elif num_params >= 1e9:
            return f"{num_params / 1e9:.1f}B"
        elif num_params >= 1e6:
            return f"{num_params / 1e6:.1f}M"
        elif num_params >= 1e3:
            return f"{num_params / 1e3:.1f}K"
        return str(int(num_params))
    
    # ========================================================================
    # Legacy compatibility methods (keep for backward compatibility)
    # ========================================================================
    
    def _is_size_tag(self, text: str) -> bool:
        """Check if text is a model size indicator (e.g., '1.5b', '70b', '7b')."""
        if not text:
            return False
        # Remove any quantization info
        text = text.split('-')[0].strip()
        return bool(self.SIZE_PATTERN.match(text)) or bool(self.COLOR_PATTERN.match(text))

    def get_capabilities_from_html(self, base_name: str) -> tuple:
        """Get capabilities by parsing the model page HTML.
        
        Returns:
            tuple: (vision, tools, thinking) - capability statuses, or None if not found
        """
        url = f"https://ollama.com/library/{base_name}"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Primary method: Find capabilities in the div with size tags
        # Capabilities are in the same div.flex-wrap as size tags, but as colored badges
        # Size tags have bg-[#color] classes, capabilities don't
        flex_wraps = soup.find_all('div', class_=lambda c: c and 'flex-wrap' in c)
        
        for div in flex_wraps:
            spans = div.find_all('span')
            if len(spans) < 2:
                continue
            
            # Check if this div contains size indicators
            has_size = False
            for span in spans:
                text = span.get_text(strip=True).lower()
                if self._is_size_tag(text):
                    has_size = True
                    break
            
            if not has_size:
                continue
            
            # This is the capabilities+sizes div
            # Capabilities are plain text spans (vision, tools, thinking)
            # Sizes are spans with bg-[*] color classes
            found_vision = False
            found_tools = False
            found_thinking = False
            
            for span in spans:
                text = span.get_text(strip=True).lower()
                classes = ' '.join(span.get('class', []))
                
                # Skip size tags (they have bg- color classes)
                if 'bg-' in classes or self._is_size_tag(text):
                    continue
                
                # Skip if this looks like a version number
                if re.match(r'^\d+(\.\d+)*$', text):
                    continue
                
                # Check for capability keywords
                if text == 'vision' or 'multimodal' in text:
                    found_vision = True
                elif text == 'tools' or 'tool use' in text:
                    found_tools = True
                elif text in ('thinking', 'reasoning'):
                    found_thinking = True
            
            if found_vision or found_tools or found_thinking:
                return (
                    "✅" if found_vision else "❌",
                    "✅" if found_tools else "❌",
                    "✅" if found_thinking else "❌"
                )
        
        # Fallback: check meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            desc = meta_desc.get('content', '').lower()
            vision = "✅" if any(w in desc for w in ['vision', 'visual', 'multimodal', 'image']) else "❌"
            tools = "✅" if any(w in desc for w in ['tool', 'function', 'api']) else "❌"
            thinking = "✅" if any(w in desc for w in ['reasoning', 'think', 'chain of thought']) else "❌"
            if vision != "❌" or tools != "❌" or thinking != "❌":
                return vision, tools, thinking
        
        return None  # Signal that HTML parsing didn't find anything definitive

    def get_capabilities_from_model_db(self, base_name: str) -> tuple:
        """Get capabilities from the built-in model knowledge base or cache.
        
        Returns:
            tuple: (vision, tools, thinking) or None if not found
        """
        # Direct lookup
        if base_name in self.MODEL_CAPABILITIES:
            caps = self.MODEL_CAPABILITIES[base_name]
            return (
                "✅" if caps['vision'] else "❌",
                "✅" if caps['tools'] else "❌",
                "✅" if caps['thinking'] else "❌"
            )
        
        # Pattern matching for variants
        # dev-* models - strip the prefix
        clean_name = base_name
        for prefix in ['dev-', '+']:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]
                break
        
        if clean_name != base_name and clean_name in self.MODEL_CAPABILITIES:
            caps = self.MODEL_CAPABILITIES[clean_name]
            return (
                "✅" if caps['vision'] else "❌",
                "✅" if caps['tools'] else "❌",
                "✅" if caps['thinking'] else "❌"
            )
        
        # Check for deepseek variants (they have thinking)
        if 'deepseek' in base_name.lower():
            if 'r1' in base_name.lower() or 'qwq' in base_name.lower():
                return "❌", "✅", "✅"
            return "❌", "✅", "❌"
        
        # Check for qwen2.5/qwen3 variants with thinking (qwq has thinking)
        if 'qwq' in base_name.lower():
            return "❌", "✅", "✅"
        
        # Check for vision models by name patterns
        if any(kw in base_name.lower() for kw in ['llava', 'moondream', 'phi.3.v', 'phi.3-v']):
            return "✅", "❌", "❌"
        
        return None

    def get_capabilities(self, model_name: str) -> tuple:
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
                    'vision': result[0] == "✅",
                    'tools': result[1] == "✅",
                    'thinking': result[2] == "✅"
                }
                return result
        except Exception as e:
            print(f"⚠️  Error parsing HTML for {base_name}: {e}")

        # Method 3: Fallback - try search API
        try:
            search_url = f"https://ollama.com/api/search?q={urllib.parse.quote(base_name)}"
            search_resp = requests.get(search_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            if search_resp.status_code == 200:
                search_data = search_resp.json()
                for item in search_data.get("results", []):
                    if item.get("name") == base_name:
                        caps = item.get("capabilities", {})
                        vision = "✅" if caps.get("vision") else "❌"
                        tools = "✅" if caps.get("tools") else "❌"
                        thinking = "✅" if caps.get("thinking") else "❌"
                        # Cache the result
                        self.MODEL_CAPABILITIES[base_name] = {
                            'vision': vision == "✅",
                            'tools': tools == "✅",
                            'thinking': thinking == "✅"
                        }
                        return vision, tools, thinking
        except Exception as e:
            print(f"⚠️  Error searching for {base_name}: {e}")

        # Final fallback: heuristic based on model name patterns
        return self._heuristic_capabilities(base_name)

    def _heuristic_capabilities(self, base_name: str) -> tuple:
        """Determine capabilities using heuristics based on model name patterns.
        
        This is the last resort when no other method succeeds.
        """
        name_lower = base_name.lower()
        
        # Vision models
        vision_models = ['llava', 'moondream', 'neural-chat', 'all-minilm']
        if any(kw in name_lower for kw in vision_models):
            return "✅", "❌", "❌"
        
        # Models with both vision and tools (llama3.2 family)
        if 'llama3.2' in name_lower:
            return "✅", "✅", "❌"
        
        # Reasoning models
        reasoning_models = ['deepseek-r1', 'deepscaler', 'qwq', 'qwq-preview', 'o1', 'o3']
        if any(kw in name_lower for kw in reasoning_models):
            return "❌", "✅", "✅"
        
        # Models with tools (most modern models)
        tool_models = ['qwen2.5', 'qwen2', 'llama3.1', 'llama3', 'mistral', 'mixtral', 
                       'codestral', 'codex', 'dolphin', 'neural-chat', 'zephyr']
        if any(kw in name_lower for kw in tool_models):
            return "❌", "✅", "❌"
        
        # Older models or specific variants without tools
        return "❌", "❌", "❌"
