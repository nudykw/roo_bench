"""API-based metadata extraction methods for CapabilitiesFetcher."""

import datetime
import json
import os
from typing import Any, cast

from api.base_client import BaseApiClient
from api.capabilities_defaults import (
    DEFAULT_CAPABILITIES,
    DENSE_ARCHITECTURES,
    DENSE_PATTERNS,
    MOE_ARCHITECTURES,
    MOE_KEYWORDS,
    MOE_PATTERNS,
)

# Type alias for capabilities tuple
CapabilitiesTuple = tuple[str, str, str]


class CapabilitiesApi:
    """Mixin class for API-based methods."""

    cache_file: str
    model_cache_file: str
    CACHE_VERSION: int
    CACHE_TTL_HOURS: int
    MODEL_CAPABILITIES: dict[str, Any]
    model_metadata: dict[str, Any]

    def __init__(self, cache_file: str | None = None, model_cache_file: str | None = None):
        """Initialize CapabilitiesFetcher with cache.
        
        Args:
            cache_file: Path to capabilities JSON cache file. Defaults to data/capabilities_cache.json
            model_cache_file: Path to model metadata JSON cache file. Defaults to data/model_cache.json
        """
        self.cache_file = cache_file or os.path.join(
            os.path.dirname(__file__), '..', 'data', 'capabilities_cache.json'
        )
        self.model_cache_file = model_cache_file or os.path.join(
            os.path.dirname(__file__), '..', 'data', 'model_cache.json'
        )
        # Start with default capabilities, then merge with cached data
        self.MODEL_CAPABILITIES = dict(DEFAULT_CAPABILITIES)
        self._load_cache()
        # Model metadata cache (full data from /api/show)
        self.model_metadata = {}  # type: ignore[assignment]
        self._load_model_cache()

    def is_cache_fresh(self) -> bool:
        """Check if the model cache is fresh (not expired).
        
        Returns:
            bool: True if cache exists and was fetched within CACHE_TTL_HOURS
        """
        if not os.path.exists(self.model_cache_file):
            return False
        
        try:
            with open(self.model_cache_file, encoding='utf-8') as f:
                cache_data = json.load(f)
            
            cache_timestamp = cache_data.get('cache_timestamp', '')
            if not cache_timestamp:
                return False
            
            # Parse timestamp and check TTL
            fetched_at = datetime.datetime.fromisoformat(cache_timestamp.replace('Z', '+00:00'))
            now = datetime.datetime.now(datetime.timezone.utc)
            age_hours = (now - fetched_at).total_seconds() / 3600
            
            return age_hours < self.CACHE_TTL_HOURS
        except (OSError, json.JSONDecodeError, ValueError):
            return False

    def _load_model_cache(self) -> None:
        """Load cached model metadata from JSON file."""
        if not os.path.exists(self.model_cache_file):
            return
        
        try:
            with open(self.model_cache_file, encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Validate cache version
            version = cache_data.get('cache_version', 0)
            if version < self.CACHE_VERSION:
                print(f"\u26a0\ufe0f  Cache version {version} is outdated (need {self.CACHE_VERSION})")
                return
            
            self.model_metadata = cache_data.get('models', {})  # type: ignore[assignment]
            timestamp = cache_data.get('cache_timestamp', 'unknown')
            print(f"\u2705 Loaded model cache: {len(self.model_metadata)} models from {timestamp} ({self.model_cache_file})")
        except (OSError, json.JSONDecodeError) as e:
            print(f"\u26a0\ufe0f  Error loading model cache: {e}")

    def _load_cache(self) -> None:
        """Load cached capabilities from JSON file."""
        if not os.path.exists(self.cache_file):
            return
        
        try:
            with open(self.cache_file, encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Merge cached data with defaults (cache takes precedence)
            if isinstance(cached_data, dict):
                self.MODEL_CAPABILITIES.update(cached_data)
                print(f"\u2705 Loaded cache: {len(cached_data)} models from {self.cache_file}")
        except (OSError, json.JSONDecodeError) as e:
            print(f"\u26a0\ufe0f  Error loading cache: {e}")

    def add_model_from_api(self, model_name: str, capabilities: dict[str, Any]) -> None:
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

    def save_cache(self) -> None:
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
            cached_only = {
                k: v for k, v in self.MODEL_CAPABILITIES.items()
                if k not in DEFAULT_CAPABILITIES
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cached_only, f, indent=2, ensure_ascii=False)
            
            if cached_only:
                print(f"\U0001f4be Saved cache: {len(cached_only)} new models to {self.cache_file}")
            else:
                print("\u2139\ufe0f  No new models to save (all models are in default database)")
        except OSError as e:
            print(f"\u26a0\ufe0f  Error saving cache: {e}")

    def save_model_cache(self) -> None:
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
                'cache_timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'models': self.model_metadata
            }
            
            with open(self.model_cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            print(f"\U0001f4be Saved model cache: {len(self.model_metadata)} models to {self.model_cache_file}")
        except OSError as e:
            print(f"\u26a0\ufe0f  Error saving model cache: {e}")

    def add_model_metadata(self, model_name: str, metadata: dict[str, Any]) -> None:
        """Add or update model metadata in the cache.
        
        Args:
            model_name: Full model name (e.g., 'qwen3.6-35b:q4_0')
            metadata: Dict containing model metadata from /api/show
        """
        base_name = model_name.split(':')[0]
        
        # Extract and normalize all relevant fields
        raw_size = metadata.get('size', 0)
        cached_entry = {
            'name': model_name,
            'base_name': base_name,
            'size': raw_size,
            'size_gb': round(raw_size / (1024**3), 1) if raw_size > 0 else "N/A",
            'params': self._extract_params(metadata),
            'total_params': self._extract_total_params(metadata),
            'active_params': self._extract_active_params(metadata),
            'quant': self._extract_quant(metadata),
            'max_ctx': self._extract_max_ctx(metadata),
            'architecture': self._extract_architecture(metadata),
            'capabilities': self._extract_capabilities(metadata),
            'moe': self._detect_moe(metadata, base_name),
            'fetched_at': datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        
        # Store with both full name and base name
        self.model_metadata[model_name] = cached_entry
        if base_name != model_name:
            # Also store with base name (but don't overwrite if full name already exists)
            if base_name not in self.model_metadata:
                base_entry = dict(cached_entry)
                base_entry['name'] = base_name
                self.model_metadata[base_name] = base_entry

    def get_model_from_cache(self, model_name: str) -> dict[str, Any] | None:
        """Get cached model metadata by name.
        
        Args:
            model_name: Model name to look up (full or base name)
            
        Returns:
            dict: Cached model metadata, or None if not found
        """
        # Direct lookup
        if model_name in self.model_metadata:
            return cast(dict[str, Any] | None, self.model_metadata[model_name])
        
        # Try base name
        base_name = model_name.split(':')[0]
        if base_name in self.model_metadata:
            return cast(dict[str, Any] | None, self.model_metadata[base_name])
        
        # Try dev-* prefix removal
        for prefix in ['dev-', '+']:
            if base_name.startswith(prefix):
                clean_name = base_name[len(prefix):]
                if clean_name in self.model_metadata:
                    return cast(dict[str, Any] | None, self.model_metadata[clean_name])
        
        return None

    def _extract_params(self, metadata: dict[str, Any]) -> str:
        """Extract human-readable params string from model metadata."""
        details = metadata.get('details', {})
        if details:
            val = details.get('parameter_size', 'N/A')
            return str(val) if val is not None else 'N/A'
        return 'N/A'

    def _extract_total_params(self, metadata: dict[str, Any]) -> str:
        """Extract total parameters count."""
        details = metadata.get('details', {})
        if details:
            return str(details.get('total_parameters', 'N/A'))
        return 'N/A'

    def _extract_active_params(self, metadata: dict[str, Any]) -> str:
        """Extract active parameters count (for MoE models)."""
        details = metadata.get('details', {})
        if details:
            moe_info = metadata.get('moe', {})
            if moe_info:
                num_experts = moe_info.get('num_experts', 0)
                experts_per_token = moe_info.get('experts_per_token', 0)
                total = details.get('total_parameters', 0)
                if num_experts > 0 and experts_per_token > 0 and total > 0:
                    active = total * experts_per_token / num_experts
                    return self._format_params(active)
            return str(details.get('total_parameters', 'N/A'))
        return 'N/A'

    def _extract_quant(self, metadata: dict[str, Any]) -> str:
        """Extract quantization format."""
        details = metadata.get('details', {})
        if details:
            quant = details.get('quantization_level', 'N/A')
            if quant and quant != 'N/A':
                return str(quant)
            qf = details.get('quantization_format', 'N/A')
            return str(qf) if qf is not None else 'N/A'
        return 'N/A'

    def _extract_architecture(self, metadata: dict[str, Any]) -> str:
        """Extract model architecture from model_info."""
        model_info = metadata.get('model_info', {})
        architecture = model_info.get('general.architecture', '')
        return architecture if architecture else 'N/A'

    def _extract_max_ctx(self, metadata: dict[str, Any]) -> int:
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

    def _extract_capabilities(self, metadata: dict[str, Any]) -> dict[str, bool]:
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
        return BaseApiClient.get_capabilities_from_model_info(model_info)

    def _detect_moe(self, metadata: dict[str, Any], base_name: str) -> dict[str, Any] | bool | None:
        """Detect if model is MoE and extract MoE details.
        
        Returns:
            dict: MoE details if confirmed MoE
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
            result: dict[str, Any] = {'is_moe': True}
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
        
        if architecture.endswith('moe'):
            return {'is_moe': True, 'architecture': architecture}
        
        for arch in MOE_ARCHITECTURES:
            if arch in combined:
                return {'is_moe': True, 'architecture': architecture}
        
        for arch in DENSE_ARCHITECTURES:
            if arch in combined and 'moe' not in basename:
                return False
        
        # Method 3: Name-based heuristics
        name_lower = base_name.lower()
        
        for pattern in MOE_PATTERNS:
            if pattern in name_lower:
                return {'is_moe': True, 'detected_by': 'name_pattern'}
        
        for pattern in DENSE_PATTERNS:
            if pattern in name_lower and 'moe' not in name_lower:
                return False
        
        # Method 4: Check for MoE-related keys in model_info
        for key in model_info.keys():
            if any(kw in key.lower() for kw in MOE_KEYWORDS):
                return {'is_moe': True, 'detected_by': 'key_pattern', 'key': key}
        
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
