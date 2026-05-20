"""Base API client for Ollama API interactions."""

import logging
import threading
from abc import ABC, abstractmethod
from typing import Any

from api.base_client_generate import BaseApiClientGenerate
from api.base_client_models import BaseApiClientModels
from api.base_client_utils import BaseApiClientUtils

logger = logging.getLogger('roo_bench')


class BaseApiClient(
    BaseApiClientGenerate,
    BaseApiClientModels,
    BaseApiClientUtils,
    ABC
):
    """Abstract base class for API clients."""

    base_url: str
    headers: dict[str, Any]
    timeout: int
    ssh_client: Any

    def __init__(self, base_url: str, headers: dict[str, Any] | None = None, timeout: int = 300):
        """Initialize base API client.

        Args:
            base_url: Ollama API base URL
            headers: HTTP headers (e.g., authentication)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.headers = headers or {}
        self.timeout = timeout

    @property
    @abstractmethod
    def is_remote(self) -> bool:
        """Return True if this is a remote client with SSH."""
        pass

    @abstractmethod
    def _monitor_vram(self, stop_event: threading.Event, max_vram_ref: list[float], vram_samples: list[float]) -> None:
        """Internal VRAM monitoring implementation."""
        pass

    @staticmethod
    def get_capabilities_from_model_info(model_info: dict[str, Any]) -> dict[str, bool]:
        """Extract capabilities from Ollama API /api/show model_info.

        Uses both direct key detection and architecture-based heuristics.

        Args:
            model_info: Dictionary from model_info field of /api/show response

        Returns:
            dict: {'vision': bool, 'tools': bool, 'thinking': bool, 'audio': bool}
        """
        capabilities = {
            'vision': False,
            'tools': False,
            'thinking': False,
            'audio': False
        }

        # Get architecture and basename for heuristic detection
        architecture = model_info.get('general.architecture', '').lower()
        basename = model_info.get('general.basename', '').lower()
        file_type = model_info.get('general.file_type', '').lower()

        # Combine all strings for easier searching
        combined = f"{architecture} {basename} {file_type}"
        combined = combined.lower()

        # Vision detection
        if any(kw in combined for kw in ['vision', 'instruct', 'clip', 'image', 'multi modal', 'moar']):
            capabilities['vision'] = True
        elif any(kw in architecture for kw in ['clip', 'vlm', 'moondream']):
            capabilities['vision'] = True
        # Check for vision-specific keys
        for key in model_info.keys():
            if any(vision_kw in key.lower() for vision_kw in ['vision', 'image', 'clip', 'multi modal']):
                capabilities['vision'] = True
                break

        # Tools/function calling detection
        if any(kw in combined for kw in ['tools', 'function', 'tool', 'function calling']):
            capabilities['tools'] = True
        elif 'tokenizer.tools' in str(model_info.get('tokenizer', {})):
            capabilities['tools'] = True
        # Check for tools-specific keys
        for key in model_info.keys():
            if 'tools' in key.lower() or 'function' in key.lower():
                capabilities['tools'] = True
                break

        # Thinking/reasoning detection
        if any(kw in combined for kw in ['thinking', 'reason', 'deepseek', 'o1', 'o3']):
            capabilities['thinking'] = True
        elif any(kw in str(model_info.get('tokenizer', {})).lower() for kw in ['thinking', 'reason']):
            capabilities['thinking'] = True
        # Check for thinking-specific keys
        for key in model_info.keys():
            if 'thinking' in key.lower() or 'reason' in key.lower():
                capabilities['thinking'] = True
                break

        # Audio detection
        if any(kw in combined for kw in ['audio', 'whisper', 'speech', 'voice']):
            capabilities['audio'] = True
        elif any(kw in str(model_info.get('tokenizer', {})).lower() for kw in ['audio', 'whisper']):
            capabilities['audio'] = True
        # Check for audio-specific keys
        for key in model_info.keys():
            if any(audio_kw in key.lower() for audio_kw in ['audio', 'whisper', 'speech', 'voice']):
                capabilities['audio'] = True
                break

        return capabilities
