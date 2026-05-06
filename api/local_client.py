"""Local API client for Ollama API interactions without SSH."""

import threading

from api.base_client import BaseApiClient
from system.gpu_monitor import get_vram_usage


class LocalApiClient(BaseApiClient):
    """Local API client without SSH."""

    def __init__(self, base_url: str, headers: dict = None, timeout: int = 300):
        """Initialize local API client.

        Args:
            base_url: Ollama API base URL
            headers: HTTP headers (e.g., authentication)
            timeout: Request timeout in seconds
        """
        super().__init__(base_url, headers, timeout)
        self.ssh_client = None

    @property
    def is_remote(self) -> bool:
        """Return True if this is a remote client with SSH."""
        return False

    def _monitor_vram(self, stop_event: threading.Event, max_vram_ref: list, vram_samples: list):
        """Monitor local VRAM during generation."""
        while not stop_event.is_set():
            v = get_vram_usage()
            if v is not None:
                vram_samples.append(v)
                if v > max_vram_ref[0]:
                    max_vram_ref[0] = v
            stop_event.wait(0.2)
