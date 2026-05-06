"""Remote API client for Ollama API interactions with SSH."""

import threading

from api.base_client import BaseApiClient
from system.ssh_client import SSHClient


class RemoteApiClient(BaseApiClient):
    """Remote API client with SSH for remote operations."""

    def __init__(self, base_url: str, headers: dict = None, timeout: int = 300,
                 ssh_client: SSHClient = None):
        """Initialize remote API client.

        Args:
            base_url: Ollama API base URL
            headers: HTTP headers (e.g., authentication)
            timeout: Request timeout in seconds
            ssh_client: SSHClient instance for remote VRAM monitoring
        """
        super().__init__(base_url, headers, timeout)
        self.ssh_client = ssh_client

    @property
    def is_remote(self) -> bool:
        """Return True if this is a remote client with SSH."""
        return True

    def _monitor_vram(self, stop_event: threading.Event, max_vram_ref: list, vram_samples: list):
        """Monitor remote VRAM during generation via SSH."""
        while not stop_event.is_set():
            v = self.ssh_client.get_vram_usage() if self.ssh_client else None
            if v is not None:
                vram_samples.append(v)
                if v > max_vram_ref[0]:
                    max_vram_ref[0] = v
            stop_event.wait(0.5)
