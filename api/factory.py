"""API client factory for creating local or remote clients based on configuration."""

from typing import Optional

from api.local_client import LocalApiClient
from api.remote_client import RemoteApiClient
from system.ssh_client import SSHClient


class ApiClientFactory:
    """Factory for creating appropriate API client based on configuration."""

    @staticmethod
    def is_remote_config(ssh_host: Optional[str] = None) -> bool:
        """Check if SSH configuration indicates remote mode.

        Args:
            ssh_host: SSH host from CLI arguments

        Returns:
            bool: True if remote configuration is detected
        """
        return bool(ssh_host)

    @staticmethod
    def create_client(
        base_url: str,
        headers: Optional[dict] = None,
        timeout: int = 300,
        ssh_host: Optional[str] = None,
        ssh_user: Optional[str] = None,
        ssh_port: int = 22,
        ssh_key: Optional[str] = None
    ):
        """Create the appropriate API client based on SSH configuration.

        Args:
            base_url: Ollama API base URL (used for HTTP requests)
            headers: HTTP headers (e.g., authentication)
            timeout: Request timeout in seconds
            ssh_host: SSH host for remote mode (used only for SSH commands)
            ssh_user: SSH user for remote mode
            ssh_port: SSH port for remote mode
            ssh_key: Path to SSH private key

        Returns:
            BaseApiClient: Either LocalApiClient or RemoteApiClient
        """
        if ApiClientFactory.is_remote_config(ssh_host):
            ssh_client = SSHClient(
                host=ssh_host,
                user=ssh_user,
                port=ssh_port,
                key_path=ssh_key
            )
            return RemoteApiClient(
                base_url=base_url,
                headers=headers,
                timeout=timeout,
                ssh_client=ssh_client
            )
        else:
            return LocalApiClient(
                base_url=base_url,
                headers=headers,
                timeout=timeout
            )
