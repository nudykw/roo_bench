"""API client factory for creating local or remote clients based on configuration."""


from api.llama_cpp_client import LlamaCppApiClient, LlamaCppRemoteApiClient
from api.local_client import LocalApiClient
from api.remote_client import RemoteApiClient
from system.ssh_client import SSHClient


class ApiClientFactory:
    """Factory for creating appropriate API client based on configuration."""

    @staticmethod
    def is_remote_config(ssh_host: str | None = None) -> bool:
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
        headers: dict | None = None,
        timeout: int = 300,
        backend_type: str = "ollama",
        ssh_host: str | None = None,
        ssh_user: str | None = None,
        ssh_port: int = 22,
        ssh_key: str | None = None,
    ):
        """Create the appropriate API client based on backend type and SSH config.

        Args:
            base_url: API base URL
            headers: HTTP headers
            timeout: Request timeout in seconds
            backend_type: "ollama" or "llama_cpp"
            ssh_host: SSH host for remote mode
            ssh_user: SSH user
            ssh_port: SSH port
            ssh_key: Path to SSH private key

        Returns:
            BaseApiClient: Appropriate client for the configured backend
        """
        is_remote = ApiClientFactory.is_remote_config(ssh_host)

        if backend_type == "llama_cpp":
            if is_remote:
                ssh_client = SSHClient(
                    host=ssh_host,
                    user=ssh_user,
                    port=ssh_port,
                    key_path=ssh_key,
                )
                return LlamaCppRemoteApiClient(
                    base_url=base_url,
                    headers=headers,
                    timeout=timeout,
                    ssh_client=ssh_client,
                )
            else:
                return LlamaCppApiClient(
                    base_url=base_url,
                    headers=headers,
                    timeout=timeout,
                )
        else:
            # Default: Ollama backend
            if is_remote:
                ssh_client = SSHClient(
                    host=ssh_host,
                    user=ssh_user,
                    port=ssh_port,
                    key_path=ssh_key,
                )
                return RemoteApiClient(
                    base_url=base_url,
                    headers=headers,
                    timeout=timeout,
                    ssh_client=ssh_client,
                )
            else:
                return LocalApiClient(
                    base_url=base_url,
                    headers=headers,
                    timeout=timeout,
                )
