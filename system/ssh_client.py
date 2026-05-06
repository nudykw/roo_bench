"""Centralized SSH client for remote operations."""

import os
import subprocess
from typing import Optional


class SSHClient:
    """Centralized SSH client for all remote operations."""

    def __init__(self, host: str = None, user: str = None,
                 port: int = 22, key_path: str = None):
        """Initialize SSH client.

        Args:
            host: SSH host (can be 'user@host' or just 'host')
            user: SSH user (optional if user@host format used)
            port: SSH port
            key_path: Path to SSH private key (auto-detected if not specified)
        """
        self.ssh_host = host
        self.ssh_user = user
        self.ssh_port = port
        self.ssh_key = key_path

    @property
    def is_configured(self) -> bool:
        """Check if SSH client has valid configuration."""
        return bool(self.ssh_host)

    def _find_default_ssh_key(self) -> Optional[str]:
        """Find default SSH private key in ~/.ssh/.

        Returns:
            Path to SSH key or None if not found
        """
        ssh_dir = os.path.expanduser("~/.ssh")

        # Priority order for default keys
        candidates = [
            "id_ed25519",
            "id_rsa",
            "id_dsa",
            "id_ecdsa",
        ]

        for candidate in candidates:
            key_path = os.path.join(ssh_dir, candidate)
            if os.path.exists(key_path):
                return key_path

        # Fallback: first private key found
        for f in os.listdir(ssh_dir):
            if f.startswith("id_") and not f.endswith(".pub"):
                return os.path.join(ssh_dir, f)

        return None

    def _resolve_key(self) -> Optional[str]:
        """Resolve SSH key path.

        Returns:
            Path to SSH key or None
        """
        return self.ssh_key or self._find_default_ssh_key()

    def _resolve_target(self) -> str:
        """Resolve SSH target (user@host).

        Returns:
            Target string in user@host format
        """
        target = self.ssh_host
        if self.ssh_user and '@' not in self.ssh_host:
            target = f"{self.ssh_user}@{self.ssh_host}"
        return target

    def build_command(self, remote_command: str) -> list:
        """Build complete SSH command.

        Args:
            remote_command: Command to execute on remote host

        Returns:
            Full SSH command as list
        """
        cmd = ['ssh', '-T', '-o', 'StrictHostKeyChecking=no',
               '-o', 'ConnectTimeout=10']

        if self.ssh_port != 22:
            cmd.extend(['-p', str(self.ssh_port)])

        key = self._resolve_key()
        if key:
            cmd.extend(['-i', key])

        cmd.append(self._resolve_target())
        cmd.append(remote_command)
        return cmd

    def execute(self, remote_command: str, timeout: int = 30,
                capture_output: bool = True) -> subprocess.CompletedProcess:
        """Execute command on remote host via SSH.

        Args:
            remote_command: Command to execute
            timeout: Timeout in seconds
            capture_output: Whether to capture stdout/stderr

        Returns:
            CompletedProcess instance
        """
        cmd = self.build_command(remote_command)
        return subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            timeout=timeout
        )

    def get_vram_usage(self, timeout: int = 30) -> Optional[int]:
        """Get VRAM usage from remote GPU via SSH.

        Returns:
            VRAM usage in bytes, or None if unavailable
        """
        if not self.is_configured:
            return None
            
        cmd = self.build_command(
            "nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader"
        )
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            if result.returncode == 0 and result.stdout.strip():
                output = result.stdout.strip()
                # nvidia-smi returns value in MiB, convert to bytes
                try:
                    vram_mib = int(output)
                    return vram_mib * 1024 * 1024
                except ValueError:
                    return None
            else:
                # Log error output for debugging
                if result.stderr:
                    pass  # print(f"DEBUG: SSH VRAM error: {result.stderr}")
        except subprocess.TimeoutExpired:
            pass  # print("DEBUG: SSH VRAM timeout")
        except Exception:
            pass  # print(f"DEBUG: SSH VRAM exception")
        return None

    def restart_ollama(self, timeout: int = 30) -> bool:
        """Restart Ollama on remote host via SSH.

        Args:
            timeout: Timeout in seconds

        Returns:
            True if successful, False otherwise
        """
        result = self.execute("sudo systemctl restart ollama", timeout=timeout)
        return result.returncode == 0
