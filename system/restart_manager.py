"""Ollama service restart logic."""

import os
import glob
import subprocess
import time
from enum import Enum
from i18n import get_text


class RestartMethod(Enum):
    """Ollama restart methods."""
    SYSTEMCTL = "systemctl"
    DOCKER = "docker"
    KILL_START = "kill_start"
    MANUAL = "manual"
    SSH = "ssh"


class RestartManager:
    """Manages Ollama service restart operations."""

    def __init__(self, method: RestartMethod = RestartMethod.SYSTEMCTL, no_restart: bool = False,
                 ssh_host: str = None, ssh_user: str = None, ssh_port: int = 22, ssh_key: str = None):
        """Initialize restart manager.

        Args:
            method: Restart method (SYSTEMCTL, DOCKER, KILL_START, MANUAL, SSH)
            no_restart: If True, restart is not performed
            ssh_host: SSH host (for SSH method)
            ssh_user: SSH user (for SSH method)
            ssh_port: SSH port (for SSH method)
            ssh_key: Path to SSH private key (for SSH method)
        """
        self.method = method
        self.no_restart = no_restart
        self.ssh_host = ssh_host
        self.ssh_user = ssh_user
        self.ssh_port = ssh_port
        self.ssh_key = ssh_key

    def _find_default_ssh_key(self) -> str:
        """Find default SSH private key in ~/.ssh/.
        
        Returns:
            str: Path to SSH key or None if not found
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

    def _build_ssh_cmd(self, command: str) -> list:
        """Build SSH command for remote execution.
        
        Args:
            command: Command to execute on remote host
            
        Returns:
            list: Full SSH command
        """
        cmd = ['ssh', '-o', 'StrictHostKeyChecking=no', '-o', 'ConnectTimeout=10']
        if self.ssh_port != 22:
            cmd.extend(['-p', str(self.ssh_port)])
        
        # Use provided key or auto-detect default
        key = self.ssh_key or self._find_default_ssh_key()
        if key:
            cmd.extend(['-i', key])
        
        # Support user@host format in ssh_host
        target = self.ssh_host
        if self.ssh_user and '@' not in self.ssh_host:
            target = f"{self.ssh_user}@{self.ssh_host}"
        
        cmd.append(target)
        cmd.append(command)
        return cmd

    def restart(self):
        """Restart Ollama using configured method."""
        if self.no_restart:
            print(get_text("restart_ollama_disabled"))
            return

        print(get_text("restart_ollama"))

        try:
            if self.method == RestartMethod.SSH:
                if not self.ssh_host:
                    print(get_text("error_ssh_no_host"))
                    return
                cmd = self._build_ssh_cmd("sudo systemctl restart ollama")
                print(get_text("restart_ssh_exec", cmd=' '.join(cmd)))
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    print(get_text("error_restart_command", cmd=' '.join(cmd), stderr=result.stderr))
                else:
                    print(get_text("restart_success"))
                time.sleep(4)
                return

            if self.method == RestartMethod.SYSTEMCTL:
                cmd = ['sudo', 'systemctl', 'restart', 'ollama']
            elif self.method == RestartMethod.DOCKER:
                cmd = ['docker', 'restart', 'ollama']
            elif self.method == RestartMethod.KILL_START:
                subprocess.run(['sudo', 'systemctl', 'stop', 'ollama'], check=False)
                time.sleep(1)
                cmd = ['sudo', 'systemctl', 'start', 'ollama']
            elif self.method == RestartMethod.MANUAL:
                cmd = ['ollama', 'restart']
            else:
                print(get_text("error_unknown_restart_method", method=self.method.value))
                return

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(get_text("error_restart_command", cmd=' '.join(cmd), stderr=result.stderr))
            else:
                print(get_text("restart_success"))

            time.sleep(4)
        except FileNotFoundError:
            fallback_cmd = (
                ['ollama', 'restart'] if self.method == RestartMethod.MANUAL
                else ['systemctl', 'restart', 'ollama']
            )
            print(get_text("error_restart_command_not_found", cmd=' '.join(fallback_cmd)))
        except PermissionError:
            print(get_text("error_restart_permission"))
        except subprocess.TimeoutExpired:
            print(get_text("error_restart_timeout"))
        except Exception as e:
            print(get_text("error_restart_unknown", error=str(e)))


def restart_ollama(method: RestartMethod = RestartMethod.SYSTEMCTL, no_restart: bool = False,
                   ssh_host: str = None, ssh_user: str = None, ssh_port: int = 22, ssh_key: str = None):
    """Convenience function to restart Ollama.

    Args:
        method: Restart method (SYSTEMCTL, DOCKER, KILL_START, MANUAL, SSH)
        no_restart: If True, restart is not performed
        ssh_host: SSH host (for SSH method)
        ssh_user: SSH user (for SSH method)
        ssh_port: SSH port (for SSH method)
        ssh_key: Path to SSH private key (for SSH method)
    """
    manager = RestartManager(
        method=method, no_restart=no_restart,
        ssh_host=ssh_host, ssh_user=ssh_user,
        ssh_port=ssh_port, ssh_key=ssh_key
    )
    manager.restart()
