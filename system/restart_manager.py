"""Ollama service restart logic."""

import subprocess
import time
from enum import Enum
from typing import Optional

from i18n import get_text
from system.ssh_client import SSHClient


class RestartMethod(Enum):
    """Ollama restart methods."""
    SYSTEMCTL = "systemctl"
    DOCKER = "docker"
    KILL_START = "kill_start"
    MANUAL = "manual"
    SSH = "ssh"


class RestartManager:
    """Manages Ollama service restart operations."""

    def __init__(self, method: RestartMethod = RestartMethod.MANUAL, no_restart: bool = False,
                 ssh_client: Optional[SSHClient] = None):
        """Initialize restart manager.

        Args:
            method: Restart method (SYSTEMCTL, DOCKER, KILL_START, MANUAL, SSH)
            no_restart: If True, restart is not performed
            ssh_client: SSHClient instance for remote operations
        """
        self.method = method
        self.no_restart = no_restart
        self.ssh_client = ssh_client

    def restart(self) -> None:
        """Restart Ollama using configured method.

        NOTE: This is a LOCAL fallback for restarting Ollama.
        For remote servers, use ollama_client.unload_model() instead,
        which handles SSH restart automatically via api/base_client.py.

        Restart priority:
        1. ollama restart (if ollama is in PATH)
        2. systemctl --user restart ollama (user service)
        3. sudo systemctl restart ollama (system service)

        For remote servers: use RestartMethod.SSH with ssh_client parameter
        or call ollama_client.unload_model() from api/base_client.py

        IMPORTANT: During benchmark, use restart_ollama() to unload ALL models
        (including those loaded by other programs).
        After analysis/translation, use ollama_client.unload_model() to unload
        ONLY the specific model.
        """
        if self.no_restart:
            print(get_text("restart_ollama_disabled"))
            return

        print(get_text("restart_ollama"))

        try:
            if self.method == RestartMethod.SSH:
                if not self.ssh_client or not self.ssh_client.is_configured:
                    print(get_text("error_ssh_no_host"))
                    return
                cmd_list = self.ssh_client.build_command("sudo systemctl restart ollama")
                print(get_text("restart_ssh_exec", cmd=' '.join(cmd_list)))
                result = self.ssh_client.execute("sudo systemctl restart ollama", timeout=30)
                if result.returncode != 0:
                    print(get_text("error_restart_command", cmd=' '.join(cmd_list), stderr=result.stderr))
                else:
                    print(get_text("restart_success"))
                time.sleep(4)
                return

            # Try 'ollama restart' first (works when ollama is in PATH)
            ollama_success = False
            try:
                cmd = ['ollama', 'restart']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    ollama_success = True
                    print(get_text("restart_success"))
            except FileNotFoundError:
                # ollama not in PATH, will try fallbacks below
                pass

            if not ollama_success:
                # Fallback: try systemctl --user (user-level service)
                fallback_success = False
                try:
                    result = subprocess.run(
                        ['systemctl', '--user', 'restart', 'ollama'],
                        capture_output=True, text=True, timeout=60
                    )
                    if result.returncode == 0:
                        fallback_success = True
                        print(get_text("restart_success"))
                except subprocess.TimeoutExpired:
                    print(get_text("error_restart_timeout"))
                    return
                except FileNotFoundError:
                    pass

                if not fallback_success:
                    # Fallback: try sudo systemctl (system-level service)
                    try:
                        result = subprocess.run(
                            ['sudo', 'systemctl', 'restart', 'ollama'],
                            capture_output=True, text=True, timeout=60
                        )
                        if result.returncode == 0:
                            print(get_text("restart_success"))
                        else:
                            print(get_text("error_restart_command", cmd='sudo systemctl restart ollama', stderr=result.stderr or "all methods failed"))
                            print("   💡 Tip: You can restart Ollama manually by running: sudo systemctl restart ollama")
                    except subprocess.TimeoutExpired:
                        print(get_text("error_restart_timeout"))
                        return
                    except FileNotFoundError:
                        print(get_text("error_restart_command", cmd='sudo systemctl restart ollama', stderr="systemctl not found"))
                        print("   💡 Tip: You can restart Ollama manually by running: sudo systemctl restart ollama")
                    except PermissionError:
                        print(get_text("error_restart_permission"))

            time.sleep(4)
        except subprocess.TimeoutExpired:
            print(get_text("error_restart_timeout"))
        except PermissionError:
            print(get_text("error_restart_permission"))
        except Exception as e:
            print(get_text("error_restart_unknown", error=str(e)))


def restart_ollama(method: RestartMethod = RestartMethod.MANUAL, no_restart: bool = False,
                   ssh_client: Optional[SSHClient] = None,
                   # Deprecated parameters for backward compatibility
                   ssh_host: Optional[str] = None, ssh_user: Optional[str] = None,
                   ssh_port: int = 22, ssh_key: Optional[str] = None) -> None:
    """Convenience function to restart Ollama.

    Args:
        method: Restart method (SYSTEMCTL, DOCKER, KILL_START, MANUAL, SSH)
        no_restart: If True, restart is not performed
        ssh_client: SSHClient instance for remote operations
        ssh_host: SSH host (for SSH method) - deprecated, use ssh_client
        ssh_user: SSH user (for SSH method) - deprecated, use ssh_client
        ssh_port: SSH port (for SSH method) - deprecated, use ssh_client
        ssh_key: Path to SSH private key (for SSH method) - deprecated, use ssh_client
    """
    # Create SSHClient from deprecated parameters if not provided
    if ssh_client is None and (ssh_host or ssh_user or ssh_key):
        ssh_client = SSHClient(host=ssh_host, user=ssh_user,
                               port=ssh_port, key_path=ssh_key)

    manager = RestartManager(
        method=method, no_restart=no_restart,
        ssh_client=ssh_client
    )
    manager.restart()
