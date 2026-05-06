"""Ollama service restart logic."""

import subprocess
import time
from enum import Enum
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
                 ssh_client: SSHClient = None):
        """Initialize restart manager.

        Args:
            method: Restart method (SYSTEMCTL, DOCKER, KILL_START, MANUAL, SSH)
            no_restart: If True, restart is not performed
            ssh_client: SSHClient instance for remote operations
        """
        self.method = method
        self.no_restart = no_restart
        self.ssh_client = ssh_client

    def restart(self):
        """Restart Ollama using configured method."""
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

            # All local methods use 'ollama restart' to avoid sudo
            # This works because Ollama runs as a user service for local sessions
            cmd = ['ollama', 'restart']

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                # Fallback: try systemctl --user (user-level service)
                try:
                    result = subprocess.run(
                        ['systemctl', '--user', 'restart', 'ollama'],
                        capture_output=True, text=True, timeout=60
                    )
                except subprocess.TimeoutExpired:
                    print(get_text("error_restart_timeout"))
                    return
                except FileNotFoundError:
                    pass
                
                if result.returncode != 0:
                    print(get_text("error_restart_command", cmd=' '.join(cmd), stderr=result.stderr or "systemctl --user also failed"))
                    print(f"   💡 Tip: You can restart Ollama manually by running: ollama restart")
                else:
                    print(get_text("restart_success"))
            else:
                print(get_text("restart_success"))

            time.sleep(4)
        except subprocess.TimeoutExpired:
            print(get_text("error_restart_timeout"))
        except FileNotFoundError:
            print(get_text("error_restart_command_not_found", cmd='ollama restart'))
        except PermissionError:
            print(get_text("error_restart_permission"))
        except Exception as e:
            print(get_text("error_restart_unknown", error=str(e)))


def restart_ollama(method: RestartMethod = RestartMethod.MANUAL, no_restart: bool = False,
                   ssh_client: SSHClient = None,
                   # Deprecated parameters for backward compatibility
                   ssh_host: str = None, ssh_user: str = None,
                   ssh_port: int = 22, ssh_key: str = None):
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
