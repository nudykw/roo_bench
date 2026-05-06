"""Ollama service restart logic."""

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


class RestartManager:
    """Manages Ollama service restart operations."""

    def __init__(self, method: RestartMethod = RestartMethod.SYSTEMCTL, no_restart: bool = False):
        """Initialize restart manager.

        Args:
            method: Restart method (SYSTEMCTL, DOCKER, KILL_START, MANUAL)
            no_restart: If True, restart is not performed
        """
        self.method = method
        self.no_restart = no_restart

    def restart(self):
        """Restart Ollama using configured method."""
        if self.no_restart:
            print(get_text("restart_ollama_disabled"))
            return

        print(get_text("restart_ollama"))

        try:
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
        except Exception as e:
            print(get_text("error_restart_unknown", error=str(e)))


def restart_ollama(method: RestartMethod = RestartMethod.SYSTEMCTL, no_restart: bool = False):
    """Convenience function to restart Ollama.

    Args:
        method: Restart method (SYSTEMCTL, DOCKER, KILL_START, MANUAL)
        no_restart: If True, restart is not performed
    """
    manager = RestartManager(method=method, no_restart=no_restart)
    manager.restart()
