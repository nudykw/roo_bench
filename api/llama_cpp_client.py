"""llama.cpp server API clients with OpenAI-compatible interface."""

import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import requests

from api.base_client import BaseApiClient
from system.gpu_monitor import get_gpu_utilization, get_vram_total, get_vram_usage
from system.ssh_client import SSHClient

logger = logging.getLogger("roo_bench")


class LlamaCppApiClient(BaseApiClient):
    """Local API client for llama.cpp server (OpenAI-compatible API)."""

    def __init__(self, base_url: str, headers: dict | None = None, timeout: int = 300):
        super().__init__(base_url, headers or {}, timeout)
        self.ssh_client = None

    @property
    def is_remote(self) -> bool:
        return False

    def _monitor_vram(
        self,
        stop_event: threading.Event,
        max_vram_ref: list[float],
        vram_samples: list[float],
    ) -> None:
        """Monitor VRAM using nvidia-smi via system/gpu_monitor."""
        while not stop_event.is_set():
            v = get_vram_usage()
            if v is not None:
                vram_samples.append(v)
                if v > max_vram_ref[0]:
                    max_vram_ref[0] = v
            stop_event.wait(0.2)

    def _get_current_metrics(
        self,
    ) -> tuple[float, float, float, float]:
        """Get current CPU, RAM, VRAM, and GPU utilization percentages.

        Returns:
            Tuple of (cpu_percent, ram_percent, vram_percent, gpu_percent)
        """
        cpu_percent = 0.0
        ram_percent = 0.0
        vram_percent = 0.0
        gpu_percent = 0.0

        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.01)
            ram = psutil.virtual_memory()
            ram_percent = ram.percent
        except ImportError:
            pass
        except Exception:
            pass

        gpu_util = get_gpu_utilization()
        if gpu_util is not None:
            gpu_percent = gpu_util
            vram = get_vram_usage()
            vram_total = get_vram_total()
            if vram is not None and vram_total is not None and vram_total > 0:
                vram_percent = (vram / vram_total) * 100

        return cpu_percent, ram_percent, vram_percent, gpu_percent

    def get_models(self) -> list[dict]:
        """Get list of available models via /v1/models endpoint.

        Returns:
            list: List of model dicts with 'name', 'size', 'digest' keys
        """
        try:
            response = requests.get(
                f"{self.base_url}/v1/models",
                headers=self.headers,
                timeout=self.timeout,
            )
            if response.status_code == 200:
                data = response.json()
                # OpenAI format: {"data": [{"id": "model-name", ...}]}
                models_raw = data.get("data", [])
                # Convert to format expected by roo_bench
                return [
                    {
                        "name": m.get("id", ""),
                        "size": 0,
                        "digest": "",
                    }
                    for m in models_raw
                ]
            return []
        except Exception as e:
            print(f"\u26a0\ufe0f  Failed to connect to llama.cpp server: {e}")
            return []

    def get_model_info(self, model_name: str) -> dict:
        """Get model info via /v1/models/{model_id} endpoint.

        Note: llama.cpp may not support individual model info,
        so we return a minimal valid dict.

        Args:
            model_name: Name of the model

        Returns:
            dict: Model information dict
        """
        try:
            response = requests.get(
                f"{self.base_url}/v1/models/{model_name}",
                headers=self.headers,
                timeout=self.timeout,
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    "name": data.get("id", model_name),
                    "parameters": "",
                    "model_info": {},
                }
            return {}
        except Exception:
            return {}

    def run_generation(
        self,
        model_name: str,
        context_size: int,
        num_runs: int = 3,
        disable_thinking: bool = True,
        temperature: float | None = None,
        prompt: str | None = None,
        prompt_metadata: dict[str, Any] | None = None,
        num_predict: int = 8192,
        on_token_update: Callable[..., Any] | None = None,
    ) -> tuple[
        float,
        int | None,
        list[dict[str, Any]],
        str | None,
        dict[str, Any] | None,
        float | None,
        dict[str, Any] | None,
    ]:
        """Run benchmark generation using /v1/completions endpoint.

        Args:
            model_name: Model name
            context_size: Context size (not used by llama.cpp directly)
            num_runs: Number of runs for averaging
            disable_thinking: Not used by llama.cpp
            temperature: Temperature value for generation
            prompt: Custom prompt to use
            prompt_metadata: Metadata about the prompt
            num_predict: Maximum number of tokens to generate
            on_token_update: Optional callback for real-time updates

        Returns:
            tuple: (avg_tps, vram, tps_list, error, prompt_metadata,
                temperature, resource_stats)
        """
        if prompt is None:
            prompt = (
                "Write a comprehensive Python script that implements a "
                "multithreaded web server. Explain every line in extreme detail."
            )

        tps_list: list[dict[str, Any]] = []
        vram: int | None = None
        error: str | None = None

        # Check if psutil is available for CPU/RAM monitoring
        try:
            import psutil
            psutil_available = True
        except ImportError:
            psutil_available = False

        # Start resource monitoring (CPU, RAM, VRAM, GPU)
        max_vram_ref: list[float] = [0.0]
        stop_monitoring = threading.Event()
        vram_samples: list[float] = []
        cpu_samples: list[float] = []
        ram_samples: list[float] = []
        gpu_samples: list[float] = []

        def resource_monitor():
            """Monitor CPU, RAM, VRAM, and GPU utilization."""
            while not stop_monitoring.is_set():
                try:
                    if psutil_available:
                        cpu_samples.append(psutil.cpu_percent(interval=0.1))
                        ram_samples.append(psutil.virtual_memory().percent)
                    gpu_util = get_gpu_utilization()
                    if gpu_util is not None:
                        gpu_samples.append(gpu_util)
                        vram = get_vram_usage()
                        if vram is not None:
                            vram_samples.append(vram)
                            if vram > max_vram_ref[0]:
                                max_vram_ref[0] = vram
                except Exception:
                    pass
                stop_monitoring.wait(0.2)

        monitor_thread = threading.Thread(target=resource_monitor, daemon=True)
        monitor_thread.start()

        try:
            for run_num in range(num_runs):
                # Build OpenAI-compatible payload
                payload: dict[str, Any] = {
                    "model": model_name,
                    "prompt": prompt,
                    "stream": True,
                    "max_tokens": num_predict if num_predict > 0 else 4096,
                    "temperature": temperature if temperature is not None else 0.1,
                }

                try:
                    response = requests.post(
                        f"{self.base_url}/v1/completions",
                        json=payload,
                        headers=self.headers,
                        timeout=self.timeout,
                        stream=True,
                    )

                    if response.status_code != 200:
                        try:
                            err_data = response.json()
                            err_msg = err_data.get("error", {}).get(
                                "message", f"HTTP {response.status_code}"
                            )
                        except Exception:
                            err_msg = (
                                f"HTTP {response.status_code}: {response.text[:100]}"
                            )
                        error = f"Error from llama.cpp API: {err_msg}"
                        break

                    # Parse SSE streaming response
                    response_text = ""
                    prompt_token_count = 0
                    completion_token_count = 0
                    generation_start_time = time.time()
                    buffer = b""
                    last_update_time: float = 0.0

                    for chunk in response.iter_content(chunk_size=8192):
                        if not chunk:
                            continue
                        buffer += chunk

                        while True:
                            idx = buffer.find(b"\n")
                            if idx == -1:
                                break

                            line = buffer[:idx].decode("utf-8", errors="replace")
                            buffer = buffer[idx + 1 :]

                            # Parse SSE format: "data: {...}"
                            if not line.startswith("data:"):
                                continue
                            json_str = line[5:].strip()
                            if json_str == "[DONE]":
                                break

                            try:
                                data = json.loads(json_str)
                                choices = data.get("choices", [])
                                if choices:
                                    choice = choices[0]
                                    delta = choice.get("delta", {})
                                    if "content" in delta:
                                        response_text += delta["content"]

                                    # Get token counts from usage (in final chunk)
                                    usage = data.get("usage", {})
                                    if usage:
                                        prompt_token_count = usage.get(
                                            "prompt_tokens", 0
                                        )
                                        completion_token_count = usage.get(
                                            "completion_tokens", 0
                                        )

                                    # Call token update callback with resource metrics
                                    if on_token_update:
                                        # Throttle updates to max 1 per second
                                        current_time = time.time()
                                        should_update = (
                                            data.get("done", False)
                                            or (current_time - last_update_time) >= 1.0
                                        )

                                        if should_update:
                                            elapsed = max(
                                                current_time - generation_start_time,
                                                0.001,
                                            )
                                            current_tps = (
                                                completion_token_count / elapsed
                                                if completion_token_count > 0
                                                else 0.0
                                            )

                                            # Get current resource metrics
                                            if psutil_available:
                                                try:
                                                    cpu_percent = psutil.cpu_percent(interval=0.01)
                                                    ram = psutil.virtual_memory()
                                                    ram_percent = ram.percent
                                                except Exception:
                                                    cpu_percent = 0.0
                                                    ram_percent = 0.0
                                            else:
                                                cpu_percent = 0.0
                                                ram_percent = 0.0

                                            gpu_util = get_gpu_utilization()
                                            vram_percent = 0.0
                                            gpu_percent = 0.0
                                            if gpu_util is not None:
                                                gpu_percent = gpu_util
                                                vram_current = get_vram_usage()
                                                vram_total = get_vram_total()
                                                if vram_current is not None and vram_total is not None and vram_total > 0:
                                                    vram_percent = (vram_current / vram_total) * 100

                                            last_update_time = current_time

                                            on_token_update(
                                                prompt_token_count,
                                                completion_token_count,
                                                completion_token_count,
                                                len(response_text),
                                                data.get("done", False),
                                                current_tps,
                                                cpu_percent,
                                                ram_percent,
                                                vram_percent,
                                                gpu_percent,
                                            )
                            except json.JSONDecodeError:
                                continue

                    # Calculate TPS for this run
                    total_duration = time.time() - generation_start_time
                    tps = (
                        completion_token_count / total_duration
                        if total_duration > 0
                        else 0
                    )

                    run_data = {
                        "run": run_num + 1,
                        "tps": tps,
                        "vram": (int(max_vram_ref[0]) if max_vram_ref[0] > 0 else None),
                        "prompt": prompt,
                        "prompt_metadata": prompt_metadata,
                        "temperature": temperature,
                        "total_duration": int(total_duration * 1e9),
                        "prompt_eval_count": prompt_token_count,
                        "eval_count": completion_token_count,
                        "response": response_text,
                    }
                    tps_list.append(run_data)

                except requests.exceptions.Timeout:
                    error = "Timeout: model was generating response too long."
                    break
                except requests.exceptions.ConnectionError:
                    error = "Connection error: llama.cpp server may have crashed."
                    break
                except Exception as e:
                    error = f"Unknown error: {e}"
                    break

        finally:
            stop_monitoring.set()
            monitor_thread.join(timeout=2)
            vram = int(max_vram_ref[0]) if max_vram_ref[0] > 0 else None

        # Calculate average TPS
        if tps_list:
            avg_tps = sum(r["tps"] for r in tps_list) / len(tps_list)
        else:
            avg_tps = 0.0

        # Build resource stats dict from cpu, ram, vram, gpu samples
        resource_stats: dict[str, Any] | None = None
        has_stats = False
        stats_dict: dict[str, Any] = {}

        # CPU stats
        if cpu_samples:
            has_stats = True
            stats_dict["cpu"] = {
                "current": cpu_samples[-1],
                "avg": sum(cpu_samples) / len(cpu_samples),
                "min": min(cpu_samples),
                "max": max(cpu_samples),
                "samples_count": len(cpu_samples),
            }

        # RAM stats
        if ram_samples:
            has_stats = True
            stats_dict["ram"] = {
                "percent_current": ram_samples[-1],
                "avg_percent": sum(ram_samples) / len(ram_samples),
                "min_percent": min(ram_samples),
                "max_percent": max(ram_samples),
                "samples_count": len(ram_samples),
            }

        # VRAM stats
        vram_total = get_vram_total()
        if vram_samples and vram_total is not None and vram_total > 0:
            has_stats = True
            stats_dict["vram"] = {
                "used_current": vram_samples[-1],
                "total": vram_total,
                "percent_current": (vram_samples[-1] / vram_total * 100),
                "avg_percent": sum(s / vram_total * 100 for s in vram_samples) / len(vram_samples),
                "min_percent": min(s / vram_total * 100 for s in vram_samples),
                "max_percent": max(s / vram_total * 100 for s in vram_samples),
                "samples_count": len(vram_samples),
            }
        elif max_vram_ref[0] > 0:
            has_stats = True
            stats_dict["vram"] = {
                "used_current": int(max_vram_ref[0]),
                "total": vram_total if vram_total is not None else 0,
                "percent_current": (max_vram_ref[0] / vram_total * 100) if vram_total and vram_total > 0 else 0,
                "avg_percent": 0,
                "min_percent": 0,
                "max_percent": (max_vram_ref[0] / vram_total * 100) if vram_total and vram_total > 0 else 0,
                "samples_count": 0,
            }

        # GPU stats
        if gpu_samples:
            has_stats = True
            stats_dict["gpu"] = {
                "current": gpu_samples[-1],
                "avg": sum(gpu_samples) / len(gpu_samples),
                "min": min(gpu_samples),
                "max": max(gpu_samples),
                "samples_count": len(gpu_samples),
            }

        if has_stats:
            resource_stats = stats_dict

        return avg_tps, vram, tps_list, error, prompt_metadata, temperature, resource_stats


class LlamaCppRemoteApiClient(LlamaCppApiClient):
    """Remote API client for llama.cpp server with SSH for VRAM monitoring."""

    def __init__(
        self,
        base_url: str,
        headers: dict | None = None,
        timeout: int = 300,
        ssh_client: SSHClient | None = None,
    ):
        super().__init__(base_url, headers, timeout)
        self.ssh_client = ssh_client

    @property
    def is_remote(self) -> bool:
        return True

    def run_generation(
        self,
        model_name: str,
        context_size: int,
        num_runs: int = 3,
        disable_thinking: bool = True,
        temperature: float | None = None,
        prompt: str | None = None,
        prompt_metadata: dict[str, Any] | None = None,
        num_predict: int = 8192,
        on_token_update: Callable[..., Any] | None = None,
    ) -> tuple[
        float,
        int | None,
        list[dict[str, Any]],
        str | None,
        dict[str, Any] | None,
        float | None,
        dict[str, Any] | None,
    ]:
        """Run benchmark generation using SSH for remote VRAM monitoring."""
        if prompt is None:
            prompt = (
                "Write a comprehensive Python script that implements a "
                "multithreaded web server. Explain every line in extreme detail."
            )

        tps_list: list[dict[str, Any]] = []
        vram: int | None = None
        error: str | None = None

        # Start remote resource monitoring via SSH
        max_vram_ref: list[float] = [0.0]
        stop_monitoring = threading.Event()
        vram_samples: list[float] = []
        vram_total: int | None = None

        def remote_resource_monitor():
            """Monitor remote VRAM via SSH using nvidia-smi."""
            nonlocal vram_total
            if not self.ssh_client or not self.ssh_client.is_configured:
                return

            # Get VRAM total once
            try:
                result = self.ssh_client.execute(
                    "nvidia-smi --query-gpu=memory.total --format=csv,nounits,noheader",
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    vram_total = int(result.stdout.strip()) * 1024 * 1024
            except Exception:
                pass

            while not stop_monitoring.is_set():
                try:
                    result = self.ssh_client.execute(
                        "nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader",
                        timeout=5,
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        lines = result.stdout.strip().split("\n")
                        v = int(lines[0].strip()) * 1024 * 1024
                        vram_samples.append(v)
                        if v > max_vram_ref[0]:
                            max_vram_ref[0] = v
                except Exception:
                    pass
                stop_monitoring.wait(0.5)

        monitor_thread = threading.Thread(target=remote_resource_monitor, daemon=True)
        monitor_thread.start()
        # Wait for at least one VRAM sample before starting generation
        time.sleep(0.6)

        try:
            for run_num in range(num_runs):
                payload: dict[str, Any] = {
                    "model": model_name,
                    "prompt": prompt,
                    "stream": True,
                    "max_tokens": num_predict if num_predict > 0 else 4096,
                    "temperature": temperature if temperature is not None else 0.1,
                }

                try:
                    response = requests.post(
                        f"{self.base_url}/v1/completions",
                        json=payload,
                        headers=self.headers,
                        timeout=self.timeout,
                        stream=True,
                    )

                    if response.status_code != 200:
                        try:
                            err_data = response.json()
                            err_msg = err_data.get("error", {}).get(
                                "message", f"HTTP {response.status_code}"
                            )
                        except Exception:
                            err_msg = (
                                f"HTTP {response.status_code}: {response.text[:100]}"
                            )
                        error = f"Error from llama.cpp API: {err_msg}"
                        break

                    # Parse SSE streaming response
                    response_text = ""
                    prompt_token_count = 0
                    completion_token_count = 0
                    generation_start_time = time.time()
                    buffer = b""
                    last_update_time: float = 0.0

                    for chunk in response.iter_content(chunk_size=8192):
                        if not chunk:
                            continue
                        buffer += chunk

                        while True:
                            idx = buffer.find(b"\n")
                            if idx == -1:
                                break

                            line = buffer[:idx].decode("utf-8", errors="replace")
                            buffer = buffer[idx + 1 :]

                            if not line.startswith("data:"):
                                continue
                            json_str = line[5:].strip()
                            if json_str == "[DONE]":
                                break

                            try:
                                data = json.loads(json_str)
                                choices = data.get("choices", [])
                                if choices:
                                    choice = choices[0]
                                    delta = choice.get("delta", {})
                                    if "content" in delta:
                                        response_text += delta["content"]

                                    usage = data.get("usage", {})
                                    if usage:
                                        prompt_token_count = usage.get(
                                            "prompt_tokens", 0
                                        )
                                        completion_token_count = usage.get(
                                            "completion_tokens", 0
                                        )

                                    if on_token_update:
                                        current_time = time.time()
                                        should_update = (
                                            data.get("done", False)
                                            or (current_time - last_update_time) >= 1.0
                                        )

                                        if should_update:
                                            elapsed = max(
                                                current_time - generation_start_time,
                                                0.001,
                                            )
                                            current_tps = (
                                                completion_token_count / elapsed
                                                if completion_token_count > 0
                                                else 0.0
                                            )

                                            # Calculate remote VRAM percent
                                            vram_percent = 0.0
                                            gpu_percent = 0.0
                                            if vram_total and vram_total > 0:
                                                vram_percent = (
                                                    (vram_samples[-1] / vram_total * 100)
                                                    if vram_samples else 0
                                                )
                                                # Use VRAM percent as proxy for GPU percent
                                                gpu_percent = vram_percent

                                            last_update_time = current_time

                                            on_token_update(
                                                prompt_token_count,
                                                completion_token_count,
                                                completion_token_count,
                                                len(response_text),
                                                data.get("done", False),
                                                current_tps,
                                                0.0,  # cpu_percent (not available remotely)
                                                0.0,  # ram_percent (not available remotely)
                                                vram_percent,
                                                gpu_percent,
                                            )
                            except json.JSONDecodeError:
                                continue

                    total_duration = time.time() - generation_start_time
                    tps = (
                        completion_token_count / total_duration
                        if total_duration > 0
                        else 0
                    )

                    run_data = {
                        "run": run_num + 1,
                        "tps": tps,
                        "vram": (int(max_vram_ref[0]) if max_vram_ref[0] > 0 else None),
                        "prompt": prompt,
                        "prompt_metadata": prompt_metadata,
                        "temperature": temperature,
                        "total_duration": int(total_duration * 1e9),
                        "prompt_eval_count": prompt_token_count,
                        "eval_count": completion_token_count,
                        "response": response_text,
                    }
                    tps_list.append(run_data)

                except requests.exceptions.Timeout:
                    error = "Timeout: model was generating response too long."
                    break
                except requests.exceptions.ConnectionError:
                    error = "Connection error: llama.cpp server may have crashed."
                    break
                except Exception as e:
                    error = f"Unknown error: {e}"
                    break

        finally:
            stop_monitoring.set()
            monitor_thread.join(timeout=2)
            vram = int(max_vram_ref[0]) if max_vram_ref[0] > 0 else None

        # Calculate average TPS
        if tps_list:
            avg_tps = sum(r["tps"] for r in tps_list) / len(tps_list)
        else:
            avg_tps = 0.0

        # Build resource stats dict from vram_samples
        resource_stats: dict[str, Any] | None = None
        if vram_samples and vram_total is not None and vram_total > 0:
            resource_stats = {
                "vram": {
                    "used_current": vram_samples[-1],
                    "total": vram_total,
                    "percent_current": (vram_samples[-1] / vram_total * 100),
                    "avg_percent": sum(s / vram_total * 100 for s in vram_samples) / len(vram_samples),
                    "min_percent": min(s / vram_total * 100 for s in vram_samples),
                    "max_percent": max(s / vram_total * 100 for s in vram_samples),
                    "samples_count": len(vram_samples),
                }
            }

        return avg_tps, vram, tps_list, error, prompt_metadata, temperature, resource_stats

    def _monitor_vram(
        self,
        stop_event: threading.Event,
        max_vram_ref: list[float],
        vram_samples: list[float],
    ) -> None:
        """Monitor remote VRAM via SSH using nvidia-smi."""
        if not self.ssh_client or not self.ssh_client.is_configured:
            return

        while not stop_event.is_set():
            try:
                result = self.ssh_client.execute(
                    "nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader",
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    # nvidia-smi returns MiB, convert to bytes
                    lines = result.stdout.strip().split("\n")
                    v = int(lines[0].strip()) * 1024 * 1024
                    vram_samples.append(v)
                    if v > max_vram_ref[0]:
                        max_vram_ref[0] = v
            except Exception:
                pass
            stop_event.wait(0.5)
