"""llama.cpp server API clients with OpenAI-compatible interface."""

import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import requests

from api.base_client import BaseApiClient
from system.gpu_monitor import get_vram_usage
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

        # Start VRAM monitoring
        max_vram_ref: list[float] = [0.0]
        stop_monitoring = threading.Event()
        vram_samples: list[float] = []
        monitor_thread = threading.Thread(
            target=self._monitor_vram,
            args=(stop_monitoring, max_vram_ref, vram_samples),
            daemon=True,
        )
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

                                    # Call token update callback
                                    if on_token_update:
                                        elapsed = max(
                                            time.time() - generation_start_time,
                                            0.001,
                                        )
                                        current_tps = (
                                            completion_token_count / elapsed
                                            if completion_token_count > 0
                                            else 0.0
                                        )
                                        on_token_update(
                                            prompt_token_count,
                                            completion_token_count,
                                            completion_token_count,
                                            len(response_text),
                                            False,
                                            current_tps,
                                            0.0,
                                            0.0,
                                            0.0,
                                            0.0,
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

        return avg_tps, vram, tps_list, error, prompt_metadata, temperature, None


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
