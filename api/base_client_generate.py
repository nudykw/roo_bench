"""Generation methods for BaseApiClient."""

import json
import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any

import requests

from i18n import get_text
from system.gpu_monitor import get_gpu_utilization, get_vram_total, get_vram_usage

logger = logging.getLogger('roo_bench')


class BaseApiClientGenerate:
    """Mixin class for generation methods."""

    base_url: str
    headers: dict
    timeout: int
    is_remote: bool
    ssh_client: Any

    def run_generation(self, model_name: str, context_size: int, num_runs: int = 3,
                       disable_thinking: bool = True,
                       temperature: float | None = None,
                       prompt: str | None = None,
                       prompt_metadata: dict[str, Any] | None = None,
                       num_predict: int = 8192,
                       on_token_update: Callable[..., Any] | None = None) -> tuple[float, int | None, list[dict[str, Any]], str | None, dict[str, Any] | None, float | None, dict[str, Any] | None]:
        """Run benchmark generation for a model with multiple runs for averaging.

        Args:
            model_name: Model name
            context_size: Context size
            num_runs: Number of runs for averaging (default: 3)
            disable_thinking: If True, disables thinking mode to prevent reasoning loops (default: True)
            temperature: Temperature value for generation (None uses model default)
            prompt: Custom prompt to use. If None, uses default benchmark prompt.
            prompt_metadata: Metadata about the prompt (id, name, mode, chain info)
            num_predict: Maximum number of tokens to generate (default: 8192, use -1 for unlimited)
            on_token_update: Optional callback function(current_prompt_tokens, current_response_tokens) for real-time updates

        Returns:
            tuple: (avg_tps, vram, tps_list, error, prompt_metadata, temperature, resource_stats)
                avg_tps: Average TPS (float)
                vram: VRAM usage in bytes (int or None if GPU unavailable)
                tps_list: List of results for each run (list of dict)
                error: Error message (str or None)
                prompt_metadata: The prompt metadata that was used
                temperature: Temperature value used (or None if default)
                resource_stats: Resource statistics dict or None
        """
        # Use custom prompt or default
        if prompt is None:
            prompt = "Write a comprehensive Python script that implements a multithreaded web server. Explain every line in extreme detail."

        tps_list: list[dict[str, Any]] = []
        vram = None
        error = None

        # Build payload with optional thinking disable and temperature
        # Note: "think" is a top-level API parameter, NOT inside "options"
        # See: https://github.com/ollama/ollama/blob/main/docs/api.md
        
        # Initialize resource monitoring
        resource_monitor = None
        if self.is_remote and self.ssh_client:
            from system.system_monitor import ResourceMonitor
            resource_monitor = ResourceMonitor('remote', 0.5, self.ssh_client)
        elif not self.is_remote:
            from system.system_monitor import ResourceMonitor
            resource_monitor = ResourceMonitor('local', 0.5)
        
        # Start resource monitoring if available
        if resource_monitor:
            resource_monitor.start_monitoring()
            # Wait for at least one sample to be collected before starting generation
            time.sleep(0.6)
        
        try:
            for run_num in range(num_runs):
                options = {
                    "num_ctx": context_size
                }
                # Only add num_predict if explicitly set (default -1 for unlimited)
                if num_predict is not None:
                    options["num_predict"] = num_predict
                
                payload: dict[str, Any] = {
                    "model": model_name,
                    "prompt": prompt,
                    "stream": True,
                    "options": options
                }
                if disable_thinking:
                    payload["think"] = False
                if temperature is not None:
                    payload["options"]["temperature"] = float(temperature)

                response = None
                max_vram_ref: list[float] = [0.0]  # Use list for mutable reference in thread
                stop_monitoring = threading.Event()
                vram_samples: list[float] = []

                # Create token update callback for this run (or None)
                run_on_token_update = on_token_update if on_token_update else None
                
                logger.info("[DEBUG] Starting run %d with stream=True, on_token_update=%s", run_num + 1, run_on_token_update is not None)

                try:
                    # Start VRAM monitoring thread before blocking request
                    monitor_thread = threading.Thread(target=self._monitor_vram, args=(stop_monitoring, max_vram_ref, vram_samples), daemon=True)
                    monitor_thread.start()

                    try:
                        logger.info("[DEBUG] Sending streaming request to %s", f"{self.base_url}/api/generate")
                        response = requests.post(
                            f"{self.base_url}/api/generate",
                            json=payload,
                            headers=self.headers,
                            timeout=self.timeout,
                            stream=True
                        )
                        logger.info("[DEBUG] Response status: %d", response.status_code)
                    except KeyboardInterrupt:
                        stop_monitoring.set()
                        monitor_thread.join(timeout=2)
                        raise

                    if response.status_code != 200:
                        stop_monitoring.set()
                        monitor_thread.join(timeout=2)
                        try:
                            err_msg = response.json().get("error", f"HTTP {response.status_code}")
                        except Exception:
                            err_msg = f"HTTP {response.status_code}: {response.text[:100]}"
                        error = get_text("error_ollama_api", error_msg=err_msg)
                        break

                    logger.info("[DEBUG] Starting to parse streaming response for run %d", run_num + 1)
                    
                    # Parse streaming response using raw bytes for real-time updates
                    response_text = ""
                    total_duration = 0
                    eval_count = 0
                    prompt_eval_count = 0
                    
                    # CPU/RAM metrics collection
                    cpu_percent = 0.0
                    ram_percent = 0.0
                    vram_percent = 0.0
                    gpu_percent = 0.0
                    
                    # Try to import psutil for CPU/RAM monitoring
                    try:
                        import psutil
                        psutil_available = True
                    except ImportError:
                        psutil_available = False
                    
                    # Use iter_content for real-time chunk processing
                    buffer = b""
                    chunk_count = 0
                    callback_count = 0
                    last_update_time: float = 0.0  # Track last callback time for throttling
                    generation_start_time = time.time()
                    logger.info("[DEBUG] Starting iter_content loop for run %d", run_num + 1)
                    
                    for chunk in response.iter_content(chunk_size=8192):
                        if not chunk:
                            continue
                        
                        chunk_count += 1
                        buffer += chunk
                        
                        # Process complete JSON objects in the buffer
                        while True:
                            # Find the next complete JSON object
                            idx = buffer.find(b"\n")
                            if idx == -1:
                                break
                            
                            line_bytes = buffer[:idx]
                            buffer = buffer[idx+1:]
                            
                            if not line_bytes.strip():
                                continue
                            
                            try:
                                data = json.loads(line_bytes.decode('utf-8'))
                                
                                # Track response text
                                if 'response' in data:
                                    response_text += data['response']
                                
                                # Track token counts (these accumulate during streaming)
                                if 'prompt_eval_count' in data:
                                    prompt_eval_count = data['prompt_eval_count']
                                if 'eval_count' in data:
                                    eval_count = data['eval_count']
                                
                                # Track duration
                                if 'total_duration' in data:
                                    total_duration = data['total_duration']
                                
                                # Collect CPU/RAM metrics
                                if psutil_available:
                                    try:
                                        cpu_percent = psutil.cpu_percent(interval=0.01)
                                        ram_percent = psutil.virtual_memory().percent
                                    except Exception:
                                        pass
                                
                                # Collect GPU metrics
                                gpu_util = get_gpu_utilization()
                                if gpu_util is not None:
                                    gpu_percent = gpu_util
                                    vram = get_vram_usage()
                                    vram_total = get_vram_total()
                                    if vram is not None and vram_total is not None:
                                        vram_percent = (vram / vram_total) * 100
                                
                                # Call token update callback for real-time display
                                # During streaming (done=False), pass response length as approximate token count
                                # When done=True, pass real token counts
                                # Throttle updates to max 1 per second using time-based check
                                current_time = time.time()
                                should_update = (data.get('done', False) or
                                               (current_time - last_update_time) >= 1.0)
                                
                                if run_on_token_update and should_update:
                                    callback_count += 1
                                    last_update_time = current_time
                                    # Estimate response tokens from response text length (~4 chars per token)
                                    estimated_response_tokens = len(response_text) // 4 if response_text else 0
                                    response_len = len(response_text)
                                    display_tokens = eval_count or estimated_response_tokens
                                    elapsed = max(current_time - generation_start_time, 0.001)
                                    current_tps = display_tokens / elapsed if display_tokens > 0 else 0.0
                                    run_on_token_update(
                                        prompt_eval_count,
                                        eval_count,
                                        estimated_response_tokens,
                                        response_len,
                                        data.get('done', False),
                                        current_tps,
                                        cpu_percent,
                                        ram_percent,
                                        vram_percent,
                                        gpu_percent,
                                    )
                                    
                            except json.JSONDecodeError:
                                continue
                    
                    logger.info("[DEBUG] Finished iter_content loop: %d chunks, %d callbacks", chunk_count, callback_count)

                    # Stop monitoring after response received
                    stop_monitoring.set()
                    monitor_thread.join(timeout=2)
                    
                    # Check if we got any meaningful response
                    if not response_text and not eval_count:
                        logger.warning("[DEBUG] Empty response for run %d", run_num + 1)
                        # Don't break - allow retry on next run

                    # Debug: Log full response structure for diagnosis
                    logger.debug(
                        "[DEBUG] API response fields: eval_count=%d prompt_eval_count=%d total_duration=%d done_reason=%r",
                        eval_count, prompt_eval_count, total_duration,
                        "streaming_complete"
                    )
                    
                    # Check if response is a string (not a list or other type)
                    if not isinstance(response_text, str):
                        logger.warning(
                            "[DEBUG] response field is not a string, type=%s, converting to str",
                            type(response_text).__name__
                        )
                        response_text = str(response_text) if response_text else ""

                    logger.info(
                        "[Expert] response collection: response_text_len=%d first100=%r",
                        len(response_text), response_text[:100]
                    )

                    # Create run_data with initial values
                    run_data = {
                        'run': run_num + 1,
                        'tps': 0,
                        'vram': None,
                        'prompt': prompt,
                        'prompt_metadata': prompt_metadata,
                        'temperature': temperature
                    }

                    # Parse TPS from server-side metrics
                    total_duration_sec = total_duration / 1e9 if total_duration else 0
                    tps = eval_count / total_duration_sec if total_duration_sec > 0 else 0

                    # Use max VRAM observed during generation
                    vram_during = max_vram_ref[0] if vram_samples else None

                    # Update run_data with actual values
                    run_data['tps'] = tps
                    run_data['vram'] = vram_during
                    run_data['total_duration'] = total_duration
                    run_data['prompt_eval_count'] = prompt_eval_count
                    run_data['eval_count'] = eval_count
                    run_data['response'] = response_text
                    tps_list.append(run_data)

                except KeyboardInterrupt:
                    # Re-raise KeyboardInterrupt to allow Ctrl+C to propagate
                    stop_monitoring.set()
                    monitor_thread.join(timeout=2)
                    raise
                except requests.exceptions.Timeout:
                    stop_monitoring.set()
                    error = get_text("error_timeout")
                    break
                except requests.exceptions.ConnectionError:
                    stop_monitoring.set()
                    error = get_text("error_crash")
                    break
                except Exception as e:
                    stop_monitoring.set()
                    err_details = get_text("error_unknown", error_details=str(e))
                    error = err_details
                    break
        finally:
            # Calculate average TPS and cleanup (always runs)
            if tps_list:
                avg_tps = sum(r['tps'] for r in tps_list) / len(tps_list)
                # Calculate standard deviation
                if len(tps_list) > 1:
                    mean = avg_tps
                    variance = sum((float(r['tps']) - mean) ** 2 for r in tps_list) / len(tps_list)
                    std_dev = math.sqrt(variance)
                else:
                    std_dev = 0.0

                # Return VRAM from the last successful run
                vram = tps_list[-1]['vram'] if tps_list else None

                # Get resource statistics if monitoring was active
                resource_stats = None
                if resource_monitor:
                    resource_monitor.stop_monitoring()
                    resource_stats = resource_monitor.get_aggregated_stats()
            else:
                avg_tps = 0.0
                vram = None
                resource_stats = None
        
        return avg_tps, vram, tps_list, error, prompt_metadata, temperature, resource_stats
