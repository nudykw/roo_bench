"""Base API client for Ollama API interactions."""

import json
import math
import threading
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple

import requests
from system.gpu_monitor import get_vram_usage
from system.ssh_client import SSHClient
from i18n import get_text


class BaseApiClient(ABC):
    """Abstract base class for API clients."""

    def __init__(self, base_url: str, headers: dict = None, timeout: int = 300):
        """Initialize base API client.

        Args:
            base_url: Ollama API base URL
            headers: HTTP headers (e.g., authentication)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.headers = headers or {}
        self.timeout = timeout

    @property
    @abstractmethod
    def is_remote(self) -> bool:
        """Return True if this is a remote client with SSH."""
        pass

    @abstractmethod
    def _monitor_vram(self, stop_event: threading.Event, max_vram_ref: list, vram_samples: list):
        """Internal VRAM monitoring implementation."""
        pass

    @staticmethod
    def get_capabilities_from_model_info(model_info: dict) -> dict:
        """Extract capabilities from Ollama API /api/show model_info.

        Uses both direct key detection and architecture-based heuristics.

        Args:
            model_info: Dictionary from model_info field of /api/show response

        Returns:
            dict: {'vision': bool, 'tools': bool, 'thinking': bool, 'audio': bool}
        """
        capabilities = {
            'vision': False,
            'tools': False,
            'thinking': False,
            'audio': False
        }

        # Get architecture and basename for heuristic detection
        architecture = model_info.get('general.architecture', '').lower()
        basename = model_info.get('general.basename', '').lower()
        file_type = model_info.get('general.file_type', '').lower()

        # Combine all strings for easier searching
        combined = f"{architecture} {basename} {file_type}"
        combined = combined.lower()

        # Vision detection
        if any(kw in combined for kw in ['vision', 'instruct', 'clip', 'image', 'multi modal', 'moar']):
            capabilities['vision'] = True
        elif any(kw in architecture for kw in ['clip', 'vlm', 'moondream']):
            capabilities['vision'] = True
        # Check for vision-specific keys
        for key in model_info.keys():
            if any(vision_kw in key.lower() for vision_kw in ['vision', 'image', 'clip', 'multi modal']):
                capabilities['vision'] = True
                break

        # Tools/function calling detection
        if any(kw in combined for kw in ['tools', 'function', 'tool', 'function calling']):
            capabilities['tools'] = True
        elif 'tokenizer.tools' in str(model_info.get('tokenizer', {})):
            capabilities['tools'] = True
        # Check for tools-specific keys
        for key in model_info.keys():
            if 'tools' in key.lower() or 'function' in key.lower():
                capabilities['tools'] = True
                break

        # Thinking/reasoning detection
        if any(kw in combined for kw in ['thinking', 'reason', 'deepseek', 'o1', 'o3']):
            capabilities['thinking'] = True
        elif any(kw in str(model_info.get('tokenizer', {})).lower() for kw in ['thinking', 'reason']):
            capabilities['thinking'] = True
        # Check for thinking-specific keys
        for key in model_info.keys():
            if 'thinking' in key.lower() or 'reason' in key.lower():
                capabilities['thinking'] = True
                break

        # Audio detection
        if any(kw in combined for kw in ['audio', 'whisper', 'speech', 'voice']):
            capabilities['audio'] = True
        elif any(kw in str(model_info.get('tokenizer', {})).lower() for kw in ['audio', 'whisper']):
            capabilities['audio'] = True
        # Check for audio-specific keys
        for key in model_info.keys():
            if any(audio_kw in key.lower() for audio_kw in ['audio', 'whisper', 'speech', 'voice']):
                capabilities['audio'] = True
                break

        return capabilities

    def get_models(self) -> list:
        """Get list of available models.

        Returns:
            list: List of model dictionaries
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=self.timeout
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("models", [])
            return []
        except Exception as e:
            print(get_text("error_ollama_connection", error=str(e)))
            return []

    def run_generation(self, model_name: str, context_size: int, num_runs: int = 3) -> tuple:
        """Run benchmark generation for a model with multiple runs for averaging.

        Args:
            model_name: Model name
            context_size: Context size
            num_runs: Number of runs for averaging (default: 3)

        Returns:
            tuple: (avg_tps, vram, tps_list, error)
                avg_tps: Average TPS (float)
                vram: VRAM usage in bytes (int or None if GPU unavailable)
                tps_list: List of results for each run (list of dict)
                error: Error message (str or None)
        """
        prompt = "Write a comprehensive Python script that implements a multithreaded web server. Explain every line in extreme detail."

        tps_list = []
        vram = None
        error = None

        for run_num in range(num_runs):
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_predict": 100,
                    "num_ctx": context_size
                }
            }

            response = None
            max_vram_ref = [0]  # Use list for mutable reference in thread
            stop_monitoring = threading.Event()
            vram_samples = []

            try:
                # Start VRAM monitoring thread
                monitor_thread = threading.Thread(target=self._monitor_vram, args=(stop_monitoring, max_vram_ref, vram_samples), daemon=True)
                monitor_thread.start()

                try:
                    response = requests.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        headers=self.headers,
                        timeout=self.timeout,
                        stream=True
                    )
                except KeyboardInterrupt:
                    stop_monitoring.set()
                    monitor_thread.join(timeout=2)
                    raise

                if response.status_code != 200:
                    stop_monitoring.set()
                    try:
                        err_msg = response.json().get("error", f"HTTP {response.status_code}")
                    except:
                        err_msg = f"HTTP {response.status_code}: {response.text[:100]}"
                    error = get_text("error_ollama_api", error_msg=err_msg)
                    break

                # Consume stream and capture final response
                final_data = None
                line_count = 0
                all_lines = []
                try:
                    for line in response.iter_lines():
                        if line:
                            line_count += 1
                            line_str = line.decode('utf-8')
                            all_lines.append(line_str)
                            # Check for done=true (with or without spaces)
                            if '"done":true' in line_str or '"done": true' in line_str:
                                try:
                                    final_data = json.loads(line_str)
                                except json.JSONDecodeError:
                                    pass
                except KeyboardInterrupt:
                    stop_monitoring.set()
                    monitor_thread.join(timeout=2)
                    raise

                # Stop monitoring after generation completes
                stop_monitoring.set()
                monitor_thread.join(timeout=2)

                # Parse final response for TPS data
                if final_data:
                    total_duration = final_data.get("total_duration", 0) / 1e9
                    eval_count = final_data.get("eval_count", 0)
                    tps = eval_count / total_duration if total_duration > 0 else 0
                else:
                    # Fallback: try to parse the last line as it might contain the stats
                    tps = 0
                    total_duration = 0
                    eval_count = 0
                    if all_lines:
                        last_line = all_lines[-1]
                        try:
                            final_data = json.loads(last_line)
                            total_duration = final_data.get("total_duration", 0) / 1e9
                            eval_count = final_data.get("eval_count", 0)
                            tps = eval_count / total_duration if total_duration > 0 else 0
                        except json.JSONDecodeError:
                            pass

                # Use max VRAM observed during generation
                vram_during = max_vram_ref[0] if vram_samples else None
                tps_list.append({"run": run_num + 1, "tps": tps, "vram": vram_during})

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

        # Calculate average TPS
        if tps_list:
            avg_tps = sum(r['tps'] for r in tps_list) / len(tps_list)
            # Calculate standard deviation
            if len(tps_list) > 1:
                mean = avg_tps
                variance = sum((r['tps'] - mean) ** 2 for r in tps_list) / len(tps_list)
                std_dev = math.sqrt(variance)
            else:
                std_dev = 0.0

            # Return VRAM from the last successful run
            vram = tps_list[-1]['vram'] if tps_list else None

            return avg_tps, vram, tps_list, None
        else:
            return 0.0, None, [], error

    def get_model_info(self, model_name: str) -> dict:
        """Get model information including current default parameters.

        Args:
            model_name: Model name

        Returns:
            dict: Model information with parameters (including num_ctx, num_predict, etc.)
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": model_name},
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            print(f"⚠️  Error getting model info for {model_name}: {e}")
            return {}

    def get_current_num_ctx(self, model_name: str) -> int:
        """Get the current default num_ctx for a model.

        Args:
            model_name: Model name

        Returns:
            int: Current num_ctx value (default: 2048)
        """
        model_info = self.get_model_info(model_name)
        if not model_info:
            return 2048

        # Look for num_ctx in model_info
        # It can be in 'parameters' field or as 'tokenizer.llama.num_ctx' in model_info
        parameters = model_info.get("parameters", "")
        if parameters:
            # Parse parameters string like "num_ctx:2048\nnum_predict:100"
            for line in parameters.split('\n'):
                line = line.strip()
                if line.startswith('num_ctx:'):
                    try:
                        return int(line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        pass

        # Also check model_info keys
        model_info_dict = model_info.get("model_info", {})
        for key, val in model_info_dict.items():
            if 'num_ctx' in key.lower() or 'context_length' in key.lower():
                try:
                    return int(val)
                except (ValueError, TypeError):
                    pass

        return 2048

    def get_running_models(self) -> list:
        """Get list of currently running models with their actual context usage.

        Uses the /api/ps endpoint which shows the actual context window being used.

        Returns:
            list: List of dicts with model info including 'n_ctx' field
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/ps",
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                # API can return a dict with 'models' key or a list
                if isinstance(data, dict) and 'models' in data:
                    return data['models']
                elif isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return [data]
                return []
            return []
        except Exception as e:
            print(f"⚠️  Error getting running models: {e}")
            return []

    def get_actual_num_ctx(self, model_name: str) -> int:
        """Get the actual num_ctx being used by a running model.

        This reads from /api/ps which shows the real-time context window.
        The /api/ps endpoint returns 'context_length' for running models.

        Args:
            model_name: Model name (or full/short ID)

        Returns:
            int: Actual num_ctx being used, or 0 if model not running
        """
        running = self.get_running_models()
        for model in running:
            # Match by name, short ID, or full ID
            model_name_field = model.get('name', '')
            model_id = model.get('id', '')
            if (model_name_field == model_name or
                model_id.endswith(model_name[-12:]) or
                model_id == model_name):
                # /api/ps returns 'context_length' for running models
                ctx_len = model.get('context_length', 0)
                if ctx_len > 0:
                    return ctx_len
                # Fallback to n_ctx (some Ollama versions)
                return model.get('n_ctx', 0)
        return 0

    def _get_vram_fallback(self, model_name: str) -> Optional[int]:
        """Get VRAM usage from Ollama API /api/ps endpoint as fallback.
        
        This is used when direct GPU monitoring is not available.
        
        Args:
            model_name: Model name to get VRAM for

        Returns:
            Optional[int]: VRAM usage in bytes, or None if unavailable
        """
        return self.get_vram_from_api(model_name)

    def verify_num_ctx(self, model_name: str, expected_ctx: int) -> tuple:
        """Verify that the model's num_ctx matches the expected value.

        Args:
            model_name: Model name
            expected_ctx: Expected num_ctx value

        Returns:
            tuple: (is_verified: bool, actual_ctx: int, message: str)
                is_verified: True if actual ctx matches expected
                actual_ctx: The actual num_ctx value found
                message: Human-readable message describing the result
        """
        actual_ctx = self.get_current_num_ctx(model_name)

        if actual_ctx == expected_ctx:
            message = get_text("ctx_verified", actual=actual_ctx, expected=expected_ctx)
            return True, actual_ctx, message

        message = get_text("ctx_mismatch", actual=actual_ctx, expected=expected_ctx)
        return False, actual_ctx, message

    def verify_num_ctx_via_show(self, model_name: str, expected_ctx: int) -> tuple:
        """Verify num_ctx by parsing the full /api/show response.

        This method looks for context_length in the model's architecture info.

        Args:
            model_name: Model name
            expected_ctx: Expected num_ctx value

        Returns:
            tuple: (is_verified: bool, details: str, found_values: dict)
                is_verified: True if context_length matches expected
                details: Detailed message about the verification
                found_values: Dict with found context-related values
        """
        model_info = self.get_model_info(model_name)
        if not model_info:
            return False, "Could not retrieve model info", {}

        found_values = {}

        # Method 1: Check 'parameters' field for num_ctx
        parameters = model_info.get("parameters", "")
        if parameters:
            for line in parameters.split('\n'):
                line = line.strip()
                if line.startswith('num_ctx:'):
                    try:
                        val = int(line.split(':')[1].strip())
                        found_values['num_ctx_param'] = val
                    except (ValueError, IndexError):
                        pass

        # Method 2: Check model_info keys for context_length or num_ctx
        model_info_dict = model_info.get("model_info", {})
        for key, val in model_info_dict.items():
            key_lower = key.lower()
            if 'context_length' in key_lower or 'num_ctx' in key_lower:
                try:
                    found_values[key] = int(val)
                except (ValueError, TypeError):
                    pass

        # Method 3: Check top-level context_length
        if 'context_length' in model_info:
            try:
                found_values['context_length'] = int(model_info['context_length'])
            except (ValueError, TypeError):
                pass

        # Check if any found value matches expected
        matched_key = None
        actual_value = None
        for key, val in found_values.items():
            if val == expected_ctx:
                matched_key = key
                actual_value = val
                break

        # If no exact match, check if any value >= expected (model can support it)
        if not actual_value:
            for key, val in found_values.items():
                if val >= expected_ctx:
                    matched_key = key
                    actual_value = val
                    break

        if actual_value:
            details = f"   🔍 via /api/show: Found context values: {found_values}"
            if actual_value == expected_ctx:
                details += f" | ✅ Matches expected ({expected_ctx})"
                return True, details, found_values
            elif actual_value >= expected_ctx:
                details += f" | ⚠️  Supports >= expected (found {actual_value} >= {expected_ctx})"
                return True, details, found_values
            else:
                details += f" | ❌ Too small (found {actual_value} < {expected_ctx})"
                return False, details, found_values

        return False, f"   🔍 via /api/show: No context values found in model info", found_values

    def verify_ctx_via_generation(self, model_name: str, context_size: int, prompt: str) -> tuple:
        """Verify context size by sending a test generation and checking prompt_eval_count.

        This method sends a prompt and verifies that the model evaluates all tokens,
        which confirms the context window is large enough.

        Args:
            model_name: Model name
            context_size: Requested context size
            prompt: Test prompt to send

        Returns:
            tuple: (is_verified: bool, eval_count: int, total_duration: float, message: str)
        """
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 10,
                    "num_ctx": context_size
                }
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers=self.headers,
                timeout=60
            )

            if response.status_code != 200:
                return False, 0, 0.0, f"   🧪 Generation test failed: HTTP {response.status_code}"

            data = response.json()
            eval_count = data.get("eval_count", 0)
            prompt_eval_count = data.get("prompt_eval_count", 0)
            total_duration = data.get("total_duration", 0) / 1e9

            # Count tokens in prompt (rough estimate: ~4 chars per token for English)
            estimated_tokens = len(prompt.split())

            if prompt_eval_count >= estimated_tokens:
                msg = (f"   🧪 Generation test: prompt_eval_count={prompt_eval_count}, "
                       f"estimated_tokens={estimated_tokens} | ✅ Context verified")
                return True, prompt_eval_count, total_duration, msg
            else:
                msg = (f"   🧪 Generation test: prompt_eval_count={prompt_eval_count}, "
                       f"estimated_tokens={estimated_tokens} | ⚠️  Not all tokens evaluated")
                return False, prompt_eval_count, total_duration, msg

        except requests.exceptions.Timeout:
            return False, 0, 0.0, "   🧪 Generation test timed out"
        except Exception as e:
            return False, 0, 0.0, f"   🧪 Generation test error: {str(e)}"

    def get_current_num_predict(self, model_name: str) -> int:
        """Get the current default num_predict for a model.

        Args:
            model_name: Model name

        Returns:
            int: Current num_predict value (default: -1)
        """
        model_info = self.get_model_info(model_name)
        if not model_info:
            return -1

        parameters = model_info.get("parameters", "")
        if parameters:
            for line in parameters.split('\n'):
                line = line.strip()
                if line.startswith('num_predict:'):
                    try:
                        val = line.split(':')[1].strip()
                        return int(val)
                    except (ValueError, IndexError):
                        pass

        return -1

    def unload_model(self, model_name: str) -> bool:
        """Unload a model from VRAM to free memory.
        
        Uses the Ollama API to check if the model is loaded and then
        restarts Ollama to free all VRAM.
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            bool: True if unloading was successful, False otherwise
        """
        import subprocess
        import time
        
        try:
            # Check if model is currently loaded
            ps_response = requests.get(
                f"{self.base_url}/api/ps",
                headers=self.headers,
                timeout=10
            )
            
            model_is_loaded = False
            loaded_model_info = None
            if ps_response.status_code == 200:
                ps_data = ps_response.json()
                models = []
                if isinstance(ps_data, dict) and 'models' in ps_data:
                    models = ps_data['models']
                elif isinstance(ps_data, list):
                    models = ps_data
                
                # Check if our model is running
                for m in models:
                    if model_name in m.get('name', ''):
                        model_is_loaded = True
                        loaded_model_info = m
                        break
            
            if model_is_loaded:
                # Model is loaded, need to restart Ollama to free VRAM
                vram_size = loaded_model_info.get('size', 0) / 1024**3 if loaded_model_info else 0
                print(f"   📋 Model '{model_name}' is loaded in VRAM ({vram_size:.1f} GB)")
                
                # Determine restart method based on SSH client availability
                if self.ssh_client and self.ssh_client.is_configured:
                    print(f"   🔄 Restarting Ollama via SSH...")
                    try:
                        exec_result = self.ssh_client.execute("sudo systemctl restart ollama", timeout=60)
                        if exec_result.returncode == 0:
                            print(f"   ⏳ Waiting for Ollama to start...")
                            time.sleep(5)
                            # Verify model is unloaded
                            verify_response = requests.get(
                                f"{self.base_url}/api/ps",
                                headers=self.headers,
                                timeout=10
                            )
                            if verify_response.status_code == 200:
                                verify_data = verify_response.json()
                                verify_models = verify_data.get('models', []) if isinstance(verify_data, dict) else []
                                still_loaded = any(model_name in m.get('name', '') for m in verify_models)
                                if still_loaded:
                                    print(f"   ⚠️  Model '{model_name}' is STILL loaded after restart!")
                                    return False
                                else:
                                    print(f"   ✅ Verified: Model '{model_name}' is unloaded from VRAM.")
                                    return True
                            return True
                        else:
                            print(f"   ⚠️  SSH restart failed: {exec_result.stderr}")
                    except Exception as ssh_err:
                        print(f"   ⚠️  SSH restart error: {ssh_err}")
                else:
                    print(f"   🔄 Restarting Ollama via systemctl...")
                    result = subprocess.run(
                        ['sudo', 'systemctl', 'restart', 'ollama'],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    if result.returncode == 0:
                        print(f"   ⏳ Waiting for Ollama to start...")
                        time.sleep(5)
                        # Verify model is unloaded
                        verify_response = requests.get(
                            f"{self.base_url}/api/ps",
                            headers=self.headers,
                            timeout=10
                        )
                        if verify_response.status_code == 200:
                            verify_data = verify_response.json()
                            verify_models = verify_data.get('models', []) if isinstance(verify_data, dict) else []
                            still_loaded = any(model_name in m.get('name', '') for m in verify_models)
                            if still_loaded:
                                print(f"   ⚠️  Model '{model_name}' is STILL loaded after restart!")
                                return False
                            else:
                                print(f"   ✅ Verified: Model '{model_name}' is unloaded from VRAM.")
                                return True
                        return True
                    else:
                        print(f"   ⚠️  systemctl restart failed: {result.stderr}")
            
            else:
                print(f"   ✅ Model '{model_name}' is not currently loaded in VRAM.")
                return True
            
            return False
            
        except Exception as e:
            print(f"   ⚠️  Error unloading model: {e}")
            # Fallback: try direct systemctl restart
            try:
                print(f"   🔄 Trying direct systemctl restart...")
                result = subprocess.run(
                    ['sudo', 'systemctl', 'restart', 'ollama'],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode == 0:
                    time.sleep(5)
                    print(f"   🧹 Ollama restarted via fallback, VRAM freed.")
                    return True
            except Exception:
                pass
            
            print(f"   ⚠️  Could not unload model '{model_name}' automatically.")
            print(f"   💡 Tip: Run 'sudo systemctl restart ollama' to free VRAM.")
            return False

    def unload_all_models(self) -> bool:
        """Unload all models from VRAM by restarting Ollama.
        
        Returns:
            bool: True if restart was successful, False otherwise
        """
        try:
            print(f"   🧹 Unloading all models from VRAM...")
            from system.restart_manager import restart_ollama, RestartMethod
            
            # Determine restart method based on SSH client availability
            if hasattr(self, 'ssh_client') and self.ssh_client and self.ssh_client.is_configured:
                restart_method = RestartMethod.SSH
            else:
                restart_method = RestartMethod.MANUAL
            
            restart_ollama(
                method=restart_method,
                no_restart=False,
                ssh_client=getattr(self, 'ssh_client', None)
            )
            return True
        except Exception as e:
            print(f"   ⚠️  Error unloading all models: {e}")
            return False
