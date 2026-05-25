"""Utility methods for BaseApiClient (verification, unloading)."""

import logging
import subprocess
import time
from typing import Any

import requests

from i18n import get_text

logger = logging.getLogger('roo_bench')


class BaseApiClientUtils:
    """Mixin class for utility methods."""

    base_url: str
    headers: dict
    ssh_client: Any

    def verify_num_ctx(self, model_name: str, expected_ctx: int) -> tuple[bool, int, str]:
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
            message = get_text("ctx_verified", actual=str(actual_ctx), expected=str(expected_ctx))
            return True, actual_ctx, message

        message = get_text("ctx_mismatch", actual=str(actual_ctx), expected=str(expected_ctx))
        return False, actual_ctx, message

    def verify_num_ctx_via_show(self, model_name: str, expected_ctx: int) -> tuple[bool, str, dict[str, Any]]:
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
        actual_value = None
        for key, val in found_values.items():
            if val == expected_ctx:
                actual_value = val
                break

        # If no exact match, check if any value >= expected (model can support it)
        if not actual_value:
            for key, val in found_values.items():
                if val >= expected_ctx:
                    actual_value = val
                    break

        if actual_value:
            details = f"   \U0001f50d via /api/show: Found context values: {found_values}"
            if actual_value == expected_ctx:
                details += f" | \u2705 Matches expected ({expected_ctx})"
                return True, details, found_values
            elif actual_value >= expected_ctx:
                details += f" | \u26a0\ufe0f  Supports >= expected (found {actual_value} >= {expected_ctx})"
                return True, details, found_values
            else:
                details += f" | \u274c Too small (found {actual_value} < {expected_ctx})"
                return False, details, found_values

        return False, "   \U0001f50d via /api/show: No context values found in model info", found_values

    def verify_ctx_via_generation(self, model_name: str, context_size: int, prompt: str) -> tuple[bool, int, float, str]:
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
                    "num_ctx": context_size,
                    "num_predict": 8192  # Sufficient for most code files
                }
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers=self.headers,
                timeout=60
            )

            if response.status_code != 200:
                return False, 0, 0.0, f"   \U0001f9ea Generation test failed: HTTP {response.status_code}"

            data = response.json()
            _eval_count = data.get("eval_count", 0)
            prompt_eval_count = data.get("prompt_eval_count", 0)
            total_duration = data.get("total_duration", 0) / 1e9

            # Count tokens in prompt (rough estimate: ~4 chars per token for English)
            estimated_tokens = len(prompt.split())

            if prompt_eval_count >= estimated_tokens:
                msg = (f"   \U0001f9ea Generation test: prompt_eval_count={prompt_eval_count}, "
                       f"estimated_tokens={estimated_tokens} | \u2705 Context verified")
                return True, prompt_eval_count, total_duration, msg
            else:
                msg = (f"   \U0001f9ea Generation test: prompt_eval_count={prompt_eval_count}, "
                       f"estimated_tokens={estimated_tokens} | \u26a0\ufe0f  Not all tokens evaluated")
                return False, prompt_eval_count, total_duration, msg

        except requests.exceptions.Timeout:
            return False, 0, 0.0, "   \U0001f9ea Generation test timed out"
        except Exception as e:
            return False, 0, 0.0, f"   \U0001f9ea Generation test error: {str(e)}"

    def unload_model(self, model_name: str) -> bool:
        """Unload a model from VRAM to free memory.
        
        Uses the Ollama API to check if the model is loaded and then
        restarts Ollama to free all VRAM.
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            bool: True if unloading was successful, False otherwise
        """
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
                print(f"   \U0001f4cb Model '{model_name}' is loaded in VRAM ({vram_size:.1f} GB)")
                
                # Determine restart method based on SSH client availability
                if self.ssh_client and self.ssh_client.is_configured:
                    print("   \U0001f504 Restarting Ollama via SSH...")
                    try:
                        exec_result = self.ssh_client.execute("sudo systemctl restart ollama", timeout=60)
                        if exec_result.returncode == 0:
                            print("   \u23f3 Waiting for Ollama to start...")
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
                                    print(f"   \u26a0\ufe0f  Model '{model_name}' is STILL loaded after restart!")
                                    return False
                                else:
                                    print(f"   \u2705 Verified: Model '{model_name}' is unloaded from VRAM.")
                                    return True
                            return True
                        else:
                            print(f"   \u26a0\ufe0f  SSH restart failed: {exec_result.stderr}")
                    except Exception as ssh_err:
                        print(f"   \u26a0\ufe0f  SSH restart error: {ssh_err}")
                else:
                    print("   \U0001f504 Restarting Ollama via systemctl...")
                    result = subprocess.run(
                        ['sudo', 'systemctl', 'restart', 'ollama'],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    if result.returncode == 0:
                        print("   \u23f3 Waiting for Ollama to start...")
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
                                print(f"   \u26a0\ufe0f  Model '{model_name}' is STILL loaded after restart!")
                                return False
                            else:
                                print(f"   \u2705 Verified: Model '{model_name}' is unloaded from VRAM.")
                                return True
                        return True
                    else:
                        print(f"   \u26a0\ufe0f  systemctl restart failed: {result.stderr}")
            
            else:
                print(f"   \u2705 Model '{model_name}' is not currently loaded in VRAM.")
                return True
            
            return False
            
        except Exception as e:
            print(f"   \u26a0\ufe0f  Error unloading model: {e}")
            # Fallback: try direct systemctl restart
            try:
                print("   \U0001f504 Trying direct systemctl restart...")
                result = subprocess.run(
                    ['sudo', 'systemctl', 'restart', 'ollama'],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode == 0:
                    time.sleep(5)
                    print("   \U0001f9f9 Ollama restarted via fallback, VRAM freed.")
                    return True
            except Exception:
                pass
            
            print(f"   \u26a0\ufe0f  Could not unload model '{model_name}' automatically.")
            print("   \U0001f4a1 Tip: Run 'sudo systemctl restart ollama' to free VRAM.")
            return False

    def unload_all_models(self) -> bool:
        """Unload all models from VRAM by restarting Ollama.
        
        Returns:
            bool: True if restart was successful, False otherwise
        """
        try:
            print("   \U0001f9f9 Unloading all models from VRAM...")
            from system.restart_manager import RestartMethod, restart_ollama
            
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
            print(f"   \u26a0\ufe0f  Error unloading all models: {e}")
            return False
