"""Ollama API communication client."""

import json
import math
import subprocess
import threading
import requests
from system.gpu_monitor import get_vram_usage
from i18n import get_text


def get_vram_usage_via_ssh(ssh_host: str, ssh_user: str, ssh_port: int = 22, ssh_key: str = None) -> int | None:
    """Get VRAM usage from remote machine via SSH.

    Args:
        ssh_host: Remote SSH host
        ssh_user: Remote SSH user
        ssh_port: Remote SSH port
        ssh_key: Path to SSH private key

    Returns:
        int or None: VRAM usage in bytes, or None if unavailable
    """
    ssh_key_arg = f"-i {ssh_key}" if ssh_key else ""
    cmd = f'ssh -t -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p {ssh_port} {ssh_key_arg} {ssh_host} "nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader" 2>/dev/null'
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            # nvidia-smi returns value in MiB, convert to bytes
            return int(result.stdout.strip()) * 1024 * 1024
    except Exception:
        pass
    
    return None


class OllamaClient:
    """Client for communicating with Ollama API."""

    def __init__(self, base_url: str, headers: dict = None, timeout: int = 300,
                 ssh_host: str = None, ssh_user: str = None,
                 ssh_port: int = 22, ssh_key: str = None):
        """Initialize Ollama client.

        Args:
            base_url: Ollama API base URL
            headers: HTTP headers (e.g., authentication)
            timeout: Request timeout in seconds
            ssh_host: SSH host for remote VRAM monitoring
            ssh_user: SSH user for remote VRAM monitoring
            ssh_port: SSH port for remote VRAM monitoring
            ssh_key: Path to SSH private key for remote VRAM monitoring
        """
        self.base_url = base_url
        self.headers = headers or {}
        self.timeout = timeout
        self.ssh_host = ssh_host
        self.ssh_user = ssh_user
        self.ssh_port = ssh_port
        self.ssh_key = ssh_key

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
        model_name_combined = f"{architecture} {basename}"
        
        for key in model_info.keys():
            key_lower = key.lower()
            
            # Vision: keys containing '.vision.' (e.g., 'gemma4.vision.block_count', 'qwen35.vision.embedding_length')
            if '.vision.' in key_lower or key_lower.endswith('.vision'):
                capabilities['vision'] = True
            
            # Audio: keys containing '.audio.' (e.g., 'gemma4.audio.attention.head_count')
            if '.audio.' in key_lower or key_lower.endswith('.audio'):
                capabilities['audio'] = True
            
            # Tools: keys containing 'tool' or 'function' (e.g., 'tool.function.tool_use')
            if 'tool' in key_lower or 'function' in key_lower:
                capabilities['tools'] = True
            
            # Thinking: keys containing 'reasoning' or 'thinking'
            if 'reasoning' in key_lower or 'thinking' in key_lower:
                capabilities['thinking'] = True
        
        # Heuristic detection based on architecture and known model families
        # These models ALWAYS have tools support (function calling)
        TOOLS_ARCHITECTURES = [
            'llama', 'qwen2', 'qwen2.5', 'qwen3', 'qwen35', 'qwen35moe',
            'mistral', 'mixtral', 'nemotron', 'nemotron_h', 'phi3', 'phi4',
            'granite', 'gemma', 'gemma3', 'gemma4', 'deepseek', 'deepseek2',
            'gptoss', 'claude', 'claude-oss'
        ]
        
        # These models have thinking/reasoning capability
        THINKING_ARCHITECTURES = [
            'qwen3', 'qwen35', 'qwen35moe',
            'deepseek', 'deepseek2', 'deepseek-r1',
            'qwq', 'o1', 'o3', 'claude', 'claude-oss',
            'nemotron', 'nemotron_h'
        ]
        
        # Check if architecture matches known families
        for arch in TOOLS_ARCHITECTURES:
            if arch in model_name_combined or arch in architecture:
                capabilities['tools'] = True
                break
        
        # Special case: qwen3.5 and qwen3.6 have both vision and tools and thinking
        if 'qwen35' in architecture or 'qwen35moe' in architecture:
            capabilities['tools'] = True
            capabilities['thinking'] = True  # Qwen3.5/3.6 have thinking
        
        for arch in THINKING_ARCHITECTURES:
            if arch in model_name_combined or arch in architecture:
                capabilities['thinking'] = True
                break
        
        return capabilities

    def get_models(self) -> list:
        """Fetch available models with details.

        Returns:
            list: List of model dictionaries with name, params, quant, size_gb, max_ctx, and capabilities
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", headers=self.headers)
            models = []
            for m in response.json().get("models", []):
                details = m.get("details", {})
                size_gb = m.get("size", 0) / (1024 ** 3)

                # Get the maximum context size and capabilities via API show
                max_ctx = 32768  # Default value if not found
                capabilities = {'vision': False, 'tools': False, 'thinking': False, 'audio': False}
                
                try:
                    show_resp = requests.post(f"{self.base_url}/api/show", json={"name": m["name"]})
                    if show_resp.status_code == 200:
                        show_data = show_resp.json()
                        model_info = show_data.get("model_info", {})
                        
                        # Look for key containing 'context_length' (e.g. 'llama.context_length' or 'qwen2.context_length')
                        for key, val in model_info.items():
                            if 'context_length' in key:
                                max_ctx = int(val)
                                break
                        
                        # Extract capabilities from model_info
                        capabilities = self.get_capabilities_from_model_info(model_info)
                except Exception as e:
                    show_error = f"Error getting details for {m['name']}: {e}"
                    print(f"⚠️  {show_error}")

                models.append({
                    "name": m["name"],
                    "params": details.get("parameter_size", "N/A"),
                    "quant": details.get("quantization_level", "N/A"),
                    "size_gb": size_gb,
                    "max_ctx": max_ctx,
                    "capabilities": capabilities
                })
            return models
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
            max_vram = 0
            stop_monitoring = threading.Event()
            vram_samples = []

            # Determine which VRAM monitoring function to use
            # SSH host can be in format "user@host" or just "host"
            use_ssh = bool(self.ssh_host)
            
            # Extract user from ssh_host if not provided separately
            ssh_user = self.ssh_user
            if not ssh_user and self.ssh_host and '@' in self.ssh_host:
                ssh_user = self.ssh_host.split('@')[0]
            
            vram_args = (self.ssh_host, ssh_user, self.ssh_port, self.ssh_key)

            def monitor_vram():
                """Monitor VRAM during generation and collect samples."""
                nonlocal max_vram
                while not stop_monitoring.is_set():
                    if use_ssh:
                        v = get_vram_usage_via_ssh(*vram_args)
                    else:
                        v = get_vram_usage()
                    if v is not None:
                        vram_samples.append(v)
                        if v > max_vram:
                            max_vram = v
                    # Sample every 500ms (SSH calls are slower)
                    stop_monitoring.wait(0.5 if use_ssh else 0.2)

            try:
                # Start VRAM monitoring thread
                monitor_thread = threading.Thread(target=monitor_vram, daemon=True)
                monitor_thread.start()

                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout,
                    stream=True
                )

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
                vram_during = max_vram if vram_samples else None
                tps_list.append({"run": run_num + 1, "tps": tps, "vram": vram_during})

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
                if response is not None:
                    err_details += f" | Raw response: {response.text[:200]}"
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
