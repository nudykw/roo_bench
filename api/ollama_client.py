"""Ollama API communication client."""

import math
import requests
from system.gpu_monitor import get_vram_usage
from i18n import get_text


class OllamaClient:
    """Client for communicating with Ollama API."""

    def __init__(self, base_url: str, headers: dict = None, timeout: int = 300):
        """Initialize Ollama client.

        Args:
            base_url: Ollama API base URL
            headers: HTTP headers (e.g., authentication)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.headers = headers or {}
        self.timeout = timeout

    def get_models(self) -> list:
        """Fetch available models with details.

        Returns:
            list: List of model dictionaries with name, params, quant, size_gb, max_ctx
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", headers=self.headers)
            models = []
            for m in response.json().get("models", []):
                details = m.get("details", {})
                size_gb = m.get("size", 0) / (1024 ** 3)

                # Get the maximum context size of the model via API show
                max_ctx = 32768  # Default value if not found
                show_error = None
                try:
                    show_resp = requests.post(f"{self.base_url}/api/show", json={"name": m["name"]})
                    if show_resp.status_code == 200:
                        model_info = show_resp.json().get("model_info", {})
                        # Look for key containing 'context_length' (e.g. 'llama.context_length' or 'qwen2.context_length')
                        for key, val in model_info.items():
                            if 'context_length' in key:
                                max_ctx = int(val)
                                break
                except Exception as e:
                    show_error = f"Error getting max_ctx for {m['name']}: {e}"
                    print(f"⚠️  {show_error}")

                models.append({
                    "name": m["name"],
                    "params": details.get("parameter_size", "N/A"),
                    "quant": details.get("quantization_level", "N/A"),
                    "size_gb": size_gb,
                    "max_ctx": max_ctx
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
                "stream": False,
                "options": {
                    "num_predict": 100,
                    "num_ctx": context_size
                }
            }

            response = None

            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout
                )

                if response.status_code != 200:
                    try:
                        err_msg = response.json().get("error", f"HTTP {response.status_code}")
                    except:
                        err_msg = f"HTTP {response.status_code}: {response.text[:100]}"
                    error = get_text("error_ollama_api", error_msg=err_msg)
                    break

                try:
                    data = response.json()
                except Exception as json_err:
                    raw_text = response.text[:200].replace('\n', ' ')
                    error = get_text("error_parsing_response", json_err=json_err, raw_text=raw_text)
                    break

                vram_after = get_vram_usage()
                total_duration = data.get("total_duration", 0) / 1e9
                eval_count = data.get("eval_count", 0)
                tps = eval_count / total_duration if total_duration > 0 else 0
                tps_list.append({"run": run_num + 1, "tps": tps, "vram": vram_after})

            except requests.exceptions.Timeout:
                error = get_text("error_timeout")
                break
            except requests.exceptions.ConnectionError:
                error = get_text("error_crash")
                break
            except Exception as e:
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
