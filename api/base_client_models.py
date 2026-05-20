"""Model management methods for BaseApiClient."""

import logging

import requests

from i18n import get_text

logger = logging.getLogger('roo_bench')


class BaseApiClientModels:
    """Mixin class for model management methods."""

    base_url: str
    headers: dict
    timeout: int

    def get_models(self) -> list[dict]:
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
                return data.get("models", [])  # type: ignore[no-any-return]
            return []
        except Exception as e:
            print(get_text("error_ollama_connection", error=str(e)))
            return []

    def get_default_temperature(self, model_name: str) -> float:
        """Get the default temperature for a model.

        Args:
            model_name: Model name

        Returns:
            float: Default temperature value (typically 0.1-0.8)
        """
        model_info = self.get_model_info(model_name)
        if not model_info:
            return 0.1  # Default fallback

        parameters = model_info.get("parameters", "")
        if parameters:
            for line in parameters.split('\n'):
                line = line.strip()
                if line.startswith('temperature:'):
                    try:
                        return float(line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        pass

        # Check model_info keys for temperature
        model_info_dict = model_info.get("model_info", {})
        for key, val in model_info_dict.items():
            if 'temperature' in key.lower():
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass

        return 0.1  # Default fallback

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
                return response.json()  # type: ignore[no-any-return]
            return {}
        except Exception as e:
            print(f"\u26a0\ufe0f  Error getting model info for {model_name}: {e}")
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

    def get_running_models(self) -> list[dict]:
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
                    models_data = data['models']
                    if isinstance(models_data, list):
                        return models_data
                elif isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    return [data]
                return []
            return []
        except Exception as e:
            print(f"\u26a0\ufe0f  Error getting running models: {e}")
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
                    return int(ctx_len)
                # Fallback to n_ctx (some Ollama versions)
                return int(model.get('n_ctx', 0))
        return 0

    def _get_vram_fallback(self, model_name: str) -> int | None:
        """Get VRAM usage from Ollama API /api/ps endpoint as fallback.

        This is used when direct GPU monitoring is not available.

        Args:
            model_name: Model name to get VRAM for

        Returns:
            Optional[int]: VRAM usage in bytes, or None if unavailable
        """
        return None

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
