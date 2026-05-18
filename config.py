import json
import os
from typing import Any


class OllamaConfig:
    """Ollama server connection configuration"""
    
    DEFAULTS = {
        'url': 'http://localhost:11434',
        'port': 11434,
        'api_key': None,
        'timeout': 300,
        'config_file': 'config.json'
    }
    
    def __init__(self, cli_args: dict[str, Any] | None = None):
        self.cli_args = cli_args or {}
        self._config = self._load_config()
        self._apply_env_vars()
        self._apply_cli_args()
    
    def _load_config(self) -> dict[str, Any]:
        """Load configuration from file"""
        config_file = self.cli_args.get('config') or self.DEFAULTS['config_file']
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error reading configuration: {e}")
                return {}
        return {}
    
    def _apply_env_vars(self):
        """Apply environment variables"""
        env_vars = {
            'url': 'OLLAMA_URL',
            'port': 'OLLAMA_PORT',
            'api_key': 'OLLAMA_API_KEY',
            'timeout': 'OLLAMA_TIMEOUT',
            'temperature_test_values': 'OLLAMA_TEMPERATURE_TEST_VALUES'
        }
        
        for key, env_name in env_vars.items():
            env_value = os.environ.get(env_name)
            if env_value is not None:
                self._config[key] = env_value
    
    def _apply_cli_args(self):
        """Apply command-line arguments"""
        cli_vars = {
            'url': 'ollama_url',
            'port': 'ollama_port',
            'api_key': 'ollama_api_key',
            'timeout': 'ollama_timeout',
            'temperature_test_values': 'temperature_test_values'
        }
        
        for key, cli_name in cli_vars.items():
            cli_value = self.cli_args.get(cli_name)
            if cli_value is not None:
                self._config[key] = cli_value
    
    @property
    def url(self) -> str:
        """Get server URL"""
        return self._config.get('url', self.DEFAULTS['url'])
    
    @property
    def port(self) -> int:
        """Get port"""
        return int(self._config.get('port', self.DEFAULTS['port']))
    
    @property
    def api_key(self) -> str | None:
        """Get API key"""
        return self._config.get('api_key')
    
    @property
    def timeout(self) -> int:
        """Get timeout"""
        return int(self._config.get('timeout', self.DEFAULTS['timeout']))
    
    @property
    def temperature_test_values(self) -> list:
        """Get temperature test values from config or default"""
        from benchmark.runner import DEFAULT_TEMPERATURES
        raw = self._config.get('temperature_test_values')
        if raw:
            # Try to parse as JSON list
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                pass
        # Default values
        return DEFAULT_TEMPERATURES

    @property
    def monitor_config(self) -> dict:
        """Get monitoring configuration"""
        defaults = {
            'collection_interval': 0.5,
            'max_samples': 1000,
            'enable_cpu_monitoring': True,
            'enable_ram_monitoring': True,
            'enable_vram_monitoring': True
        }
        raw = self._config.get('monitor_config')
        if raw:
            # Try to parse as JSON
            try:
                config = json.loads(raw)
                # Merge with defaults
                for key, value in defaults.items():
                    if key not in config:
                        config[key] = value
                return config
            except (json.JSONDecodeError, TypeError):
                pass
        return defaults
    
    @property
    def base_url(self) -> str:
        """Get base URL (considering port)"""
        if self.port == 11434 and 'localhost' in self.url and ':11434' not in self.url:
            return f"{self.url.replace('localhost', '127.0.0.1')}:11434"
        return self.url
    
    @property
    def prompts_file(self) -> str:
        """Get path to prompts configuration file."""
        return self._config.get('prompts_file', os.path.join(os.path.dirname(__file__), 'prompts.jsonc'))
    
    def get_headers(self) -> dict[str, str]:
        """Get headers for requests"""
        headers = {}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers
