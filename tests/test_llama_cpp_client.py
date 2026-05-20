"""Tests for llama.cpp API clients."""

import threading
from unittest.mock import patch

import pytest

from api.llama_cpp_client import LlamaCppApiClient, LlamaCppRemoteApiClient


class TestLlamaCppApiClient:
    """Tests for local llama.cpp client."""

    @pytest.fixture
    def client(self):
        return LlamaCppApiClient(base_url="http://localhost:8080", timeout=30)

    def test_is_remote_false(self, client):
        """Test that local client reports is_remote=False."""
        assert client.is_remote is False

    def test_ssh_client_none(self, client):
        """Test that local client has no ssh_client."""
        assert client.ssh_client is None

    @patch("api.llama_cpp_client.requests.get")
    def test_get_models_success(self, mock_get, client):
        """Test get_models with successful response."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "data": [
                {"id": "llama3:8b", "object": "model"},
                {"id": "mistral:7b", "object": "model"},
            ]
        }
        models = client.get_models()
        assert len(models) == 2
        assert models[0]["name"] == "llama3:8b"
        assert models[1]["name"] == "mistral:7b"

    @patch("api.llama_cpp_client.requests.get")
    def test_get_models_empty(self, mock_get, client):
        """Test get_models with empty response."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"data": []}
        models = client.get_models()
        assert models == []

    @patch("api.llama_cpp_client.requests.get")
    def test_get_models_connection_error(self, mock_get, client):
        """Test get_models with connection error."""
        mock_get.side_effect = Exception("Connection refused")
        models = client.get_models()
        assert models == []

    @patch("api.llama_cpp_client.requests.get")
    def test_get_model_info(self, mock_get, client):
        """Test get_model_info with successful response."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "id": "llama3:8b",
            "object": "model",
        }
        info = client.get_model_info("llama3:8b")
        assert info["name"] == "llama3:8b"

    @patch("api.llama_cpp_client.requests.get")
    def test_get_model_info_error(self, mock_get, client):
        """Test get_model_info with error."""
        mock_get.side_effect = Exception("Not found")
        info = client.get_model_info("nonexistent")
        assert info == {}

    def test_monitor_vram_stops_on_event(self, client):
        """Test that _monitor_vram stops when event is set."""
        stop_event = threading.Event()
        max_vram_ref: list[float] = [0.0]
        vram_samples: list[float] = []

        # Set stop event immediately
        stop_event.set()

        # Should return immediately without hanging
        client._monitor_vram(stop_event, max_vram_ref, vram_samples)


class TestLlamaCppRemoteApiClient:
    """Tests for remote llama.cpp client."""

    @pytest.fixture
    def ssh_client(self):
        from system.ssh_client import SSHClient

        return SSHClient(host="remote-server", user="test", port=22)

    @pytest.fixture
    def client(self, ssh_client):
        return LlamaCppRemoteApiClient(
            base_url="http://remote-server:8080",
            ssh_client=ssh_client,
        )

    def test_is_remote_true(self, client):
        """Test that remote client reports is_remote=True."""
        assert client.is_remote is True

    def test_has_ssh_client(self, client):
        """Test that remote client has ssh_client."""
        assert client.ssh_client is not None

    def test_inherits_from_local_client(self, client):
        """Test that remote client inherits from LlamaCppApiClient."""
        assert isinstance(client, LlamaCppApiClient)


class TestFactoryIntegration:
    """Tests for ApiClientFactory with llama.cpp backend."""

    def test_factory_creates_llama_cpp_local(self):
        """Test factory creates LlamaCppApiClient for llama_cpp backend."""
        from api.factory import ApiClientFactory

        client = ApiClientFactory.create_client(
            base_url="http://localhost:8080",
            backend_type="llama_cpp",
        )
        assert isinstance(client, LlamaCppApiClient)
        assert client.is_remote is False

    def test_factory_creates_llama_cpp_remote(self):
        """Test factory creates LlamaCppRemoteApiClient for remote llama_cpp."""
        from api.factory import ApiClientFactory

        client = ApiClientFactory.create_client(
            base_url="http://remote:8080",
            backend_type="llama_cpp",
            ssh_host="remote",
            ssh_user="test",
        )
        assert isinstance(client, LlamaCppRemoteApiClient)
        assert client.is_remote is True

    def test_factory_default_ollama(self):
        """Test factory creates LocalApiClient for default ollama backend."""
        from api.factory import ApiClientFactory
        from api.local_client import LocalApiClient

        client = ApiClientFactory.create_client(
            base_url="http://localhost:11434",
        )
        assert isinstance(client, LocalApiClient)

    def test_factory_ollama_remote(self):
        """Test factory creates RemoteApiClient for remote ollama."""
        from api.factory import ApiClientFactory
        from api.remote_client import RemoteApiClient

        client = ApiClientFactory.create_client(
            base_url="http://remote:11434",
            backend_type="ollama",
            ssh_host="remote",
            ssh_user="test",
        )
        assert isinstance(client, RemoteApiClient)
        assert client.is_remote is True


class TestConfigIntegration:
    """Tests for config.py backend_type support."""

    def test_default_backend_type(self):
        """Test default backend_type is 'ollama'."""
        from config import OllamaConfig

        config = OllamaConfig(cli_args={})
        assert config.backend_type == "ollama"

    def test_backend_type_from_cli(self):
        """Test backend_type can be set via CLI args."""
        from config import OllamaConfig

        config = OllamaConfig(cli_args={"backend_type": "llama_cpp"})
        assert config.backend_type == "llama_cpp"

    def test_backend_type_from_env(self, monkeypatch):
        """Test backend_type can be set via environment variable."""
        from config import OllamaConfig

        monkeypatch.setenv("ROO_BENCH_BACKEND_TYPE", "llama_cpp")
        config = OllamaConfig(cli_args={})
        assert config.backend_type == "llama_cpp"
