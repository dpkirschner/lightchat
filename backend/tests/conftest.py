"""Pytest configuration and shared test fixtures for the LightChat backend."""
import os
import sys
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.main import app
from backend.config import AppConfig, APP_NAME, APP_AUTHOR


@pytest.fixture(scope="session")
def app_config(tmp_path_factory):
    """Create a test app config with a temporary directory for logs and data."""
    temp_dir = tmp_path_factory.mktemp("test_data")
    return AppConfig(
        log_dir=str(temp_dir / "logs"),
        data_dir=str(temp_dir / "data"),
        logging_enabled=True,
        debug=True
    )


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI application."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def mock_settings(monkeypatch, tmp_path):
    """Patch settings for testing."""
    # Create a test config directory
    test_config_dir = tmp_path / "config"
    test_config_dir.mkdir()
    
    # Set environment variables for config paths
    monkeypatch.setenv("LIGHTCHAT_CONFIG_DIR", str(test_config_dir))
    
    # Create a test config
    test_config = AppConfig(
        log_dir=str(tmp_path / "logs"),
        data_dir=str(tmp_path / "data"),
        logging_enabled=True,
        debug=True
    )
    
    return test_config


@pytest.fixture
def mock_providers():
    """Mock the providers module for testing."""
    with patch('backend.main.providers') as mock_providers:
        yield mock_providers


@pytest.fixture
def mock_chat_engine():
    """Mock the chat_engine module for testing."""
    with patch('backend.main.chat_engine') as mock_chat_engine:
        yield mock_chat_engine


@pytest.fixture
def mock_app_config(app_config):
    """Create a mock app config for testing with a temporary directory."""
    # Create a mock app config with test settings
    from backend.config import ProviderConfig
    
    mock_config = AppConfig(
        default_provider="ollama_default",
        providers=[
            ProviderConfig(
                id="ollama_default",
                name="Ollama",
                type="ollama",
                host="http://localhost:11434"
            ),
            ProviderConfig(
                id="openai_default",
                name="OpenAI",
                type="openai",
                api_key="test_key"
            )
        ],
        log_dir=app_config.log_dir,
        data_dir=app_config.data_dir,
        logging_enabled=True,
        debug=True
    )
    return mock_config
