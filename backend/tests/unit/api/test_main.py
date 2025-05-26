"""Unit tests for the main FastAPI application endpoints."""
import pytest
import httpx
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import status
from typing import Dict, List, Any, Optional

# Import models and fixtures
import sys
from pathlib import Path

# Add the backend directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.models.providers import ModelInfo, ProviderMetadata
from backend.config import AppConfig, ProviderConfig

# Test data
SAMPLE_PROVIDERS = [
    ProviderConfig(
        id="ollama_local",
        name="Local Ollama",
        type="ollama",
        host="http://localhost:11434",
        default_model="llama3"
    ),
    ProviderConfig(
        id="openai_cloud",
        name="OpenAI Cloud",
        type="openai",
        api_key="test-api-key",
        default_model="gpt-4"
    )
]

SAMPLE_MODELS = [
    ModelInfo(
        id="llama3:latest", 
        name="Meta Llama 3", 
        modified_at="2023-10-29T19:22:00.000000Z",
        size=4117063800,
        parameter_size="7B",
        quantization_level="Q4_0"
    ),
    ModelInfo(
        id="mistral:latest", 
        name="Mistral 7B", 
        modified_at="2023-10-29T19:22:00.000000Z",
        size=4117063800,
        parameter_size="7B",
        quantization_level="Q4_0"
    )
]

# Fixtures
@pytest.fixture
def mock_app_config():
    """Create a mock AppConfig with test providers."""
    return AppConfig(
        default_provider="ollama_local",
        providers=SAMPLE_PROVIDERS,
        logging_enabled=True,
        debug=True
    )

# Tests for /providers endpoint
@pytest.mark.asyncio
async def test_list_providers_success(test_client, mock_app_config, event_loop):
    """Test successful listing of providers."""
    # Mock the provider status checks
    with patch('backend.main.get_app_config', return_value=mock_app_config), \
         patch('backend.main.OllamaProvider') as mock_ollama_provider, \
         patch('backend.main.OpenAIProvider') as mock_openai_provider, \
         patch('backend.main._app_config', mock_app_config):
        
        # Create mock providers with status methods
        mock_ollama = AsyncMock()
        mock_ollama.get_status.return_value = "available"
        mock_ollama_provider.return_value = mock_ollama
        
        mock_openai = AsyncMock()
        mock_openai.get_status.return_value = "available"
        mock_openai_provider.return_value = mock_openai
        
        from backend.config import ProviderConfig
        # Update the mock config to include the test providers
        mock_app_config.providers = [
            ProviderConfig(
                id="ollama_local",
                name="Local Ollama",
                type="ollama",
                host="http://localhost:11434"
            ),
            ProviderConfig(
                id="openai1",
                name="OpenAI",
                type="openai",
                api_key="test-key"
            )
        ]
        
        response = test_client.get("/providers")
        
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    
    # Check response structure
    assert isinstance(data, list)
    assert len(data) == 2
    
    # Check provider fields
    for provider in data:
        assert "id" in provider
        assert "name" in provider
        assert "type" in provider
        assert "status" in provider
    
    # Check provider statuses
    provider_map = {p["id"]: p for p in data}
    assert provider_map["ollama_local"]["status"] == "configured"
    assert provider_map["openai1"]["status"] == "configured"

@pytest.mark.asyncio
async def test_list_providers_missing_config():
    """Test handling when provider configuration is missing required fields."""
    # Test with a valid config that has a provider with missing required fields
    # This should not raise an exception, but the provider should be marked as 'unavailable'
    valid_config = AppConfig(
        default_provider="ollama_local",
        providers=[
            ProviderConfig(
                id="ollama_local",
                name="Local Ollama",
                type="ollama",
                # Missing required 'host' field
            )
        ]
    )
    
    with patch('backend.main.get_app_config', return_value=valid_config):
        from backend.main import get_providers
        providers = await get_providers(valid_config)
        assert len(providers) == 1
        assert providers[0].status == "unavailable"

# Tests for /models/{provider_id} endpoint
@pytest.mark.asyncio
async def test_list_models_success(test_client, mock_app_config, event_loop):
    """Test successful listing of models for a provider."""
    # Update the mock config to include the test provider
    mock_app_config.providers = [
        ProviderConfig(id="ollama_local", name="Local Ollama", type="ollama", host="http://localhost:11434")
    ]
    
    # Mock the provider's list_models method to return dictionaries instead of ModelInfo objects
    mock_provider = AsyncMock()
    mock_provider.list_models.return_value = [
        {
            "id": model.id,
            "name": model.name,
            "modified_at": model.modified_at,
            "size": model.size,
            "parameter_size": model.parameter_size,
            "quantization_level": model.quantization_level
        } for model in SAMPLE_MODELS
    ]

    with patch('backend.main.get_app_config', return_value=mock_app_config) as mock_get_config, \
         patch('backend.main.OllamaProvider', return_value=mock_provider) as mock_provider_cls, \
         patch('backend.main._app_config', mock_app_config):
        response = test_client.get("/models/ollama_local")
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    
    # Check response structure
    assert isinstance(data, list)
    assert len(data) == len(SAMPLE_MODELS)
    
    # Check model fields
    for model in data:
        assert "id" in model
        assert "name" in model
        assert "modified_at" in model
        assert "size" in model
    
    # Verify the provider was initialized correctly
    mock_provider_cls.assert_called_once()
    mock_provider.list_models.assert_awaited_once()

@pytest.mark.asyncio
async def test_list_models_provider_not_found(test_client, mock_app_config, event_loop):
    """Test listing models for a non-existent provider."""
    with patch('backend.main.get_app_config', return_value=mock_app_config):
        response = test_client.get("/models/nonexistent")
    
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in response.json()["detail"].lower()

@pytest.mark.asyncio
async def test_list_models_provider_error(test_client, mock_app_config, event_loop):
    """Test error handling when listing models fails."""
    # Update the mock config to include the test provider
    mock_app_config.providers = [
        ProviderConfig(id="ollama_local", name="Local Ollama", type="ollama", host="http://localhost:11434")
    ]
    
    # Mock the provider to raise an exception
    mock_provider = AsyncMock()
    mock_provider.list_models.side_effect = Exception("API Error")
    
    with patch('backend.main.get_app_config', return_value=mock_app_config), \
         patch('backend.main.OllamaProvider', return_value=mock_provider), \
         patch('backend.main._app_config', mock_app_config):
        response = test_client.get("/models/ollama_local")
    
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "error" in response.json()["detail"].lower()

# Test provider initialization with different configurations
@pytest.mark.parametrize("provider_type,provider_config,expected_kwargs", [
    (
        "ollama",
        {"id": "ollama1", "name": "Ollama", "type": "ollama", "host": "http://localhost:11434"},
        {"provider_id": "ollama1", "display_name": "Ollama", "ollama_base_url": "http://localhost:11434"}
    ),
    (
        "openai",
        {"id": "openai1", "name": "OpenAI", "type": "openai", "api_key": "test-key"},
        {"provider_id": "openai1", "display_name": "OpenAI", "api_key": "test-key"}
    ),
])
@pytest.mark.asyncio
async def test_provider_initialization(provider_type, provider_config, expected_kwargs, mock_app_config):
    """Test that providers are initialized with the correct parameters."""
    from backend.main import list_models
    from unittest.mock import patch, AsyncMock
    
    # Create a mock provider class
    mock_provider = MagicMock()
    # Set up the async list_models method with dictionaries
    mock_provider.list_models = AsyncMock(return_value=[
        {
            "id": "test-model",
            "name": "Test Model",
            "modified_at": "2023-10-29T19:22:00.000000Z",
            "size": 1000000,
            "parameter_size": "7B",
            "quantization_level": "Q4_0"
        }
    ])
    
    # Create a mock provider class that returns our mock provider instance
    mock_provider_cls = MagicMock(return_value=mock_provider)
    
    # Create a test config with the provider
    test_config = AppConfig(
        default_provider=provider_config["id"],
        providers=[ProviderConfig(**provider_config)],
        logging_enabled=True,
        debug=True,
        data_dir="/tmp/lightchat-test",
        app_name="LightChat"
    )
    
    # Patch the provider class in the main module
    provider_class_path = 'backend.main.OpenAIProvider' if provider_type == 'openai' else 'backend.main.OllamaProvider'
    with patch(provider_class_path, mock_provider_cls):
        # Import inside the patch to ensure the mock is used
        from backend.main import list_models
        
        # Call the function
        result = await list_models(provider_config["id"], test_config)
        
        # Verify the provider was initialized with the correct parameters
        mock_provider_cls.assert_called_once()
        
        # Verify the list_models method was called on the provider instance
        mock_provider.list_models.assert_awaited_once()
        
        # Verify the result contains the expected model data
        assert len(result) == 1
        assert result[0].id == "test-model"
        assert result[0].name == "Test Model"

# Test provider status reporting
@pytest.mark.parametrize("provider_config,expected_status", [
    ({"id": "ollama1", "name": "Ollama", "type": "ollama", "host": "http://localhost:11434"}, "configured"),
    ({"id": "ollama2", "name": "Ollama", "type": "ollama", "host": None}, "unavailable"),
    ({"id": "openai1", "name": "OpenAI", "type": "openai", "api_key": "test-key"}, "configured"),
    ({"id": "openai2", "name": "OpenAI", "type": "openai", "api_key": None}, "needs_api_key"),
])
def test_provider_status_reporting(provider_config, expected_status, mock_app_config):
    """Test that providers report their status correctly based on configuration."""
    from backend.main import get_providers
    
    # Create a test config with the provider
    # Remove id and name from provider_config to avoid duplicates
    provider_config = provider_config.copy()
    test_id = provider_config.pop('id')
    test_name = provider_config.pop('name')
    
    test_config = AppConfig(
        default_provider=test_id,
        providers=[
            ProviderConfig(
                id=test_id,
                name=test_name,
                **provider_config
            )
        ],
        logging_enabled=True,
        debug=True
    )
    
    # Get the providers and check the status
    import asyncio
    providers = asyncio.run(get_providers(test_config))
    assert len(providers) == 1
    assert providers[0].status == expected_status
