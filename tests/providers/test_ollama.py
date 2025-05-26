"""Tests for the Ollama provider implementation."""
import json
from unittest.mock import patch, AsyncMock
import pytest
import httpx
import pytest_asyncio # Usually not needed to be imported directly if using @pytest_asyncio.fixture

from backend.providers.ollama import OllamaProvider
from backend.models.providers import ModelInfo # This was in your original, keeping it

# Sample response from Ollama's /api/tags endpoint
SAMPLE_MODELS_RESPONSE = {
    "models": [
        {
            "name": "llama3:latest",
            "model": "llama3:latest",
            "modified_at": "2023-10-29T19:22:00.000000Z",
            "size": 4117063800,
            "digest": "sha256:...",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "7B",
                "quantization_level": "Q4_0"
            }
        },
        {
            "name": "mistral:latest",
            "model": "mistral:latest",
            "modified_at": "2023-10-30T10:15:00.000000Z",
            "size": 4117063800,
            "digest": "sha256:...",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "mistral",
                "families": ["mistral"],
                "parameter_size": "7B",
                "quantization_level": "Q5_K_M"
            }
        }
    ]
}

@pytest.fixture # No need for @pytest_asyncio.fixture if it's not an async generator
def ollama_provider():
    """Fixture to create an OllamaProvider instance for testing."""
    return OllamaProvider(
        provider_id="test_ollama",
        display_name="Test Ollama",
        ollama_base_url="http://test-ollama:11434"
    )

@pytest.mark.asyncio
async def test_ollama_provider_initialization():
    """Test that OllamaProvider is initialized correctly."""
    provider = OllamaProvider(
        provider_id="test_id",
        display_name="Test Provider",
        ollama_base_url="http://custom:1234"
    )
    
    assert provider.get_id() == "test_id"
    assert provider.get_name() == "Test Provider"
    assert provider.ollama_base_url == "http://custom:1234"

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url_input, expected_api_url",
    [
        ("http://custom-url:1234", "http://custom-url:1234/api/tags"),
        ("http://custom-url:1234/", "http://custom-url:1234/api/tags"), # With trailing slash
    ]
)
async def test_ollama_base_url_normalization(base_url_input, expected_api_url, httpx_mock):
    """Test that the base URL is correctly normalized for API calls."""
    provider = OllamaProvider("test_id", "Test Name", ollama_base_url=base_url_input)
    
    httpx_mock.add_response(url=expected_api_url, json={"models": []}, status_code=200)
    
    await provider.list_models()
    
    assert len(httpx_mock.get_requests()) == 1
    assert str(httpx_mock.get_requests()[0].url) == expected_api_url

@pytest.mark.asyncio
async def test_list_models_success(ollama_provider, httpx_mock):
    """Test successful model listing from Ollama API."""
    httpx_mock.add_response(
        url="http://test-ollama:11434/api/tags",
        json=SAMPLE_MODELS_RESPONSE,
        status_code=200
    )
    
    models = await ollama_provider.list_models()
    
    assert len(models) == 2
    assert models[0]["id"] == "llama3:latest"
    assert models[0]["name"] == "llama3:latest"
    assert models[0]["modified_at"] == "2023-10-29T19:22:00.000000Z"
    assert models[0]["size"] == 4117063800
    assert models[0]["parameter_size"] == "7B"
    assert models[0]["quantization_level"] == "Q4_0"
    
    assert models[1]["id"] == "mistral:latest"
    assert models[1]["parameter_size"] == "7B"
    assert models[1]["quantization_level"] == "Q5_K_M"

@pytest.mark.asyncio
async def test_list_models_success_empty(ollama_provider, httpx_mock):
    """Test successful model listing when Ollama returns an empty model list."""
    httpx_mock.add_response(
        url="http://test-ollama:11434/api/tags",
        json={"models": []},
        status_code=200
    )
    
    models = await ollama_provider.list_models()
    
    assert models == []

@pytest.mark.asyncio
async def test_list_models_http_request_error(ollama_provider, httpx_mock, caplog):
    """Test handling of HTTP RequestError (e.g. connection) when listing models."""
    httpx_mock.add_exception(
        httpx.RequestError("Connection error")
    )
    
    models = await ollama_provider.list_models()
    
    assert models == []
    assert "Failed to connect to Ollama API" in caplog.text

@pytest.mark.asyncio
async def test_list_models_http_status_error(ollama_provider, httpx_mock, caplog):
    """Test handling of HTTP status errors (e.g., 404, 500) when listing models."""
    httpx_mock.add_response(
        url="http://test-ollama:11434/api/tags",
        status_code=500,
        json={"error": "server-side issue"} # Ollama might not return JSON on 500, but good to test
    )
    
    models = await ollama_provider.list_models()
    
    assert models == []
    assert "Error processing Ollama API response" in caplog.text
    # httpx.HTTPStatusError includes the status code in its string representation
    assert "500 Internal Server Error" in caplog.text 

@pytest.mark.asyncio
async def test_list_models_malformed_json_response(ollama_provider, httpx_mock, caplog):
    """Test handling of malformed JSON response from Ollama API."""
    httpx_mock.add_response(
        url="http://test-ollama:11434/api/tags",
        content=b"This is not valid JSON",
        status_code=200
    )
    
    models = await ollama_provider.list_models()
    
    assert models == []
    assert "Unexpected error fetching models from Ollama" in caplog.text
    # Check for part of a potential JSONDecodeError message if possible and desirable
    # For example: assert "json.decoder.JSONDecodeError" in caplog.text or similar

@pytest.mark.asyncio
async def test_list_models_invalid_response_structure(ollama_provider, httpx_mock, caplog):
    """Test handling of invalid response structure (e.g., missing 'models' key) from Ollama API."""
    httpx_mock.add_response(
        url="http://test-ollama:11434/api/tags",
        json={"invalid_key": "no models here"}, # Missing 'models' key
        status_code=200
    )
    
    models = await ollama_provider.list_models()
    
    assert models == []
    # This would likely be caught by the (httpx.HTTPStatusError, KeyError) block
    # if data.get("models", []) fails gracefully or if a KeyError happens earlier.
    # Given data.get("models", []) it will result in an empty list of models from the provider,
    # rather than an error log, if "models" key is missing.
    # If the intention is to log an error if 'models' key is missing, the provider code would need adjustment.
    # For now, based on current provider code, this should just return an empty list without error log.
    # Let's adjust the test based on current provider logic:
    # assert "Error processing Ollama API response" in caplog.text # This might not be logged
    # If the provider's `data.get("models", [])` handles it, no error is logged by that specific block.
    # The generic exception might catch something if other processing fails.
    # Based on the current `list_models` logic, a missing "models" key leads to an empty list,
    # which is a valid scenario, not an error to be logged by the error handlers.
    # Thus, the following assertions are more appropriate for this specific "missing key" test:
    assert models == []
    assert not caplog.records # No error should be logged if "models" key is simply missing

@pytest.mark.asyncio
async def test_list_models_with_missing_optional_fields(ollama_provider, httpx_mock):
    """Test handling of missing optional fields in the Ollama API response."""
    response_data = {
        "models": [
            {
                "name": "simple:latest", # Added name for consistency if it's used for display
                "model": "simple:latest",
                "modified_at": "2023-10-29T19:22:00.000000Z",
                "size": 1000000,
                # "digest": "missing", # digest is not used by the provider's parsing logic
                "details": {} # Empty details, so parameter_size and quantization_level will be None
            }
        ]
    }
    
    httpx_mock.add_response(
        url="http://test-ollama:11434/api/tags",
        json=response_data,
        status_code=200
    )
    
    models = await ollama_provider.list_models()
    
    assert len(models) == 1
    assert models[0]["id"] == "simple:latest"
    assert models[0]["name"] == "simple:latest" # Assuming model['model'] is used for name too
    assert models[0]["parameter_size"] is None
    assert models[0]["quantization_level"] is None

@pytest.mark.asyncio
async def test_chat_stream_not_implemented(ollama_provider):
    """Test that chat_stream raises NotImplementedError."""
    with pytest.raises(NotImplementedError) as exc_info:
        async for _ in ollama_provider.chat_stream("test prompt"):
            pass # pragma: no cover (if you want to exclude this from coverage)
    
    assert "Chat functionality will be implemented in a subsequent story" in str(exc_info.value)