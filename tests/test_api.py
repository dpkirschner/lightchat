"""Unit tests for the FastAPI endpoints."""

import json
import pytest
import pytest_asyncio
import httpx
from fastapi import status
from pydantic import ValidationError
from typing import get_args 

from backend.main import app
from backend.models.providers import ProviderMetadata, ProviderStatus

@pytest_asyncio.fixture
async def client():
    """Create an async test client for the FastAPI app."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client

@pytest.mark.asyncio
async def test_health_check(client: httpx.AsyncClient):
    """Test the health check endpoint."""
    response = await client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "ok"}

@pytest.mark.asyncio
async def test_root_endpoint(client: httpx.AsyncClient):
    """Test the root endpoint."""
    response = await client.get("/")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"message": "LightChat Backend Active"}


@pytest.mark.asyncio
async def test_get_providers_success(client: httpx.AsyncClient):
    """Test the /providers endpoint returns a valid response."""
    response = await client.get("/providers")
    
    # Check status code
    assert response.status_code == status.HTTP_200_OK
    
    # Parse response data
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0  # Should return at least one provider
    
    # Validate the first provider matches our expected structure
    provider = data[0]
    assert "id" in provider
    assert "name" in provider
    assert "type" in provider
    assert "status" in provider
    assert provider["status"] in get_args(ProviderStatus)

# Sample models response for testing
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
        }
    ]
}

@pytest.mark.asyncio
async def test_get_models_success(client: httpx.AsyncClient, httpx_mock):
    """Test successful retrieval of models from a provider."""
    # Mock the Ollama API response
    httpx_mock.add_response(
        url="http://localhost:11434/api/tags",
        json=SAMPLE_MODELS_RESPONSE,
        status_code=200
    )
    
    response = await client.get("/models/ollama_default")
    
    # Check status code
    assert response.status_code == status.HTTP_200_OK
    
    # Parse and validate response
    models = response.json()
    assert isinstance(models, list)
    assert len(models) == 1
    
    # Validate model structure
    model = models[0]
    assert "id" in model
    assert "name" in model
    assert "modified_at" in model
    assert "size" in model
    assert "parameter_size" in model
    assert "quantization_level" in model

@pytest.mark.asyncio
async def test_get_models_provider_not_found(client: httpx.AsyncClient):
    """Test that a 404 is returned for unknown provider IDs."""
    response = await client.get("/models/nonexistent_provider")
    
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["detail"] == "Provider 'nonexistent_provider' not found"

@pytest.mark.asyncio
async def test_get_models_ollama_unavailable(client: httpx.AsyncClient, httpx_mock, caplog):
    """Test handling of Ollama service being unavailable."""
    # Mock a connection error
    httpx_mock.add_exception(httpx.RequestError("Connection error"))
    
    response = await client.get("/models/ollama_default")
    
    # Should still return 200 with empty list
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == []
    
    # Verify error was logged
    assert "Failed to connect to Ollama API" in caplog.text


@pytest.mark.asyncio
async def test_providers_response_structure():
    """Test that the providers endpoint returns data matching the ProviderMetadata model."""
    # This test doesn't make an HTTP request but validates the model directly
    # to ensure our hardcoded response matches the expected schema
    
    # This should not raise an exception if the data is valid
    provider = ProviderMetadata(
        id="test_id",
        name="Test Provider",
        type="local",
        status="configured"
    )
    
    # Test with invalid type
    with pytest.raises(ValidationError):
        ProviderMetadata(
            id="test_id",
            name="Test Provider",
            type="invalid_type",  # Invalid type
            status="configured"
        )
    
    # Test with invalid status
    with pytest.raises(ValidationError):
        ProviderMetadata(
            id="test_id",
            name="Test Provider",
            type="local",
            status="invalid_status"  # Invalid status
        )


@pytest.mark.asyncio
async def test_providers_content(client: httpx.AsyncClient):
    """Test the content of the providers response matches our expectations."""
    response = await client.get("/providers")
    data = response.json()
    
    # Check the first provider matches our hardcoded values
    provider = data[0]
    assert provider["id"] == "ollama_default"
    assert provider["name"] == "Ollama"
    assert provider["status"] in get_args(ProviderStatus)
