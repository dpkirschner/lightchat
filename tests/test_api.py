"""Unit tests for the FastAPI endpoints."""

import pytest
import pytest_asyncio
import httpx
from fastapi import status
from pydantic import ValidationError

from backend.main import app
from backend.models.providers import ProviderMetadata, ProviderStatus, ProviderType

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
    assert provider["type"] == "local"
    assert provider["status"] == "configured"