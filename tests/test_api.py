"""Unit tests for the FastAPI endpoints."""

import pytest
import pytest_asyncio
import httpx
from fastapi import status
from backend.main import app

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