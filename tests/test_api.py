"""API tests for the FastAPI endpoints, using asynchronous client."""

import json
import pytest
import pytest_asyncio
import httpx
from fastapi import status
from pydantic import ValidationError
from typing import get_args, List, Dict, Any, AsyncGenerator
import asyncio


from unittest.mock import patch, MagicMock # AsyncMock might not be needed if MagicMock's return is an async gen

from backend.main import app
from backend.models import ProviderMetadata, ProviderStatus, SSEEvent  # Import from the main models package

# Asynchronous client fixture for all async tests in this file
@pytest_asyncio.fixture
async def async_api_client(monkeypatch): 
    """Create an async test client for the FastAPI app, patching sse-starlette event."""

    # sse-starlette.sse.AppStatus.should_exit_event is created at import time.
    # We need to replace it with an event bound to the current test's event loop.
    try:
        from sse_starlette.sse import AppStatus # Try to import AppStatus
        
        # Create a new asyncio.Event that will be bound to the current test's event loop
        new_event_for_current_loop = asyncio.Event()
        
        # Use monkeypatch to replace the class-level attribute for the duration of the fixture's scope
        monkeypatch.setattr(AppStatus, 'should_exit_event', new_event_for_current_loop)
        
    except ImportError:
        logger = logging.getLogger(__name__) # Or use pytest's logging
        logger.warning("Could not import sse_starlette.sse.AppStatus to monkeypatch. "
                         "RuntimeError for different event loop might persist.")

    transport = httpx.ASGITransport(app=app) # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

@pytest.mark.asyncio
async def test_health_check(async_api_client: httpx.AsyncClient):
    """Test the health check endpoint."""
    response = await async_api_client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "ok"}

@pytest.mark.asyncio
async def test_root_endpoint(async_api_client: httpx.AsyncClient):
    """Test the root endpoint."""
    response = await async_api_client.get("/")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"message": "LightChat Backend Active"}


@pytest.mark.asyncio
async def test_get_providers_success(async_api_client: httpx.AsyncClient):
    """Test the /providers endpoint returns a valid response."""
    response = await async_api_client.get("/providers")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0 
    provider = data[0]
    # Basic structure validation
    ProviderMetadata(**provider) # This will raise ValidationError if structure is wrong
    assert provider["id"] == "ollama_default" # Example check
    assert provider["status"] in get_args(ProviderStatus)


SAMPLE_MODELS_RESPONSE = {
    "models": [{
        "name": "llama3:latest", "model": "llama3:latest",
        "modified_at": "2023-10-29T19:22:00.000000Z", "size": 4117063800, "digest": "sha256:...",
        "details": {"parent_model": "", "format": "gguf", "family": "llama", "families": ["llama"], "parameter_size": "7B", "quantization_level": "Q4_0"}
    }]
}

@pytest.mark.asyncio
async def test_get_models_success(async_api_client: httpx.AsyncClient, httpx_mock):
    """Test successful retrieval of models from a provider."""
    httpx_mock.add_response(
        url="http://localhost:11434/api/tags", # This URL is called by OllamaProvider
        json=SAMPLE_MODELS_RESPONSE,
        status_code=200
    )
    response = await async_api_client.get("/models/ollama_default")
    assert response.status_code == status.HTTP_200_OK
    models = response.json()
    assert isinstance(models, list)
    assert len(models) == 1
    model = models[0]
    assert "id" in model
    assert model["id"] == "llama3:latest"

@pytest.mark.asyncio
async def test_get_models_provider_not_found(async_api_client: httpx.AsyncClient):
    """Test that a 404 is returned for unknown provider IDs."""
    response = await async_api_client.get("/models/nonexistent_provider")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["detail"] == "Provider 'nonexistent_provider' not found"

@pytest.mark.asyncio
async def test_get_models_ollama_unavailable(async_api_client: httpx.AsyncClient, httpx_mock, caplog):
    """Test handling of Ollama service being unavailable when listing models."""
    httpx_mock.add_exception(httpx.RequestError("Connection error"))
    response = await async_api_client.get("/models/ollama_default")
    assert response.status_code == status.HTTP_200_OK # Endpoint itself handles error gracefully
    assert response.json() == [] # Expect empty list as per OllamaProvider logic
    assert "Failed to connect to Ollama API" in caplog.text

@pytest.mark.asyncio
async def test_providers_response_structure(): # No client needed for this specific test
    """Test that the ProviderMetadata model validates correctly (schema check)."""
    ProviderMetadata(id="test_id", name="Test Provider", type="local", status="configured")
    with pytest.raises(ValidationError):
        ProviderMetadata(id="test_id", name="Test Provider", type="invalid_type", status="configured")
    with pytest.raises(ValidationError):
        ProviderMetadata(id="test_id", name="Test Provider", type="local", status="invalid_status")

@pytest.mark.asyncio
async def test_providers_content(async_api_client: httpx.AsyncClient):
    """Test the content of the /providers response matches expectations."""
    response = await async_api_client.get("/providers")
    data = response.json()
    provider = data[0]
    assert provider["id"] == "ollama_default"
    assert provider["name"] == "Ollama"
    assert provider["status"] in get_args(ProviderStatus)

async def parse_sse_events_from_aiter_lines(aiter_lines: AsyncGenerator[str, None]) -> List[Dict[str, Any]]:
    """Helper to parse SSE events from an async line iterator."""
    events = []
    current_event_name = "message"
    current_data_parts = []
    async for line in aiter_lines:
        line = line.strip()
        if not line:
            if current_data_parts:
                full_data_str = "".join(current_data_parts)
                try:
                    events.append({"name": current_event_name, "data": json.loads(full_data_str)})
                except json.JSONDecodeError:
                    events.append({"name": current_event_name, "data_raw": full_data_str, "parse_error": True})
                current_data_parts = []
                current_event_name = "message"
            continue
        if line.startswith("event:"):
            current_event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current_data_parts.append(line[len("data:"):].strip())
            
    if current_data_parts: # Process any final event
        full_data_str = "".join(current_data_parts)
        try:
            events.append({"name": current_event_name, "data": json.loads(full_data_str)})
        except json.JSONDecodeError:
            events.append({"name": current_event_name, "data_raw": full_data_str, "parse_error": True})
    return events

@pytest.mark.asyncio
async def test_chat_endpoint_success(async_api_client: httpx.AsyncClient):
    """Test successful chat streaming (async)."""
    mock_engine_events = [{"token": "Hello"}, {"token": " there"}, {"token": "!"}]
    expected_sse_payloads = [
        SSEEvent(token="Hello").model_dump(exclude_none=False),
        SSEEvent(token=" there").model_dump(exclude_none=False),
        SSEEvent(token="!").model_dump(exclude_none=False)
    ]
    
    async def mock_async_gen_func(*args, **kwargs):
        for event in mock_engine_events:
            yield event
    
    mock_generator = mock_async_gen_func()

    with patch('backend.main.stream_chat_response', return_value=mock_generator) as patched_stream_fn:
        response = await async_api_client.post(
            "/chat",
            json={"prompt": "Hello!", "provider_id": "ollama_default", "model_id": "llama3:latest"},
            headers={"Accept": "text/event-stream"}
        )
        
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        
        parsed_events = await parse_sse_events_from_aiter_lines(response.aiter_lines())
        
        assert len(parsed_events) == len(expected_sse_payloads)
        for sse_event, expected_payload in zip(parsed_events, expected_sse_payloads):
            assert sse_event.get("name", "message") == "message"
            assert not sse_event.get("parse_error")
            assert sse_event["data"] == expected_payload
        
        patched_stream_fn.assert_called_once_with(
            prompt="Hello!",
            provider_id="ollama_default",
            model_id="llama3:latest",
            settings=None
        )

@pytest.mark.asyncio
async def test_chat_endpoint_error(async_api_client: httpx.AsyncClient):
    """Test error handling in chat streaming (async)."""
    mock_engine_error_event = {"error": "Provider not available"}
    expected_sse_payload = SSEEvent(error="Provider not available").model_dump(exclude_none=False)
    
    async def mock_error_gen_func(*args, **kwargs):
        yield mock_engine_error_event
            
    mock_error_generator = mock_error_gen_func()
    
    with patch('backend.main.stream_chat_response', return_value=mock_error_generator) as patched_stream_fn:
        response = await async_api_client.post(
            "/chat",
            json={"prompt": "Hello!", "provider_id": "invalid_provider", "model_id": "llama3:latest"},
            headers={"Accept": "text/event-stream"}
        )
        
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        
        parsed_events = await parse_sse_events_from_aiter_lines(response.aiter_lines())
        
        assert len(parsed_events) == 1
        sse_event = parsed_events[0]
        # For error responses, the event type should be 'error'
        assert sse_event.get("name") == "error"
        assert not sse_event.get("parse_error")
        assert sse_event["data"] == expected_sse_payload
        
        patched_stream_fn.assert_called_once_with(
            prompt="Hello!",
            provider_id="invalid_provider",
            model_id="llama3:latest",
            settings=None
        )

@pytest.mark.asyncio
async def test_chat_endpoint_validation_error(async_api_client: httpx.AsyncClient):
    """Test request validation for the /chat endpoint (async)."""
    response = await async_api_client.post(
        "/chat",
        json={"provider_id": "ollama_default"}, # Missing 'prompt'
        headers={"Accept": "text/event-stream"}
    )
    assert response.status_code == 422
    error_data = response.json()
    assert "detail" in error_data
    prompt_error_found = False
    for detail in error_data["detail"]:
        if "prompt" in detail.get("loc", []) and detail.get("type") == "missing":
            prompt_error_found = True
            break
    assert prompt_error_found, "Validation error for missing 'prompt' not found."