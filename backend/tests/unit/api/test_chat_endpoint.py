"""Integration tests for the FastAPI endpoints using httpx.AsyncClient."""
import json
import pytest
import pytest_asyncio
import httpx
from fastapi import status
import asyncio
from pydantic import ValidationError # For test_providers_response_structure
from typing import get_args, List, Dict, Any, AsyncGenerator # Ensure all are imported

from unittest.mock import patch, MagicMock # Use MagicMock for patching stream_chat_response

from backend.main import app
from backend.models import ProviderMetadata, ProviderStatus, SSEEvent

# Asynchronous client fixture for all tests in this file
@pytest_asyncio.fixture
async def async_api_client(monkeypatch): # Add monkeypatch fixture
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

# --- Tests for general endpoints ---
@pytest.mark.asyncio
async def test_health_check(async_api_client: httpx.AsyncClient):
    response = await async_api_client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "ok"}

@pytest.mark.asyncio
async def test_root_endpoint(async_api_client: httpx.AsyncClient):
    response = await async_api_client.get("/")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"message": "LightChat Backend Active"}

@pytest.mark.asyncio
async def test_get_providers_success(async_api_client: httpx.AsyncClient):
    response = await async_api_client.get("/providers")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    provider = data[0]
    ProviderMetadata(**provider) # Validate structure
    assert provider["id"] == "ollama_default"
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
    httpx_mock.add_response(
        url="http://localhost:11434/api/tags", json=SAMPLE_MODELS_RESPONSE, status_code=200
    )
    response = await async_api_client.get("/models/ollama_default")
    assert response.status_code == status.HTTP_200_OK
    models = response.json()
    assert isinstance(models, list) and len(models) == 1
    assert models[0]["id"] == "llama3:latest"

@pytest.mark.asyncio
async def test_get_models_provider_not_found(async_api_client: httpx.AsyncClient):
    response = await async_api_client.get("/models/nonexistent_provider")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["detail"] == "Provider 'nonexistent_provider' not found"

@pytest.mark.asyncio
async def test_get_models_ollama_unavailable(async_api_client: httpx.AsyncClient, httpx_mock, caplog):
    httpx_mock.add_exception(httpx.RequestError("Connection error"))
    response = await async_api_client.get("/models/ollama_default")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == []
    assert "Failed to connect to Ollama API" in caplog.text

@pytest.mark.asyncio
async def test_providers_response_structure():
    ProviderMetadata(id="test_id", name="Test Provider", type="local", status="configured")
    with pytest.raises(ValidationError):
        ProviderMetadata(id="test_id", name="Test Provider", type="invalid_type", status="configured")

@pytest.mark.asyncio
async def test_providers_content(async_api_client: httpx.AsyncClient):
    response = await async_api_client.get("/providers")
    data = response.json()
    provider = data[0]
    assert provider["id"] == "ollama_default"
    assert provider["name"] == "Ollama"
    assert provider["status"] in get_args(ProviderStatus)


# --- Asynchronous tests for /chat endpoint ---

async def parse_sse_events_from_async_response(response: httpx.Response) -> List[Dict[str, Any]]:
    """Helper to parse SSE events from an httpx.AsyncClient streaming response.
    
    Returns:
        List of parsed events, where each event is a dict with:
        - event: The event type (defaults to "message" if not specified)
        - data: The parsed JSON data (if valid JSON)
        - error: Present if there was an error parsing the data
        - raw_data: The raw data string (if data couldn't be parsed as JSON)
    """
    events = []
    current_event = "message"
    current_data_parts = []
    
    async for line in response.aiter_lines():
        line = line.strip()
        if not line:
            # Empty line marks the end of an event
            if current_data_parts:
                event_data = {"event": current_event}
                full_data_str = "".join(current_data_parts)
                try:
                    event_data["data"] = json.loads(full_data_str)
                except json.JSONDecodeError:
                    event_data["raw_data"] = full_data_str
                    event_data["error"] = "Failed to parse JSON data"
                events.append(event_data)
                
                # Reset for next event
                current_data_parts = []
                current_event = "message"
            continue
            
        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current_data_parts.append(line[len("data:"):].strip())
    
    # Handle any remaining data after the loop
    if current_data_parts:
        event_data = {"event": current_event}
        full_data_str = "".join(current_data_parts)
        try:
            event_data["data"] = json.loads(full_data_str)
        except json.JSONDecodeError:
            event_data["raw_data"] = full_data_str
            event_data["error"] = "Failed to parse JSON data"
        events.append(event_data)
        
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
    
    mock_generator_object = mock_async_gen_func()

    with patch('backend.main.stream_chat_response', MagicMock(return_value=mock_generator_object)) as patched_stream_fn:
        response = await async_api_client.post(
            "/chat",
            json={"prompt": "Hello!", "provider_id": "ollama_default", "model_id": "llama3:latest"},
            headers={"Accept": "text/event-stream"}
        )
        
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        
        parsed_events = await parse_sse_events_from_async_response(response)
        
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
            
    mock_error_generator_object = mock_error_gen_func()
    
    with patch('backend.main.stream_chat_response', MagicMock(return_value=mock_error_generator_object)) as patched_stream_fn:
        response = await async_api_client.post(
            "/chat",
            json={"prompt": "Hello!", "provider_id": "invalid_provider", "model_id": "llama3:latest"},
            headers={"Accept": "text/event-stream"}
        )
        
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        
        parsed_events = await parse_sse_events_from_async_response(response)
        
        assert len(parsed_events) == 1
        sse_event = parsed_events[0]
        # For error responses, the event type should be 'error'
        assert sse_event.get("event") == "error"
        # The data should match our expected payload
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
    assert prompt_error_found, "Validation error for missing 'prompt' field not found."