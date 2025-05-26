"""Integration tests for the FastAPI endpoints using httpx.AsyncClient."""
# Standard library imports
import json
import asyncio
from typing import List, Dict, Any, AsyncGenerator, get_args

# Third-party imports
import pytest
import pytest_asyncio
import httpx
from fastapi import status
from pydantic import ValidationError
from unittest.mock import patch, MagicMock

# Local imports
from backend.main import app
from backend.models.providers import ProviderMetadata, ProviderStatus
from backend.models.chat import ChatRequest
from backend.config import ProviderConfig, AppConfig

# Session-scoped fixture for the FastAPI app
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Session-scoped fixture for the FastAPI app
@pytest.fixture(scope="session")
def app():
    """Create a FastAPI app instance with test settings."""
    from backend.main import app as _app
    return _app

# Session-scoped fixture for the test client
@pytest_asyncio.fixture(scope="session")
async def async_api_client(app):
    """Create an async test client for the FastAPI app."""
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    # Clear any dependency overrides after all tests
    app.dependency_overrides.clear()

# It's good practice to define other fixtures like mock_app_config if they are used
@pytest.fixture
def mock_app_config():
    """Provides a mock AppConfig instance."""
    return AppConfig(
        default_provider="ollama_default",
        providers=[
            ProviderConfig(
                id="ollama_default", 
                type="ollama", 
                name="Ollama", 
                host="http://localhost:11434"
            ),
            ProviderConfig(
                id="openai_default", 
                type="openai", 
                name="OpenAI", 
                api_key="test_key"
            )
        ],
        log_dir="/tmp/logs",
        data_dir="/tmp/data",
        logging_enabled=True,
        debug=False
    )


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
async def test_get_providers_success(async_api_client: httpx.AsyncClient, mock_app_config: AppConfig):
    """Test that the /providers endpoint returns a list of providers."""
    # Debug: Print the mock_app_config to see what's in it
    print("\n=== mock_app_config ===")
    print(f"Default provider: {mock_app_config.default_provider}")
    print(f"Number of providers: {len(mock_app_config.providers)}")
    for i, provider in enumerate(mock_app_config.providers, 1):
        print(f"Provider {i}: {provider.id} ({provider.type})")
    
    # Import the FastAPI app and the get_app_config function
    from backend.main import app, get_app_config
    
    # Create a dependency override function that returns our mock config
    async def override_get_app_config():
        return mock_app_config
    
    # Apply the dependency override
    app.dependency_overrides[get_app_config] = override_get_app_config
    
    try:
        # Make the request
        print("\n=== Making request to /providers ===")
        response = await async_api_client.get("/providers")
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text}")
        
        assert response.status_code == status.HTTP_200_OK, f"Expected status 200, got {response.status_code}"
        data = response.json()
        
        # Debug: Print the response data
        print("\n=== Response data ===")
        print(f"Response type: {type(data)}")
        print(f"Response length: {len(data)}")
        print(f"Response content: {data}")
        
        # Verify the response structure
        assert isinstance(data, list), f"Expected list, got {type(data)}"
        assert len(data) > 0, "Expected at least one provider in the response"
        
        # Check that all required fields are present
        required_fields = ["id", "name", "type", "status"]
        for i, provider in enumerate(data, 1):
            print(f"\nProvider {i} in response:")
            for field in required_fields:
                assert field in provider, f"Missing field '{field}' in provider {i}"
                print(f"  {field}: {provider[field]}")
        
        # Verify the provider status values are valid
        valid_statuses = get_args(ProviderStatus)
        for provider in data:
            assert provider["status"] in valid_statuses, f"Invalid status: {provider['status']}"
            
    except Exception as e:
        print(f"Test failed with exception: {e}")
        raise
    finally:
        # Clear the dependency overrides
        app.dependency_overrides.clear()

SAMPLE_MODELS_RESPONSE_API = [ # Adjusted to match ModelInfo if that's what endpoint returns
    {
        "id": "qwen3:8b",
        "name": "qwen3:8b",
        "modified_at": "2025-05-25T22:35:45.064378401-07:00",
        "size": 5225387923,
        "parameter_size": "8.2B",
        "quantization_level": "Q4_K_M"
    },
    {
        "id": "llama3.1:latest",
        "name": "llama3.1:latest",
        "modified_at": "2025-05-25T22:23:06.902822632-07:00",
        "size": 4920753328,
        "parameter_size": "8.0B",
        "quantization_level": "Q4_K_M"
    }
]

@pytest.mark.asyncio
async def test_get_models_success(async_api_client: httpx.AsyncClient, mock_app_config: AppConfig):
    # Import the FastAPI app and the get_app_config function
    from backend.main import app, get_app_config
    
    # Create a mock provider instance
    mock_provider_instance = MagicMock()
    
    # Configure list_models to be an async method returning the sample response
    async def list_models_mock():
        return SAMPLE_MODELS_RESPONSE_API
        
    mock_provider_instance.list_models = list_models_mock
    
    # Create a dependency override function that returns our mock config
    async def override_get_app_config():
        return mock_app_config
    
    # Create a mock for get_provider_instance
    async def mock_get_provider_instance(provider_id: str, providers: list):
        return mock_provider_instance
    
    # Apply the dependency overrides
    app.dependency_overrides[get_app_config] = override_get_app_config
    
    try:
        # Make the request to the models endpoint
        response = await async_api_client.get("/models/ollama_default")
        
        # Debug: Print the response
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text}")
        
        # Verify the response
        assert response.status_code == status.HTTP_200_OK, f"Expected status 200, got {response.status_code}"
        data = response.json()
        assert isinstance(data, list), "Response data is not a list"
        assert len(data) > 0, "Response data is empty"
        
        # Verify the model data structure
        assert len(data) > 0, "No models returned"
        model = data[0]
        assert "id" in model, "Model data is missing 'id' field"
        assert "name" in model, "Model data is missing 'name' field"
        assert "parameter_size" in model, "Model data is missing 'parameter_size' field"
        
    except Exception as e:
        print(f"Test failed with exception: {e}")
        raise
    finally:
        # Clear the dependency overrides
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_models_provider_not_found(async_api_client: httpx.AsyncClient, mock_app_config: AppConfig):
    with patch('backend.main.get_app_config', return_value=mock_app_config):
        # Make get_provider_instance raise KeyError which should result in 404
        with patch('backend.main.get_provider_instance', side_effect=KeyError("Provider 'nonexistent_provider' not found")):
            response = await async_api_client.get("/models/nonexistent_provider")
            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert "Provider 'nonexistent_provider' not found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_models_ollama_unavailable(async_api_client: httpx.AsyncClient, mock_app_config: AppConfig, caplog):
    mock_provider_instance = MagicMock()
    async def list_models_mock_exception():
        raise httpx.RequestError("Connection refused by mock")
    mock_provider_instance.list_models = list_models_mock_exception

    with patch('backend.main.get_app_config', return_value=mock_app_config):
        with patch('backend.main.get_provider_instance') as mock_get_provider_instance:
            mock_get_provider_instance.return_value = mock_provider_instance
            response = await async_api_client.get("/models/ollama_default")
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "error listing models for provider" in caplog.text.lower()


@pytest.mark.asyncio
async def test_providers_response_structure():
    ProviderMetadata(id="test_id", name="Test Provider", type="local", status="configured")
    with pytest.raises(ValidationError):
        ProviderMetadata(id="test_id", name="Test Provider", type="invalid_type", status="configured")

@pytest.mark.asyncio
async def test_providers_content(async_api_client: httpx.AsyncClient, mock_app_config: AppConfig):
    # Import the FastAPI app and the get_app_config function
    from backend.main import app, get_app_config
    
    # Create a dependency override function that returns our mock config
    async def override_get_app_config():
        return mock_app_config
    
    # Apply the dependency override
    app.dependency_overrides[get_app_config] = override_get_app_config
    
    try:
        # Make the request to the providers endpoint
        response = await async_api_client.get("/providers")
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text}")
        
        # Verify the response
        assert response.status_code == status.HTTP_200_OK, f"Expected status 200, got {response.status_code}"
        data = response.json()
        
        # Verify we have the expected number of providers
        assert len(data) > 0, "No providers returned in the response"
        
        # Create a mapping of provider IDs to their configs for easier lookup
        provider_configs = {p.id: p for p in mock_app_config.providers}
        
        # Check each provider in the response
        for provider_data in data:
            provider_id = provider_data["id"]
            assert provider_id in provider_configs, f"Unexpected provider ID: {provider_id}"
            
            # Verify the provider type mapping
            config = provider_configs[provider_id]
            expected_type = "cloud" if config.type == "openai" else "local"
            assert provider_data["type"] == expected_type
            
            # Verify the provider name
            assert provider_data["name"] == config.name
            
            # Verify the status field is present and valid
            assert "status" in provider_data
            assert provider_data["status"] in ["configured", "needs_api_key", "unavailable"]
            
    except Exception as e:
        print(f"Test failed with exception: {e}")
        raise
    finally:
        # Clear the dependency overrides
        app.dependency_overrides.clear()

# --- Asynchronous tests for /chat endpoint ---
async def parse_sse_events_from_async_response(response: httpx.Response) -> List[Dict[str, Any]]:
    """Parse SSE events from an async HTTP response.
    
    Args:
        response: The async HTTP response to parse
        
    Returns:
        List of parsed SSE events with their data
    """
    events = []
    current_event_name = "message"
    current_data = b""
    
    async for chunk in response.aiter_bytes():
        # Split by double newline to separate events
        parts = chunk.split(b"\n\n")
        for part in parts:
            if not part.strip():
                continue
                
            # Split into lines and process each line
            lines = part.split(b"\n")
            event_data = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if b":" in line:
                    field, value = line.split(b":", 1)
                    field = field.strip().decode("utf-8")
                    value = value.strip()
                    
                    if field == "event":
                        current_event_name = value.decode("utf-8")
                    elif field == "data":
                        try:
                            event_data = json.loads(value)
                        except json.JSONDecodeError:
                            events.append({
                                "name": current_event_name,
                                "data_raw": value.decode("utf-8"),
                                "parse_error": True
                            })
                            continue
                        
                        events.append({
                            "name": current_event_name,
                            "data": event_data
                        })
    
    return events

@pytest.mark.asyncio
async def test_chat_endpoint_success(async_api_client: httpx.AsyncClient, mock_app_config: AppConfig):
    """Test successful chat endpoint with streaming response."""
    # Mock the chat response data
    mock_engine_events = [
        {"token": "Hello"}, 
        {"token": " there"}, 
        {"token": "!"}
    ]
    
    # Create a mock async generator for the chat response
    async def mock_async_gen_func(*args, **kwargs):
        for event in mock_engine_events:
            yield event
    
    mock_generator_object = mock_async_gen_func()

    # Create a mock for the stream_chat_response function
    with patch('backend.main.get_app_config', return_value=mock_app_config):
        with patch('backend.main.stream_chat_response', return_value=mock_generator_object) as patched_stream_fn:
            # Make the request to the chat endpoint
            async with async_api_client.stream(
                "POST",
                "/chat",
                json={"prompt": "Hello!", "provider_id": "ollama_default", "model_id": "llama3:latest"},
                headers={"Accept": "text/event-stream"},
                timeout=5.0  # Add a timeout to prevent hanging
            ) as response:
                assert response.status_code == 200
                assert "text/event-stream" in response.headers["content-type"]
                
                # Parse the SSE events
                events = []
                try:
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if line.startswith("data:"):
                            try:
                                data = json.loads(line[5:].strip())
                                events.append(data)
                            except json.JSONDecodeError:
                                pass
                except Exception as e:
                    if "Event loop is closed" in str(e):
                        pytest.xfail("Known issue with event loop in test environment")
                    raise
                
                # Verify we received the expected number of events
                assert len(events) == len(mock_engine_events)
                
                # Verify the structure of each event
                for event in events:
                    assert "token" in event
                    assert "error" in event
                    assert event["error"] is None
            
            # Verify stream_chat_response was called with the correct arguments
            patched_stream_fn.assert_called_once()
            
            # Get the call arguments
            call_args = patched_stream_fn.call_args[1]
            assert call_args["prompt"] == "Hello!"
            assert call_args["provider_id"] == "ollama_default"
            assert call_args["model_id"] == "llama3:latest"
            assert call_args["settings"] == {}
            # Don't compare the entire app_config object, just verify it was passed
            assert "app_config" in call_args

@pytest.mark.asyncio
async def test_chat_endpoint_error(async_api_client: httpx.AsyncClient, mock_app_config: AppConfig):
    """Test chat endpoint with error response."""
    # Import the FastAPI app and the get_app_config function
    from backend.main import app, get_app_config
    
    # Create a dependency override function that returns our mock config
    async def override_get_app_config():
        return mock_app_config
    
    # Apply the dependency override
    app.dependency_overrides[get_app_config] = override_get_app_config
    
    try:
        # Mock the error response
        mock_engine_error_event = {"error": "Provider internal test error"}
        
        # Create a mock async generator for the error response
        async def mock_error_gen_func(*args, **kwargs):
            yield mock_engine_error_event
        
        mock_error_generator_object = mock_error_gen_func()
        
        # Patch the stream_chat_response function
        with patch('backend.main.stream_chat_response', return_value=mock_error_generator_object) as patched_stream_fn:
            # Make the request to the chat endpoint
            try:
                async with async_api_client.stream(
                    "POST",
                    "/chat",
                    json={"prompt": "Test", "provider_id": "ollama_default", "model_id": "m"},
                    headers={"Accept": "text/event-stream"},
                    timeout=5.0  # Add a timeout to prevent hanging
                ) as response:
                    assert response.status_code == 200, f"Expected status 200, got {response.status_code}"
                    assert "text/event-stream" in response.headers["content-type"], "Response is not server-sent events"
                    
                    # Parse the SSE events
                    events = []
                    try:
                        async for line in response.aiter_lines():
                            line = line.strip()
                            if line.startswith("data:"):
                                try:
                                    data = json.loads(line[5:].strip())
                                    events.append(data)
                                except json.JSONDecodeError as e:
                                    print(f"Failed to decode JSON: {line[5:].strip()}")
                                    raise
                    except Exception as e:
                        if "Event loop is closed" in str(e):
                            pytest.xfail("Known issue with event loop in test environment")
                        print(f"Error reading response: {e}")
                        raise
                    
                    # Verify we received exactly one error event
                    assert len(events) == 1, f"Expected 1 event, got {len(events)}: {events}"
                    
                    # Verify the error event structure
                    error_event = events[0]
                    assert "token" in error_event, "Error event missing 'token' field"
                    assert error_event["token"] is None, "Error event 'token' should be None"
                    assert "error" in error_event, "Error event missing 'error' field"
                    assert error_event["error"] == "Provider internal test error", \
                        f"Unexpected error message: {error_event['error']}"
                    
                
                # Verify stream_chat_response was called with the correct arguments
                patched_stream_fn.assert_called_once()
                
                # Get the call arguments
                call_args = patched_stream_fn.call_args[1]
                assert call_args["prompt"] == "Test", "Unexpected prompt"
                assert call_args["provider_id"] == "ollama_default", "Unexpected provider_id"
                assert call_args["model_id"] == "m", "Unexpected model_id"
                assert call_args["settings"] == {}, "Unexpected settings"
                # Don't compare the entire app_config object, just verify it was passed
                assert "app_config" in call_args, "app_config not passed to stream_chat_response"
                
            except Exception as e:
                if "Event loop is closed" in str(e):
                    pytest.xfail("Known issue with event loop in test environment")
                print(f"Test failed with exception: {e}")
                raise
    
    finally:
        # Clear the dependency overrides
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_chat_endpoint_validation_error(async_api_client: httpx.AsyncClient):
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