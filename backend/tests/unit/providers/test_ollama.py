"""Tests for the Ollama provider implementation."""
# Standard library imports
import json
import logging

# Third-party imports
import pytest
import httpx
from unittest.mock import patch, AsyncMock

# Local imports
from ....providers.ollama import OllamaProvider
# from backend.models.providers import ModelInfo # Keeping as it was in your original


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

@pytest.fixture
def ollama_provider():
    """Fixture to create an OllamaProvider instance for testing."""
    return OllamaProvider(
        provider_id="test_ollama",
        display_name="Test Ollama",
        ollama_base_url="http://test-ollama:11434"
    )

# --- Tests for OllamaProvider.__init__ and getters ---
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
        ("http://custom-url:1234/", "http://custom-url:1234/api/tags"), 
    ]
)
async def test_ollama_base_url_normalization_for_list_models(base_url_input, expected_api_url, httpx_mock):
    """Test that the base URL is correctly normalized for list_models API calls."""
    provider = OllamaProvider("test_id", "Test Name", ollama_base_url=base_url_input)
    httpx_mock.add_response(url=expected_api_url, json={"models": []}, status_code=200)
    await provider.list_models()
    assert len(httpx_mock.get_requests()) == 1
    assert str(httpx_mock.get_requests()[0].url) == expected_api_url

# --- Tests for list_models ---
@pytest.mark.asyncio
async def test_list_models_success(ollama_provider, httpx_mock):
    """Test successful model listing from Ollama API."""
    httpx_mock.add_response(
        url=f"{ollama_provider.ollama_base_url}/api/tags",
        json=SAMPLE_MODELS_RESPONSE,
        status_code=200
    )
    models = await ollama_provider.list_models()
    assert len(models) == 2
    assert models[0]["id"] == "llama3:latest"
    assert models[0]["parameter_size"] == "7B"
    assert models[1]["id"] == "mistral:latest"

@pytest.mark.asyncio
async def test_list_models_success_empty(ollama_provider, httpx_mock):
    """Test successful model listing when Ollama returns an empty model list."""
    httpx_mock.add_response(
        url=f"{ollama_provider.ollama_base_url}/api/tags",
        json={"models": []},
        status_code=200
    )
    models = await ollama_provider.list_models()
    assert models == []

@pytest.mark.asyncio
async def test_list_models_http_request_error(ollama_provider, httpx_mock, caplog):
    """Test handling of HTTP RequestError when listing models."""
    httpx_mock.add_exception(httpx.RequestError("Connection refused"))
    models = await ollama_provider.list_models()
    assert models == []
    assert "Failed to connect to Ollama API" in caplog.text
    assert "list_models" in caplog.text

@pytest.mark.asyncio
async def test_list_models_http_status_error(ollama_provider, httpx_mock, caplog):
    """Test handling of HTTP status errors when listing models."""
    httpx_mock.add_response(
        url=f"{ollama_provider.ollama_base_url}/api/tags",
        status_code=503, text="Service Unavailable"
    )
    models = await ollama_provider.list_models()
    assert models == []
    assert "Error processing Ollama API response" in caplog.text
    assert "503 Service Unavailable" in caplog.text # Check for status and text from response
    assert "list_models" in caplog.text

@pytest.mark.asyncio
async def test_list_models_malformed_json_response(ollama_provider, httpx_mock, caplog):
    """Test handling of malformed JSON response when listing models."""
    httpx_mock.add_response(
        url=f"{ollama_provider.ollama_base_url}/api/tags",
        content=b"{not_valid_json_at_all",
        status_code=200
    )
    models = await ollama_provider.list_models()
    assert models == []
    assert "Failed to parse JSON response" in caplog.text # Specific error from provider
    assert "list_models" in caplog.text

@pytest.mark.asyncio
async def test_list_models_invalid_response_structure(ollama_provider, httpx_mock, caplog):
    """Test that a missing 'models' key in a valid JSON 200 response is handled gracefully."""
    httpx_mock.add_response(
        url=f"{ollama_provider.ollama_base_url}/api/tags",
        json={"some_other_key": "data"}, # Missing 'models' key
        status_code=200
    )

    with caplog.at_level(logging.ERROR):
        caplog.clear() 
        models = await ollama_provider.list_models()

    assert models == []
    error_messages_to_check = [
        "Error processing Ollama API response", 
        "Unexpected error fetching models",
        "Failed to parse JSON response" 
    ]
    found_error_log = False
    for record in caplog.records:
        if record.levelno >= logging.ERROR: # Check for ERROR or CRITICAL logs
            for msg_part in error_messages_to_check:
                if msg_part in record.message:
                    found_error_log = True
                    break
            if found_error_log:
                break
    assert not found_error_log, f"An unexpected error was logged: {caplog.text}" 

@pytest.mark.asyncio
async def test_list_models_with_missing_optional_fields(ollama_provider, httpx_mock):
    """Test handling of missing optional fields in the Ollama API response for list_models."""
    response_data = {
        "models": [{
            "model": "simple:latest", "name": "simple:latest",
            "modified_at": "2023-10-29T19:22:00Z", "size": 1000, "details": {}
        }]
    }
    httpx_mock.add_response(
        url=f"{ollama_provider.ollama_base_url}/api/tags",
        json=response_data, status_code=200
    )
    models = await ollama_provider.list_models()
    assert len(models) == 1
    assert models[0]["id"] == "simple:latest"
    assert models[0]["parameter_size"] is None
    assert models[0]["quantization_level"] is None

# --- Tests for chat_stream ---
@pytest.mark.asyncio
async def test_chat_stream_success(ollama_provider, httpx_mock):
    """Test successful chat stream with Ollama API."""
    mock_sse_lines = [
        b'data: {"message": {"content": "Hello"}, "done": false}\n\n',
        b'data: {"message": {"content": " there"}, "done": false}\n\n',
        b'data: {"message": {"content": "!"}, "done": true}\n\n',
    ]
    
    httpx_mock.add_response(
        method="POST", url=f"{ollama_provider.ollama_base_url}/api/chat",
        content=b"".join(mock_sse_lines), 
        status_code=200, headers={"Content-Type": "application/x-ndjson"}
    )
    
    tokens = []
    async for chunk in ollama_provider.chat_stream("Hello!", model_id="llama3:latest"):
        if "token" in chunk:
            tokens.append(chunk["token"])
        elif "error" in chunk:
            pytest.fail(f"Stream yielded an error: {chunk['error']}") # Fail test if error is yielded
            
    assert tokens == ["Hello", " there", "!"]
    assert len(httpx_mock.get_requests()) == 1
    request = httpx_mock.get_requests()[0]
    assert request.method == "POST"
    assert json.loads(request.content) == {
        "model": "llama3:latest",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": True
    }

@pytest.mark.asyncio
async def test_chat_stream_missing_model_id(ollama_provider, caplog):
    """Test chat_stream with missing model_id yields an error."""
    chunks = [chunk async for chunk in ollama_provider.chat_stream("test prompt")]
    
    assert len(chunks) == 1
    assert "error" in chunks[0]
    assert chunks[0]["error"] == "model_id is required for Ollama chat requests"
    assert "model_id is required" in caplog.text

@pytest.mark.asyncio
async def test_chat_stream_connection_error(ollama_provider, httpx_mock, caplog):
    """Test chat_stream handles connection errors with retry."""
    # Add two exceptions to test the retry logic
    httpx_mock.add_exception(httpx.ConnectError("Connection failed"))
    httpx_mock.add_exception(httpx.ConnectError("Connection failed"))
    
    chunks = [chunk async for chunk in ollama_provider.chat_stream("test", model_id="llama3:latest")]
    
    assert len(chunks) == 1
    assert "error" in chunks[0]
    assert "Failed to connect to Ollama service after multiple retries" in chunks[0]["error"]
    assert "Failed to connect to Ollama API after 2 attempts" in caplog.text
    assert "Request error (attempt 1/2)" in caplog.text
    # The actual log messages don't include the method name, so we'll just verify the error level messages
    assert any(record.levelname == "ERROR" for record in caplog.records)

@pytest.mark.asyncio
async def test_chat_stream_http_error(ollama_provider, httpx_mock, caplog):
    """Test chat_stream handles HTTP errors."""
    httpx_mock.add_response(
        method="POST", url=f"{ollama_provider.ollama_base_url}/api/chat",
        status_code=404, text="Model not found"
    )
    
    chunks = [chunk async for chunk in ollama_provider.chat_stream("test", model_id="nonexistent_model")]
    
    assert len(chunks) == 1
    assert "error" in chunks[0]
    assert chunks[0]["error"] == "Ollama API error: 404"
    assert "Ollama API returned HTTP error" in caplog.text
    assert "404" in caplog.text # Check for status code in log

@pytest.mark.asyncio
async def test_chat_stream_ollama_returns_error_in_data(ollama_provider, httpx_mock, caplog):
    """Test chat_stream when Ollama returns an error object in the stream."""
    mock_sse_error = [
        b'data: {"error": "model not loaded due to reasons"}\n\n',
    ]
    httpx_mock.add_response(
        method="POST", url=f"{ollama_provider.ollama_base_url}/api/chat",
        content=b"".join(mock_sse_error), status_code=200,
        headers={"Content-Type": "application/x-ndjson"}
    )
    chunks = [chunk async for chunk in ollama_provider.chat_stream("hi", model_id="unloaded_model")]
    assert len(chunks) == 1
    assert "error" in chunks[0]
    assert chunks[0]["error"] == "Ollama API error: model not loaded due to reasons"
    assert "Ollama API error in stream: model not loaded due to reasons" in caplog.text

@pytest.mark.asyncio
async def test_chat_stream_invalid_json_in_stream(ollama_provider, httpx_mock, caplog):
    """Test chat_stream handles invalid JSON in the SSE response."""
    mock_sse_invalid_json = [
        b'data: {"message": {"content": "Valid"}, "done": false}\n\n',
        b'data: {this is not json}\n\n', # This line will cause JSONDecodeError
        b'data: {"message": {"content": "More valid"}, "done": false}\n\n', # This won't be reached
    ]
    httpx_mock.add_response(
        method="POST", url=f"{ollama_provider.ollama_base_url}/api/chat",
        content=b"".join(mock_sse_invalid_json), status_code=200,
        headers={"Content-Type": "application/x-ndjson"}
    )
    
    results = [chunk async for chunk in ollama_provider.chat_stream("test", model_id="llama3:latest")]
    
    assert len(results) == 2 
    assert results[0] == {"token": "Valid"}
    assert "error" in results[1]
    assert results[1]["error"] == "Failed to parse Ollama stream response"
    assert "Failed to parse Ollama JSON response line" in caplog.text
    assert "{this is not json}" in caplog.text 

@pytest.mark.asyncio
async def test_chat_stream_with_settings(ollama_provider, httpx_mock):
    """Test chat_stream with additional settings are correctly passed in payload."""
    mock_sse_response = [b'data: {"message": {"content": "Response"}, "done": true}\n\n']
    httpx_mock.add_response(
        method="POST", url=f"{ollama_provider.ollama_base_url}/api/chat",
        content=b"".join(mock_sse_response), status_code=200,
        headers={"Content-Type": "application/x-ndjson"}
    )
    
    settings = {"temperature": 0.5, "top_p": 0.8, "options": {"seed": 123}}
    
    _ = [chunk async for chunk in ollama_provider.chat_stream("prompt", model_id="llama3:latest", settings=settings)]
    
    assert len(httpx_mock.get_requests()) == 1
    request = httpx_mock.get_requests()[0]
    payload = json.loads(request.content)
    
    assert payload["model"] == "llama3:latest"
    assert payload["messages"] == [{"role": "user", "content": "prompt"}]
    assert payload["stream"] is True
    assert payload["temperature"] == 0.5
    assert payload["top_p"] == 0.8
    assert payload["options"]["seed"] == 123


@pytest.mark.asyncio
async def test_chat_stream_retry_after_connection_error(ollama_provider, httpx_mock, caplog):
    """Test that chat_stream retries after a connection error and succeeds."""
    # First attempt fails with connection error
    httpx_mock.add_exception(httpx.ConnectError("Connection refused"))
    
    # Second attempt succeeds
    mock_sse_response = [b'data: {"message": {"content": "Hello"}, "done": true}\n\n']
    httpx_mock.add_response(
        method="POST", url=f"{ollama_provider.ollama_base_url}/api/chat",
        content=b"".join(mock_sse_response), status_code=200,
        headers={"Content-Type": "application/x-ndjson"}
    )
    
    chunks = [chunk async for chunk in ollama_provider.chat_stream("test", model_id="llama3:latest")]
    
    assert len(chunks) == 1
    assert chunks[0] == {"token": "Hello"}
    assert len(httpx_mock.get_requests()) == 2
    assert "Request error (attempt 1/2): Connection refused" in caplog.text


@pytest.mark.asyncio
async def test_chat_stream_retry_after_retryable_status(ollama_provider, httpx_mock, caplog):
    """Test that chat_stream retries after receiving a retryable status code."""
    # First attempt fails with 503
    httpx_mock.add_response(
        method="POST", url=f"{ollama_provider.ollama_base_url}/api/chat",
        status_code=503, text="Service Unavailable"
    )
    
    # Second attempt succeeds
    mock_sse_response = [b'data: {\"message\": {\"content\": \"Hello\"}, \"done\": true}\n\n']
    httpx_mock.add_response(
        method="POST", url=f"{ollama_provider.ollama_base_url}/api/chat",
        content=b"".join(mock_sse_response), status_code=200,
        headers={"Content-Type": "application/x-ndjson"}
    )
    
    chunks = [chunk async for chunk in ollama_provider.chat_stream("test", model_id="llama3:latest")]
    
    assert len(chunks) == 1
    assert chunks[0] == {"token": "Hello"}
    assert len(httpx_mock.get_requests()) == 2
    assert "Received retryable status 503, will retry..." in caplog.text


@pytest.mark.asyncio
async def test_chat_stream_fails_after_max_retries(ollama_provider, httpx_mock, caplog):
    """Test that chat_stream fails after max retry attempts."""
    # Both attempts fail with connection errors
    httpx_mock.add_exception(httpx.ConnectError("Connection refused"))
    httpx_mock.add_exception(httpx.ConnectError("Connection refused"))
    
    chunks = [chunk async for chunk in ollama_provider.chat_stream("test", model_id="llama3:latest")]
    
    assert len(chunks) == 1
    assert "error" in chunks[0]
    assert "Failed to connect to Ollama service after multiple retries" in chunks[0]["error"]
    assert len(httpx_mock.get_requests()) == 2
    assert "Failed to connect to Ollama API after 2 attempts" in caplog.text


@pytest.mark.asyncio
async def test_chat_stream_non_retryable_error(ollama_provider, httpx_mock, caplog):
    """Test that chat_stream doesn't retry on non-retryable errors."""
    # 400 Bad Request - should not be retried
    httpx_mock.add_response(
        method="POST", url=f"{ollama_provider.ollama_base_url}/api/chat",
        status_code=400, text="Bad Request"
    )
    
    chunks = [chunk async for chunk in ollama_provider.chat_stream("test", model_id="invalid-model")]
    
    assert len(chunks) == 1
    assert "error" in chunks[0]
    assert "Ollama API error: 400" in chunks[0]["error"]
    assert len(httpx_mock.get_requests()) == 1  # No retry for 400


@pytest.mark.asyncio
async def test_chat_stream_success_first_attempt(ollama_provider, httpx_mock, caplog):
    """Test that chat_stream succeeds on first attempt without retry."""
    mock_sse_response = [b'data: {\"message\": {\"content\": \"Hello\"}, \"done\": true}\n\n']
    httpx_mock.add_response(
        method="POST", url=f"{ollama_provider.ollama_base_url}/api/chat",
        content=b"".join(mock_sse_response), status_code=200,
        headers={"Content-Type": "application/x-ndjson"}
    )
    
    chunks = [chunk async for chunk in ollama_provider.chat_stream("test", model_id="llama3:latest")]
    
    assert len(chunks) == 1
    assert chunks[0] == {"token": "Hello"}
    assert len(httpx_mock.get_requests()) == 1  # Only one attempt made
    assert "Retrying Ollama request" not in caplog.text  # No retry message