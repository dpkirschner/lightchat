"""Tests for the chat engine module."""
import pytest
from unittest.mock import AsyncMock, patch

from backend.chat_engine import stream_chat_response, ProviderNotFoundError
from backend.providers.ollama import OllamaProvider
from backend.models.chat import ChatRequest
from backend.models.schemas import SSEEvent

# Mock message types for testing
class MessageRole:
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ChatMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content
        
    def to_dict(self):
        return {"role": self.role, "content": self.content}


@pytest.mark.asyncio
async def test_stream_chat_response_with_ollama(caplog):
    """Test streaming chat response with Ollama provider."""
    
    async def mock_ollama_stream_gen(*args, **kwargs):
        yield {"token": "Hello"}
        yield {"token": " there"}
        yield {"token": "!"}

    with patch('backend.chat_engine.OllamaProvider', spec=OllamaProvider) as mock_provider_class:
        # Set up the mock provider instance
        mock_provider_instance = mock_provider_class.return_value
        mock_provider_instance.chat_stream.return_value = mock_ollama_stream_gen()
        mock_provider_instance.provider_id = "ollama"
        mock_provider_instance.display_name = "Ollama"
        
        # Create a test request
        test_request = ChatRequest(
            prompt="Test prompt",
            provider_id="ollama_default",  # Match the expected provider_id
            model_id="llama2",
            settings={"temperature": 0.7}
        )
        
        # Call the function under test
        responses = []
        async for chunk in stream_chat_response(
            prompt=test_request.prompt,
            provider_id=test_request.provider_id,
            model_id=test_request.model_id,
            settings=test_request.settings
        ):
            responses.append(chunk)
            
        # Verify the results
        assert len(responses) == 3
        assert responses[0]["token"] == "Hello"
        assert responses[1]["token"] == " there"
        assert responses[2]["token"] == "!"
        
        # Verify the provider was created with the correct parameters
        mock_provider_class.assert_called_once_with(
            provider_id="ollama_default",
            display_name="Ollama",
            ollama_base_url="http://localhost:11434"
        )
        
        # Verify chat_stream was called with the correct arguments
        mock_provider_instance.chat_stream.assert_called_once_with(
            "Test prompt", 
            "llama2", 
            {"temperature": 0.7}
        )
        
        # Verify no errors were logged
        assert not any(record.levelno == pytest.logging.ERROR for record in caplog.records)


@pytest.mark.asyncio
async def test_stream_chat_response_with_unknown_provider(caplog):
    """Test streaming with an unknown provider ID yields an error and logs it."""
    chunks = []
    async for chunk in stream_chat_response("Hello!", "unknown_provider", "model123"):
        chunks.append(chunk)
    
    assert len(chunks) == 1
    assert "error" in chunks[0]
    assert "Provider 'unknown_provider' not recognized." in chunks[0]["error"]
    assert "Provider 'unknown_provider' not recognized." in caplog.text


@pytest.mark.asyncio
async def test_stream_chat_response_with_provider_generic_error(caplog):
    """Test error handling when the provider's chat_stream raises a generic Exception."""
    
    async def error_stream_gen(*args, **kwargs):
        raise Exception("Provider internal error")
        yield 

    with patch('backend.chat_engine.OllamaProvider', spec=OllamaProvider) as mock_provider_class:
        mock_provider_instance = mock_provider_class.return_value
        mock_provider_instance.chat_stream.return_value = error_stream_gen()
        
        chunks = []
        async for chunk in stream_chat_response("Hello!", "ollama_default", "llama3:latest"):
            chunks.append(chunk)
        
        assert len(chunks) == 1
        assert "error" in chunks[0]
        assert chunks[0]["error"] == "Error streaming from provider 'ollama_default': Provider internal error"
        # Check the log from chat_engine.py's except block
        assert "Error streaming from provider 'ollama_default': Provider internal error" in caplog.text
        mock_provider_instance.chat_stream.assert_called_once_with("Hello!", "llama3:latest", None)


@pytest.mark.asyncio
async def test_stream_chat_response_with_provider_not_implemented_error(caplog):
    """Test handling of NotImplementedError from provider's chat_stream."""

    async def not_implemented_stream_gen(*args, **kwargs):
        raise NotImplementedError("Feature X not ready")
        yield 

    with patch('backend.chat_engine.OllamaProvider', spec=OllamaProvider) as mock_provider_class:
        mock_provider_instance = mock_provider_class.return_value
        mock_provider_instance.chat_stream.return_value = not_implemented_stream_gen()
        
        chunks = []
        async for chunk in stream_chat_response("Hello!", "ollama_default", "llama3:latest"):
            chunks.append(chunk)
        
        assert len(chunks) == 1
        assert "error" in chunks[0]
        assert chunks[0]["error"] == "Chat stream not implemented for provider 'ollama_default'. Details: Feature X not ready"
        # Check the log from chat_engine.py's except block for NotImplementedError
        assert "NotImplementedError from provider 'ollama_default': Feature X not ready" in caplog.text
        mock_provider_instance.chat_stream.assert_called_once_with("Hello!", "llama3:latest", None)


@pytest.mark.asyncio
async def test_stream_chat_response_with_settings(caplog):
    """Test streaming with custom settings are passed to the provider."""

    async def mock_ollama_stream_settings_gen(*args, **kwargs):
        yield {"token": "Response with settings"}

    with patch('backend.chat_engine.OllamaProvider', spec=OllamaProvider) as mock_provider_class:
        mock_provider_instance = mock_provider_class.return_value
        mock_provider_instance.chat_stream.return_value = mock_ollama_stream_settings_gen()
        
        settings = {"temperature": 0.7, "max_tokens": 100}
        chunks = []
        async for chunk in stream_chat_response("Test", "ollama_default", "llama3:latest", settings):
            chunks.append(chunk)
        
        assert chunks == [{"token": "Response with settings"}]
        mock_provider_instance.chat_stream.assert_called_once_with("Test", "llama3:latest", settings)
        assert not any(record.levelno == pytest.logging.ERROR for record in caplog.records)