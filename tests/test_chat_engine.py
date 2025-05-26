"""Tests for the chat engine module."""
import pytest
from unittest.mock import AsyncMock, patch

from backend.chat_engine import stream_chat_response
from backend.providers.ollama import OllamaProvider


@pytest.mark.asyncio
async def test_stream_chat_response_with_ollama(caplog):
    """Test streaming chat response with Ollama provider."""
    
    async def mock_ollama_stream_gen(*args, **kwargs): # Renamed for clarity
        yield {"token": "Hello"}
        yield {"token": " there"}
        yield {"token": "!"}

    with patch('backend.chat_engine.OllamaProvider', spec=OllamaProvider) as mock_provider_class:
        mock_provider_instance = mock_provider_class.return_value
        # Set return_value to an instance of the async generator
        mock_provider_instance.chat_stream.return_value = mock_ollama_stream_gen()
        
        chunks = []
        async for chunk in stream_chat_response("Hello!", "ollama_default", "llama3:latest"):
            chunks.append(chunk)
        
        assert chunks == [{"token": "Hello"}, {"token": " there"}, {"token": "!"}]
        
        mock_provider_class.assert_called_once_with(
            provider_id="ollama_default",
            display_name="Ollama",
            ollama_base_url="http://localhost:11434"
        )
        mock_provider_instance.chat_stream.assert_called_once_with("Hello!", "llama3:latest", None)
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