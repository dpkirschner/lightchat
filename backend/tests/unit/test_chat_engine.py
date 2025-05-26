"""Tests for the chat engine module."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

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

    # Mock the provider config
    mock_provider_config = MagicMock()
    mock_provider_config.type = "ollama"
    mock_provider_config.id = "ollama_default"
    mock_provider_config.name = "Ollama"
    mock_provider_config.host = "http://localhost:11434"
    mock_provider_config.default_model = "llama2"
    
    # Mock the app config
    mock_app_config = MagicMock()
    mock_app_config.get_provider.return_value = mock_provider_config
    
    with patch('backend.chat_engine.OllamaProvider', spec=OllamaProvider) as mock_provider_class:
        # Set up the mock provider instance
        mock_provider_instance = mock_provider_class.return_value
        mock_provider_instance.chat_stream.return_value = mock_ollama_stream_gen()
        mock_provider_instance.provider_id = "ollama_default"
        mock_provider_instance.display_name = "Ollama"
        
        # Create a test request
        test_request = ChatRequest(
            prompt="Test prompt",
            provider_id="ollama_default",
            model_id="llama2",
            settings={"temperature": 0.7}
        )
        
        # Call the function under test with mocked app config
        responses = []
        async for chunk in stream_chat_response(
            prompt=test_request.prompt,
            provider_id=test_request.provider_id,
            model_id=test_request.model_id,
            settings=test_request.settings,
            app_config=mock_app_config
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
        
        # Get the actual call arguments
        call_args = mock_provider_instance.chat_stream.call_args[0]
        # Verify the prompt and model_id
        assert call_args[0] == "Test prompt"
        assert call_args[1] == "llama2"
        # Verify temperature is in settings
        assert call_args[2]["temperature"] == 0.7
        # Verify system_prompt is in settings (added by the implementation)
        assert "system_prompt" in call_args[2]
        
        # Verify no errors were logged
        import logging
        assert not any(record.levelno == logging.ERROR for record in caplog.records)


@pytest.mark.asyncio
async def test_stream_chat_response_with_unknown_provider(caplog):
    """Test streaming with an unknown provider ID yields an error and logs it."""
    # Mock the app config to return None for the provider
    mock_app_config = MagicMock()
    mock_app_config.get_provider.return_value = None
    
    # Test with a non-existent provider
    responses = []
    async for chunk in stream_chat_response(
        prompt="Test prompt",
        provider_id="unknown_provider",
        model_id="some_model",
        app_config=mock_app_config
    ):
        responses.append(chunk)
        
    assert len(responses) == 1
    assert "Provider 'unknown_provider' not found in configuration." in responses[0]["error"]
    assert any("Provider 'unknown_provider' not found in configuration" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_stream_chat_response_with_provider_generic_error(caplog):
    """Test error handling when the provider's chat_stream raises a generic Exception."""
    # Mock the provider config
    mock_provider_config = MagicMock()
    mock_provider_config.type = "ollama"
    mock_provider_config.id = "test_provider"
    mock_provider_config.name = "Test Provider"
    mock_provider_config.host = "http://localhost:11434"
    
    # Mock the app config
    mock_app_config = MagicMock()
    mock_app_config.get_provider.return_value = mock_provider_config
    
    # Create a mock provider that raises an exception
    mock_provider = MagicMock()
    mock_provider.chat_stream.side_effect = Exception("Some internal error")
    mock_provider.provider_id = "test_provider"
    
    with patch('backend.chat_engine.OllamaProvider', return_value=mock_provider):
        responses = []
        async for chunk in stream_chat_response(
            prompt="Test prompt",
            provider_id="test_provider",
            model_id="test_model",
            app_config=mock_app_config
        ):
            responses.append(chunk)
            
        assert len(responses) == 1
        assert "Error in chat stream for provider 'test_provider'" in responses[0]["error"]
        
        # Verify the error was logged
        assert any("Error in chat stream for provider 'test_provider'" in record.message 
                  for record in caplog.records)


@pytest.mark.asyncio
async def test_stream_chat_response_with_provider_not_implemented_error(caplog):
    """Test handling of NotImplementedError from provider's chat_stream."""
    # Mock the provider config
    mock_provider_config = MagicMock()
    mock_provider_config.type = "ollama"
    mock_provider_config.id = "test_provider"
    mock_provider_config.name = "Test Provider"
    mock_provider_config.host = "http://localhost:11434"
    
    # Mock the app config
    mock_app_config = MagicMock()
    mock_app_config.get_provider.return_value = mock_provider_config
    
    # Create a mock provider that raises NotImplementedError
    mock_provider = MagicMock()
    mock_provider.chat_stream.side_effect = NotImplementedError("Feature X not ready")
    mock_provider.provider_id = "test_provider"
    
    with patch('backend.chat_engine.OllamaProvider', return_value=mock_provider):
        responses = []
        async for chunk in stream_chat_response(
            prompt="Test prompt",
            provider_id="test_provider",
            model_id="test_model",
            app_config=mock_app_config
        ):
            responses.append(chunk)
            
        assert len(responses) == 1
        assert "Chat stream not implemented for provider 'test_provider'" in responses[0]["error"]
        
        # Verify the error was logged
        assert any("not implemented for provider 'test_provider'" in record.message 
                  for record in caplog.records)


@pytest.mark.asyncio
async def test_stream_chat_response_with_settings(caplog):
    """Test streaming with custom settings are passed to the provider."""
    # Mock the provider config
    mock_provider_config = MagicMock()
    mock_provider_config.type = "ollama"
    mock_provider_config.id = "test_provider"
    mock_provider_config.name = "Test Provider"
    mock_provider_config.host = "http://localhost:11434"
    
    # Mock the app config
    mock_app_config = MagicMock()
    mock_app_config.get_provider.return_value = mock_provider_config
    
    # Create a mock provider
    mock_provider = MagicMock()
    mock_provider.chat_stream.return_value = AsyncMock()
    mock_provider.chat_stream.return_value.__aiter__.return_value = [{"token": "Response with settings"}]
    mock_provider.provider_id = "test_provider"
    
    with patch('backend.chat_engine.OllamaProvider', return_value=mock_provider):
        test_settings = {"temperature": 0.9, "max_tokens": 100}
        responses = []
        async for chunk in stream_chat_response(
            prompt="Test prompt with settings",
            provider_id="test_provider",
            model_id="test_model",
            settings=test_settings,
            app_config=mock_app_config
        ):
            responses.append(chunk)
            
        # Verify the settings were passed to the provider
        mock_provider.chat_stream.assert_called_once()
        call_args = mock_provider.chat_stream.call_args[0]
        assert call_args[0] == "Test prompt with settings"
        assert call_args[1] == "test_model"
        # Check that our settings are included
        assert call_args[2]["temperature"] == 0.9
        assert call_args[2]["max_tokens"] == 100
        # Check that system_prompt is included (added by the implementation)
        assert "system_prompt" in call_args[2]
