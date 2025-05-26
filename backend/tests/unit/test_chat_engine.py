"""Tests for the chat engine module."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import logging # Import logging for caplog

from backend.chat_engine import stream_chat_response
from backend.providers.ollama import OllamaProvider # For spec
from backend.config import AppConfig, ProviderConfig # For mock_app_config

# Test data for AppConfig mock
SAMPLE_CHAT_ENGINE_PROVIDERS = [
    ProviderConfig(
        id="ollama_default",
        name="Local Ollama",
        type="ollama",
        host="http://localhost:11434",
        default_model="llama3",
        system_prompt="You are a helpful Ollama assistant." # Example system prompt
    ),
    ProviderConfig(
        id="openai_default", # Example, assuming OpenAIProvider might be used
        name="OpenAI Cloud",
        type="openai",
        api_key="test_openai_key",
        default_model="gpt-4",
        system_prompt="You are a helpful OpenAI assistant."
    )
]

@pytest.fixture
def mock_app_config_for_chat_engine():
    """Provides a mock AppConfig specifically for chat_engine tests."""
    return AppConfig(
        default_provider="ollama_default",
        providers=SAMPLE_CHAT_ENGINE_PROVIDERS,
        logging_enabled=True,
        # debug=True # Only include if 'debug' is a valid field in your AppConfig
    )


@pytest.mark.asyncio
async def test_stream_chat_response_with_ollama(caplog, mock_app_config_for_chat_engine: AppConfig):
    """Test streaming chat response with Ollama provider successfully."""
    
    async def mock_ollama_stream_gen_func(*args, **kwargs):
        yield {"token": "Hello"}
        yield {"token": " there"}
        yield {"token": "!"}

    # Patch the OllamaProvider class within chat_engine
    with patch('backend.chat_engine.OllamaProvider', spec=OllamaProvider) as mock_ollama_provider_class:
        mock_ollama_instance = mock_ollama_provider_class.return_value
        # Set chat_stream to a MagicMock that returns an instance of the async generator
        mock_ollama_instance.chat_stream = MagicMock(return_value=mock_ollama_stream_gen_func())
        
        # Patch load_app_config where it's looked up (backend.config)
        with patch('backend.config.load_app_config', return_value=mock_app_config_for_chat_engine) as mock_load_config:
            chunks = []
            # Call with app_config=None to test the internal load_app_config path
            async for chunk in stream_chat_response(
                prompt="Hello!", 
                provider_id="ollama_default", 
                model_id="llama3:latest", 
                app_config=None # This will trigger load_app_config internally
            ):
                chunks.append(chunk)
            
            assert chunks == [{"token": "Hello"}, {"token": " there"}, {"token": "!"}]
            mock_load_config.assert_called_once() # Verify load_app_config was called

            expected_ollama_config = next(p for p in mock_app_config_for_chat_engine.providers if p.id == "ollama_default")
            mock_ollama_provider_class.assert_called_once_with(
                provider_id=expected_ollama_config.id,
                display_name=expected_ollama_config.name,
                ollama_base_url=expected_ollama_config.host
            )
            # Default settings passed to provider.chat_stream is an empty dict if None is given to stream_chat_response
            # and no system_prompt is configured or settings provided already includes it.
            expected_settings_for_provider = {}
            if expected_ollama_config.system_prompt:
                expected_settings_for_provider["system_prompt"] = expected_ollama_config.system_prompt
            mock_ollama_instance.chat_stream.assert_called_once_with("Hello!", "llama3:latest", expected_settings_for_provider)
            
            # Check that no ERROR level logs were emitted by the function under test
            # (logging from the provider itself might still occur if not fully mocked for errors)
            assert not any(record.levelno >= logging.ERROR and record.name.startswith("lightchat.chat_engine") for record in caplog.records)


@pytest.mark.asyncio
async def test_stream_chat_response_with_unknown_provider(caplog, mock_app_config_for_chat_engine: AppConfig):
    """Test streaming with an unknown provider ID yields an error and logs it."""
    with patch('backend.config.load_app_config', return_value=mock_app_config_for_chat_engine):
        chunks = []
        async for chunk in stream_chat_response(
            "Hello!", "unknown_provider", "model123", app_config=mock_app_config_for_chat_engine
        ):
            chunks.append(chunk)
        
        assert len(chunks) == 1
        assert "error" in chunks[0]
        assert chunks[0]["error"] == "Provider 'unknown_provider' not found in configuration."
        assert "Provider 'unknown_provider' not found in configuration." in caplog.text


@pytest.mark.asyncio
async def test_stream_chat_response_with_provider_generic_error(caplog, mock_app_config_for_chat_engine: AppConfig):
    """Test error handling when the provider's chat_stream raises a generic Exception during iteration."""
    
    async def error_stream_gen_func(*args, **kwargs):
        raise Exception("Provider internal error")
        yield # To make it a generator

    with patch('backend.chat_engine.OllamaProvider', spec=OllamaProvider) as mock_ollama_provider_class:
        mock_ollama_instance = mock_ollama_provider_class.return_value
        mock_ollama_instance.chat_stream = MagicMock(return_value=error_stream_gen_func())
        
        with patch('backend.config.load_app_config', return_value=mock_app_config_for_chat_engine):
            chunks = []
            async for chunk in stream_chat_response(
                "Hello!", "ollama_default", "llama3:latest", app_config=mock_app_config_for_chat_engine
            ):
                chunks.append(chunk)
            
            assert len(chunks) == 1
            assert "error" in chunks[0]
            assert chunks[0]["error"] == "Unexpected error: Provider internal error"
            # Check the log from chat_engine.py's *inner* try-except block around the stream iteration
            assert "Unexpected error in chat stream" in caplog.text # Log from inner try-except
            assert "Provider internal error" in caplog.text # The original exception string
            mock_ollama_instance.chat_stream.assert_called_once()


@pytest.mark.asyncio
async def test_stream_chat_response_with_provider_not_implemented_error(caplog, mock_app_config_for_chat_engine: AppConfig):
    """Test handling of NotImplementedError from provider's chat_stream."""

    async def not_implemented_stream_gen_func(*args, **kwargs):
        raise NotImplementedError("Feature X not ready")
        yield 

    with patch('backend.chat_engine.OllamaProvider', spec=OllamaProvider) as mock_ollama_provider_class:
        mock_ollama_instance = mock_ollama_provider_class.return_value
        mock_ollama_instance.chat_stream = MagicMock(return_value=not_implemented_stream_gen_func())

        with patch('backend.config.load_app_config', return_value=mock_app_config_for_chat_engine):
            chunks = []
            async for chunk in stream_chat_response(
                "Hello!", "ollama_default", "llama3:latest", app_config=mock_app_config_for_chat_engine
            ):
                chunks.append(chunk)
            
            assert len(chunks) == 1
            assert "error" in chunks[0]
            # Corrected assertion: Expect the error message from the inner 'except Exception'
            assert chunks[0]["error"] == "Unexpected error: Feature X not ready"
            
            # Check the log from chat_engine.py's inner try-except block
            assert "Unexpected error in chat stream" in caplog.text # Logged by the inner 'except Exception'
            assert "Feature X not ready" in caplog.text # The str(e) part logged
            mock_ollama_instance.chat_stream.assert_called_once()


@pytest.mark.asyncio
async def test_stream_chat_response_with_settings(caplog, mock_app_config_for_chat_engine: AppConfig):
    """Test streaming with custom settings are passed to the provider."""

    async def mock_ollama_stream_settings_gen_func(*args, **kwargs):
        # This function now receives the actual arguments passed to chat_stream
        # print(f"Mocked chat_stream called with: args={args}, kwargs={kwargs}") # For debugging
        # We can access settings from kwargs['settings'] or args[2] if called positionally.
        # The mock below (MagicMock) will capture these, so direct inspection here isn't strictly needed.
        yield {"token": "Response with settings"}

    with patch('backend.chat_engine.OllamaProvider', spec=OllamaProvider) as mock_ollama_provider_class:
        mock_ollama_instance = mock_ollama_provider_class.return_value
        # The MagicMock will wrap the call to the generator function, allowing arg inspection.
        # We assign the function itself to be wrapped by MagicMock to inspect its call args later.
        # Or, more simply, set the return_value to an instance, and assert calls on the attribute.
        mock_ollama_instance.chat_stream = MagicMock(return_value=mock_ollama_stream_settings_gen_func())
        
        settings_to_pass = {"temperature": 0.7, "max_tokens": 100}
        
        ollama_provider_config = next((p for p in mock_app_config_for_chat_engine.providers if p.id == "ollama_default"), None)
        assert ollama_provider_config is not None, "Test setup error: ollama_default provider missing in mock_app_config"

        with patch('backend.config.load_app_config', return_value=mock_app_config_for_chat_engine):
            chunks = []
            async for chunk in stream_chat_response(
                "Test", 
                "ollama_default", 
                "llama3:latest", 
                settings=settings_to_pass, 
                app_config=mock_app_config_for_chat_engine
            ):
                chunks.append(chunk)
            
            assert chunks == [{"token": "Response with settings"}]
            
            expected_provider_settings = settings_to_pass.copy()
            # Check if system_prompt from config should be merged
            if ollama_provider_config.system_prompt and "system_prompt" not in expected_provider_settings:
                 expected_provider_settings["system_prompt"] = ollama_provider_config.system_prompt
            
            mock_ollama_instance.chat_stream.assert_called_once_with("Test", "llama3:latest", expected_provider_settings)
            assert not any(record.levelno >= logging.ERROR for record in caplog.records)

@pytest.mark.asyncio
async def test_stream_chat_response_missing_model_id_uses_default(caplog, mock_app_config_for_chat_engine: AppConfig):
    """Test that default_model is used if model_id is None."""
    async def mock_stream_gen_func(*args, **kwargs):
        yield {"token": "Default model response"}

    with patch('backend.chat_engine.OllamaProvider', spec=OllamaProvider) as mock_ollama_provider_class:
        mock_ollama_instance = mock_ollama_provider_class.return_value
        mock_ollama_instance.chat_stream = MagicMock(return_value=mock_stream_gen_func())
        
        with patch('backend.config.load_app_config', return_value=mock_app_config_for_chat_engine):
            chunks = []
            async for chunk in stream_chat_response(
                "Test prompt", "ollama_default", model_id=None, app_config=mock_app_config_for_chat_engine
            ):
                chunks.append(chunk)

            assert chunks == [{"token": "Default model response"}]
            
            ollama_config = next(p for p in mock_app_config_for_chat_engine.providers if p.id == "ollama_default")
            expected_settings = {}
            if ollama_config.system_prompt:
                expected_settings["system_prompt"] = ollama_config.system_prompt

            mock_ollama_instance.chat_stream.assert_called_once_with(
                "Test prompt", 
                ollama_config.default_model, # Expects the default_model from config
                expected_settings
            )

@pytest.mark.asyncio
async def test_stream_chat_response_no_model_id_and_no_default(caplog, mock_app_config_for_chat_engine: AppConfig):
    """Test error if no model_id and no default_model is configured."""
    modified_providers = []
    for p_conf_dict in mock_app_config_for_chat_engine.model_dump().get("providers", []):
        if p_conf_dict["id"] == "ollama_default":
            p_conf_dict["default_model"] = None 
        modified_providers.append(ProviderConfig(**p_conf_dict))
    
    bad_config = AppConfig(
        default_provider="ollama_default",
        providers=modified_providers,
        # Ensure other required AppConfig fields are present if your model enforces them
        logging_enabled=mock_app_config_for_chat_engine.logging_enabled,
        # debug=mock_app_config_for_chat_engine.debug # if applicable
    )

    # Patch OllamaProvider to observe its instantiation and method calls
    with patch('backend.chat_engine.OllamaProvider', spec=OllamaProvider) as mock_ollama_provider_class:
        mock_ollama_instance = mock_ollama_provider_class.return_value
        # We don't need to configure chat_stream further for this test,
        # as it should not be called.

        with patch('backend.config.load_app_config', return_value=bad_config):
            chunks = []
            async for chunk in stream_chat_response(
                "Test prompt", "ollama_default", model_id=None, app_config=bad_config
            ):
                chunks.append(chunk)

            assert len(chunks) == 1
            assert "error" in chunks[0]
            assert "No model_id provided and no default_model configured" in chunks[0]["error"]
            assert "No model_id provided and no default_model configured" in caplog.text
            
            mock_ollama_provider_class.assert_called_once() # Provider class IS instantiated
            mock_ollama_instance.chat_stream.assert_not_called() # But its chat_stream method is NOT


@pytest.mark.asyncio
async def test_stream_chat_response_ollama_missing_host_config(caplog, mock_app_config_for_chat_engine: AppConfig):
    """Test error if Ollama provider is missing 'host' configuration."""
    modified_providers = []
    for p_conf_dict in mock_app_config_for_chat_engine.model_dump().get("providers", []):
        if p_conf_dict["id"] == "ollama_default":
            p_conf_dict["host"] = None # Remove host
        modified_providers.append(ProviderConfig(**p_conf_dict))
    
    bad_config = AppConfig(providers=modified_providers)

    with patch('backend.config.load_app_config', return_value=bad_config):
        chunks = []
        async for chunk in stream_chat_response(
            "Test", "ollama_default", "llama3", app_config=bad_config
        ):
            chunks.append(chunk)
        
        assert len(chunks) == 1
        assert "error" in chunks[0]
        assert "Ollama provider 'ollama_default' is missing required 'host' configuration." in chunks[0]["error"]
        assert "Ollama provider 'ollama_default' is missing required 'host' configuration." in caplog.text