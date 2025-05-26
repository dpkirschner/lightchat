"""
Chat engine module for handling chat interactions with different LLM providers.
"""
import logging
from typing import Any, AsyncGenerator, Dict, Optional

# Assuming your providers are in a 'providers' subdirectory relative to this file
from .providers.ollama import OllamaProvider
# from .providers.openai import OpenAIProvider # Example for future
# from .providers.base import LLMProvider # Example if you have a base class

logger = logging.getLogger(__name__)


class ProviderNotFoundError(Exception):
    """Raised when a requested provider is not found."""
    pass


async def stream_chat_response(
    prompt: str,
    provider_id: str,
    model_id: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None
) -> AsyncGenerator[Dict[str, str], None]:
    """Stream chat response from the specified provider.
    
    Args:
        prompt: The user's input prompt
        provider_id: Identifier for the provider to use (e.g., 'ollama_default')
        model_id: Optional model ID to use for the request
        settings: Optional provider-specific settings for the chat request
        
    Yields:
        Dict[str, str]: Chunks of the chat response in the format {"token": "..."}
        or error messages in the format {"error": "..."}
    """
    provider: Any = None # Initialize with a type hint if you have a base LLMProvider class

    # Provider selection logic
    # This will expand as you add more providers
    if provider_id == "ollama_default":
        provider = OllamaProvider(
            provider_id="ollama_default", # This ID should match what this engine expects
            display_name="Ollama",
            ollama_base_url="http://localhost:11434" # Consider making this configurable
        )
    # Example for another provider:
    # elif provider_id == "openai_default":
    #     provider = OpenAIProvider(
    #         provider_id="openai_default",
    #         display_name="OpenAI",
    #         api_key="YOUR_API_KEY_FROM_CONFIG_OR_ENV"
    #     )
    else:
        error_msg = f"Provider '{provider_id}' not recognized."
        logger.error(error_msg)
        yield {"error": error_msg}
        return

    # Stream response from the selected provider
    try:
        # Cleaner way: Directly iterate, assuming provider.chat_stream()
        # returns an async generator object as per its contract.
        async for event_data in provider.chat_stream(prompt, model_id, settings):
            yield event_data
    except NotImplementedError as e:
        # This handles cases where the provider's chat_stream is explicitly not implemented.
        error_msg = f"Chat stream not implemented for provider '{provider_id}'. Details: {str(e)}"
        logger.warning(f"NotImplementedError from provider '{provider_id}': {str(e)}") # Log as warning
        yield {"error": error_msg}
    except Exception as e:
        # This catches other unexpected errors from the provider's stream.
        error_msg = f"Error streaming from provider '{provider_id}': {str(e)}"
        logger.error(error_msg, exc_info=True) # Log with traceback for debugging
        # Yielding the full str(e) can be useful for debugging but consider if it's too verbose for clients.
        # You might prefer a more generic message for the client in some cases.
        yield {"error": error_msg}