"""
Chat engine module for handling chat interactions with different LLM providers.
"""
import logging
from typing import Any, AsyncGenerator, Dict, Optional, List

# Import providers
from .providers.ollama import OllamaProvider
from .providers.openai import OpenAIProvider
from .config import AppConfig, ProviderConfig

# Get logger instance
logger = logging.getLogger("lightchat.chat_engine")


class ProviderNotFoundError(Exception):
    """Raised when a requested provider is not found."""
    pass


async def stream_chat_response(
    prompt: str,
    provider_id: str,
    model_id: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None,
    app_config: Optional[AppConfig] = None
) -> AsyncGenerator[Dict[str, str], None]:
    """Stream chat response from the specified provider.
    
    Args:
        prompt: The user's input prompt
        provider_id: Identifier for the provider to use (e.g., 'ollama_default')
        model_id: Optional model ID to use for the request
        settings: Optional provider-specific settings for the chat request
        app_config: Optional AppConfig instance for provider configuration
        
    Yields:
        Dict[str, str]: Chunks of the chat response in the format {"token": "..."}
        or error messages in the format {"error": "..."}
    """
    # Log the incoming request
    logger.info(
        "Processing chat request",
        extra={
            "event": "chat_request_start",
            "provider_id": provider_id,
            "model_id": model_id,
            "prompt_length": len(prompt),
            "has_settings": settings is not None
        }
    )

    # Get provider configuration
    if app_config is None:
        from .config import load_app_config
        app_config = load_app_config()
    
    provider_config = app_config.get_provider(provider_id)
    if not provider_config:
        error_msg = f"Provider '{provider_id}' not found in configuration."
        logger.error(error_msg, extra={"event": "provider_not_found", "provider_id": provider_id})
        yield {"error": error_msg}
        return

    # Initialize the provider based on configuration
    provider = None
    try:
        if provider_config.type == "ollama":
            if not provider_config.host:
                error_msg = f"Ollama provider '{provider_id}' is missing required 'host' configuration."
                logger.error(error_msg, extra={"event": "provider_config_error", "provider_id": provider_id})
                yield {"error": error_msg}
                return
                
            provider = OllamaProvider(
                provider_id=provider_config.id,
                display_name=provider_config.name,
                ollama_base_url=provider_config.host
            )
            
        elif provider_config.type == "openai":
            if not provider_config.api_key:
                error_msg = f"OpenAI provider '{provider_id}' is missing required 'api_key' configuration."
                logger.error(error_msg, extra={"event": "provider_config_error", "provider_id": provider_id})
                yield {"error": error_msg}
                return
                
            provider = OpenAIProvider(
                provider_id=provider_config.id,
                display_name=provider_config.name,
                api_key=provider_config.api_key
            )
        else:
            error_msg = f"Unsupported provider type: {provider_config.type}"
            logger.error(error_msg, extra={"event": "unsupported_provider_type", "provider_type": provider_config.type})
            yield {"error": error_msg}
            return
            
        # Log the provider initialization
        logger.debug(
            f"Initialized {provider_config.type} provider: {provider_id}",
            extra={
                "event": "provider_initialized",
                "provider_id": provider_id,
                "provider_type": provider_config.type
            }
        )
        
        # Prepare settings for the provider
        provider_settings = settings or {}
        if provider_config.system_prompt and "system_prompt" not in provider_settings:
            provider_settings["system_prompt"] = provider_config.system_prompt
            
        # Use default model if none provided
        model_to_use = model_id or provider_config.default_model
        if not model_to_use:
            error_msg = f"No model_id provided and no default_model configured for provider: {provider_id}"
            logger.error(error_msg, extra={"event": "missing_model_id", "provider_id": provider_id})
            yield {"error": error_msg}
            return
            
        # Log the start of the chat stream
        logger.info(
            "Starting chat stream",
            extra={
                "event": "chat_stream_start",
                "provider_id": provider_id,
                "model_id": model_to_use,
                "has_system_prompt": "system_prompt" in provider_settings
            }
        )
        
        # Stream the response
        response_tokens = []
        has_error = False
        
        async for event_data in provider.chat_stream(prompt, model_to_use, provider_settings):
            if "error" in event_data:
                logger.error(
                    "Error in chat stream",
                    extra={
                        "event": "chat_stream_error",
                        "provider_id": provider_id,
                        "model_id": model_to_use,
                        "error": event_data["error"]
                    }
                )
                has_error = True
            elif "token" in event_data:
                response_tokens.append(event_data["token"])
                
            yield event_data
            
        # Log the completion of the chat stream
        if has_error:
            logger.warning(
                "Chat stream completed with errors",
                extra={
                    "event": "chat_stream_complete_with_errors",
                    "provider_id": provider_id,
                    "model_id": model_to_use,
                    "response_length": len("".join(response_tokens))
                }
            )
        else:
            logger.info(
                "Chat stream completed successfully",
                extra={
                    "event": "chat_stream_complete",
                    "provider_id": provider_id,
                    "model_id": model_to_use,
                    "response_length": len("".join(response_tokens))
                }
            )
            
    except NotImplementedError as e:
        error_msg = f"Chat stream not implemented for provider '{provider_id}'. Details: {str(e)}"
        logger.error(
            error_msg,
            extra={
                "event": "not_implemented_error",
                "provider_id": provider_id,
                "error": str(e)
            },
            exc_info=True
        )
        yield {"error": error_msg}
        
    except Exception as e:
        error_msg = f"Error in chat stream for provider '{provider_id}': {str(e)}"
        logger.error(
            error_msg,
            extra={
                "event": "chat_stream_error",
                "provider_id": provider_id,
                "model_id": model_id,
                "error": str(e)
            },
            exc_info=True
        )
        yield {"error": error_msg}