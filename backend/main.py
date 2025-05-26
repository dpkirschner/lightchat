import json
import logging
from typing import List, Optional, AsyncGenerator, Dict, Any
import atexit

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from backend.models import ModelInfo, ProviderMetadata, ChatRequest, SSEEvent
from backend.providers.ollama import OllamaProvider
from backend.providers.openai import OpenAIProvider
from backend.chat_engine import stream_chat_response
from backend.config import load_app_config, AppConfig, ProviderConfig
from backend.logger import setup_logging
from typing import Union, Optional


def get_provider_instance(provider_id: str, providers: List[ProviderConfig]) -> Union[OllamaProvider, OpenAIProvider]:
    """Get an instance of the specified provider.
    
    Args:
        provider_id: The ID of the provider to get an instance for
        providers: List of provider configurations
        
    Returns:
        An instance of the specified provider
        
    Raises:
        KeyError: If the provider is not found or not properly configured
    """
    provider_config = next((p for p in providers if p.id == provider_id), None)
    if not provider_config:
        raise KeyError(f"Provider '{provider_id}' not found")
        
    if provider_config.type == "ollama":
        if not provider_config.host:
            raise ValueError(f"Ollama provider '{provider_id}' is missing required 'host' configuration")
        return OllamaProvider(
            provider_id=provider_config.id,
            display_name=provider_config.name,
            ollama_base_url=provider_config.host
        )
    elif provider_config.type == "openai":
        if not provider_config.api_key:
            raise ValueError(f"OpenAI provider '{provider_id}' is missing required 'api_key' configuration")
        return OpenAIProvider(
            provider_id=provider_config.id,
            display_name=provider_config.name,
            api_key=provider_config.api_key
        )
    else:
        raise ValueError(f"Unsupported provider type: {provider_config.type}")

logger = logging.getLogger("lightchat.main")

# Global variable to hold the log listener
log_listener = None

# Global app config - initialized on first use
_app_config = None

def get_app_config() -> AppConfig:
    """Get the application configuration, loading it if necessary."""
    global _app_config
    if _app_config is None:
        _app_config = load_app_config()
    return _app_config

app = FastAPI(
    title="LightChat API",
    description="Backend API for LightChat application",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "LightChat Backend Active"}

@app.get("/health", status_code=200)
async def health_check():
    return {"status": "ok"}

@app.get(
    "/models/{provider_id}",
    response_model=List[ModelInfo],
    summary="List available models from a provider",
    description="Returns a list of models available from the specified provider.",
    response_description="A list of model information objects"
)
async def list_models(
    provider_id: str,
    config: AppConfig = Depends(get_app_config)
) -> List[ModelInfo]:
    """
    List all available models from the specified provider.
    
    Args:
        provider_id: The ID of the provider to list models from
        
    Returns:
        List[ModelInfo]: A list of model information objects
        
    Raises:
        HTTPException: If the provider is not found or not configured
    """
    try:
        # Get the provider instance
        provider = get_provider_instance(provider_id, config.providers)
        
        # Get and return the list of models
        models_data = await provider.list_models()
        return [ModelInfo(**model_data) for model_data in models_data]
        
    except KeyError as e:
        logger.warning(f"Provider '{provider_id}' not found in configuration")
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except ValueError as e:
        logger.error(f"Configuration error for provider '{provider_id}': {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error listing models for provider '{provider_id}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error listing models: {str(e)}"
        )


@app.get(
    "/providers",
    response_model=List[ProviderMetadata],
    summary="List available LLM providers",
    description="Returns metadata for all configured LLM providers.",
    response_description="A list of provider metadata objects"
)
async def get_providers(
    config: AppConfig = Depends(get_app_config)
) -> List[ProviderMetadata]:
    """
    Retrieve metadata for all configured LLM providers.
    
    Returns:
        List[ProviderMetadata]: A list of provider metadata objects with status
    """
    providers_metadata = []
    
    # Debug log the providers in the config
    logger.debug(f"Found {len(config.providers)} providers in config: {[p.id for p in config.providers]}")
    
    for provider in config.providers:
        # Default status is configured
        status: str = "configured"
        
        # Check provider-specific requirements
        if provider.type == "openai" and not provider.api_key:
            status = "needs_api_key"
        elif provider.type == "ollama" and not provider.host:
            status = "unavailable"
        
        # Map provider type to the expected ProviderType
        provider_type: str = "cloud" if provider.type == "openai" else "local"
        
        # Create the provider metadata
        provider_meta = ProviderMetadata(
            id=provider.id,
            name=provider.name,
            type=provider_type,  # type: ignore[arg-type]
            status=status  # type: ignore[arg-type]
        )
        
        providers_metadata.append(provider_meta)
        logger.debug(f"Added provider: {provider_meta}")
    
    logger.debug(f"Returning {len(providers_metadata)} configured providers")
    return providers_metadata

async def chat_event_generator(
    chat_request: ChatRequest,
    config: AppConfig = Depends(get_app_config)
) -> AsyncGenerator[Dict[str, str], None]:
    """Generate SSE events from the chat stream.
    
    Args:
        chat_request: The chat request data
        config: The application configuration
        
    Yields:
        Dict with 'data' key containing the JSON-encoded event
    """
    try:
        # Log the chat request
        logger.info(
            "Processing chat request",
            extra={
                "event": "chat_request_received",
                "provider_id": chat_request.provider_id,
                "model_id": chat_request.model_id,
                "has_settings": bool(chat_request.settings)
            }
        )
        
        # Stream the chat response
        async for event_data_dict in stream_chat_response(
            prompt=chat_request.prompt,
            provider_id=chat_request.provider_id,
            model_id=chat_request.model_id,
            settings=chat_request.settings or {},
            app_config=config
        ):
            sse_event = SSEEvent(**event_data_dict)
            current_event_type = "error" if sse_event.error else "message"
            yield {
                "event": current_event_type,
                "data": sse_event.model_dump_json()
            }
    except Exception as e:
        logger.exception("Error in chat event generator")
        # Send error as SSE event
        error_event = SSEEvent(error=f"An error occurred: {str(e)}")
        yield {
            "event": "error",
            "data": error_event.model_dump_json()
        }


@app.post(
    "/chat",
    response_class=EventSourceResponse,
    summary="Stream chat completion",
    description="Stream chat completion using the specified provider and model",
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "Stream of chat completion events",
        }
    }
)
async def chat_endpoint(chat_request: ChatRequest):
    """Stream chat completion from the specified provider.
    
    Args:
        chat_request: The chat request data
        
    Returns:
        EventSourceResponse: Server-Sent Events stream
    """
    logger.info(f"Received chat request for provider: {chat_request.provider_id}")
    
    # Return the event source response
    return EventSourceResponse(
        chat_event_generator(chat_request),
        media_type="text/event-stream",
        ping=30,  # Send ping every 30 seconds to keep connection alive
        ping_message_factory=lambda: {"event": "ping"}
    )


if __name__ == "__main__":
    import uvicorn
    
    # Initialize our custom logging system
    # Note: setup_logging configures the 'lightchat' logger.
    # Logs from other modules (like uvicorn or FastAPI) won't go through this
    # unless they are configured to use a child of 'lightchat' or the root logger
    # is also configured with these handlers (which logger.py currently doesn't do).
    app_config = get_app_config()
    if app_config.logging_enabled:  # Assuming we might want to disable it via config later
        log_listener_instance = setup_logging(app_config)
        
        # Store the listener globally so it can be stopped
        globals()['log_listener'] = log_listener_instance

        # Get the application-specific logger instance
        lightchat_logger = logging.getLogger("lightchat")

        # Log application start with provider info
        provider_info = [
            f"{p.type} ({p.id}): {p.name}" 
            for p in app_config.providers
        ]
        
        lightchat_logger.info(
            "LightChat backend starting...", 
            extra={
                'event': 'application_startup',
                'provider_count': len(app_config.providers),
                'providers': provider_info,
                'default_provider': app_config.default_provider
            }
        )
        
        # Log provider configuration status
        for provider in app_config.providers:
            status = "configured"
            if provider.type == "openai" and not provider.api_key:
                status = "needs_api_key"
            elif provider.type == "ollama" and not provider.host:
                status = "needs_configuration"
                
            lightchat_logger.info(
                f"Provider configured: {provider.id}",
                extra={
                    'event': 'provider_config_status',
                    'provider_id': provider.id,
                    'provider_type': provider.type,
                    'status': status,
                    'has_host': bool(provider.host),
                    'has_api_key': bool(provider.api_key)
                }
            )

        # Register shutdown handler
        def shutdown_logging():
            lightchat_logger.info(
                "LightChat backend shutting down...", 
                extra={'event_type': 'application_shutdown'}
            )
            if globals()['log_listener']:
                globals()['log_listener'].stop()
                lightchat_logger.info(
                    "Logging queue listener stopped.", 
                    extra={'event_type': 'logging_shutdown'}
                )

        atexit.register(shutdown_logging)
    else:
        # Fallback to basic logging if our custom logging is disabled
        logging.basicConfig(level=logging.INFO)
        logging.info("Custom logging disabled, using basicConfig.")

    # The uvicorn logger configuration below is for uvicorn's own server logs.
    # It's separate from our application's 'lightchat' logger.
    # We remove the generic logging.basicConfig that was here before.
    # logger = logging.getLogger("uvicorn") # This line is fine if you want to reference uvicorn's logger
    
    # Log startup complete
    lightchat_logger = logging.getLogger("lightchat")
    lightchat_logger.info(
        "LightChat backend ready", 
        extra={'event': 'application_ready'}
    )
    
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
