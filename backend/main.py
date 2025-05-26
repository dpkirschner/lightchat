import json
import logging
from typing import List, Optional, AsyncGenerator, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from backend.models.providers import ModelInfo, ProviderMetadata, ProviderStatus, ProviderType
from backend.providers.ollama import OllamaProvider
from backend.chat_engine import stream_chat_response
from backend.models.providers import ChatRequest, SSEEvent

logger = logging.getLogger(__name__)

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
async def list_models(provider_id: str) -> List[ModelInfo]:
    """
    List all available models from the specified provider.
    
    Args:
        provider_id: The ID of the provider to list models from
        
    Returns:
        List[ModelInfo]: A list of model information objects
        
    Raises:
        HTTPException: If the provider is not found
    """
    # For now, we only support the hardcoded Ollama provider
    if provider_id == "ollama_default":
        provider = OllamaProvider(
            provider_id="ollama_default",
            display_name="Ollama"
        )
        models_data = await provider.list_models()
        return [ModelInfo(**model_data) for model_data in models_data]
    
    raise HTTPException(
        status_code=404,
        detail=f"Provider '{provider_id}' not found"
    )


@app.get(
    "/providers",
    response_model=List[ProviderMetadata],
    summary="List available LLM providers",
    description="Returns metadata for all configured LLM providers.",
    response_description="A list of provider metadata objects"
)
async def get_providers() -> List[ProviderMetadata]:
    """
    Retrieve metadata for all configured LLM providers.
    
    For now, returns a hardcoded list with a single Ollama provider.
    In future implementations, this will be dynamically generated from configuration.
    """
    return [
        ProviderMetadata(
            id="ollama_default",
            name="Ollama",
            type="local",
            status="configured"
        )
    ]

async def chat_event_generator(chat_request: ChatRequest) -> AsyncGenerator[Dict[str, str], None]:
    """Generate SSE events from the chat stream.
    
    Args:
        chat_request: The chat request data
        
    Yields:
        Dict with 'data' key containing the JSON-encoded event
    """
    try:
        async for event_data_dict in stream_chat_response(
            prompt=chat_request.prompt,
            provider_id=chat_request.provider_id,
            model_id=chat_request.model_id,
            settings=chat_request.settings
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
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("uvicorn")
    
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
