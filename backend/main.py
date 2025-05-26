from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

from backend.models.providers import ModelInfo, ProviderMetadata, ProviderStatus, ProviderType
from backend.providers.ollama import OllamaProvider

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

if __name__ == "__main__":
    import uvicorn
    import logging
    
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
