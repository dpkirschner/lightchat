from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from backend.models.providers import ProviderMetadata, ProviderStatus, ProviderType

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
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
