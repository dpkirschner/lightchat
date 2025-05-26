"""
Pydantic models for LLM provider metadata.
"""
from typing import List, Literal
from pydantic import BaseModel, ConfigDict

ProviderType = Literal["local", "cloud"]
ProviderStatus = Literal["configured", "needs_api_key", "unavailable"]

class ProviderMetadata(BaseModel):
    """Metadata for an LLM provider."""
    id: str
    name: str
    type: ProviderType
    status: ProviderStatus

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "ollama_default",
                "name": "Ollama",
                "type": "local",
                "status": "configured"
            }
        }
    )