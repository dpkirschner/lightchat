"""
Pydantic models for LLM provider metadata, model information, and chat functionality.
"""
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

ProviderType = Literal["local", "cloud"]
ProviderStatus = Literal["configured", "needs_api_key", "unavailable"]


class ModelInfo(BaseModel):
    """Information about an available model from a provider."""
    id: str = Field(..., description="Unique identifier for the model")
    name: str = Field(..., description="Display name of the model")
    modified_at: str = Field(..., description="ISO 8601 timestamp of when the model was last modified")
    size: int = Field(..., description="Size of the model in bytes")
    parameter_size: Optional[str] = Field(
        None,
        description="Size of the model parameters (e.g., '7B', '13B')"
    )
    quantization_level: Optional[str] = Field(
        None,
        description="Quantization level of the model (e.g., 'Q4_0', 'Q5_K_M')"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "llama3:latest",
                "name": "llama3:latest",
                "modified_at": "2023-10-29T19:22:00.000000Z",
                "size": 4117063800,
                "parameter_size": "7B",
                "quantization_level": "Q4_0"
            }
        }
    )

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


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    prompt: str = Field(..., description="The user's input prompt")
    provider_id: str = Field(..., description="ID of the LLM provider to use")
    model_id: Optional[str] = Field(
        None,
        description="ID of the model to use. If not provided, the provider's default will be used"
    )
    settings: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional provider-specific settings"
    )


class SSEEvent(BaseModel):
    """Model for Server-Sent Events (SSE) data payload."""
    token: Optional[str] = Field(
        None,
        description="A chunk of the generated text response"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if an error occurred"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "token": "Hello, how can I help you today?",
                "error": None
            },
            "example_error": {
                "token": None,
                "error": "Failed to connect to provider"
            }
        }
    )