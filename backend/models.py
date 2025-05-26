"""Pydantic models for the LightChat API."""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


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

    class Config:
        json_schema_extra = {
            "example": {
                "token": "Hello, how can I help you today?",
                "error": None
            },
            "example_error": {
                "token": None,
                "error": "Failed to connect to provider"
            }
        }
