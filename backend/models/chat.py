"""
Chat-related Pydantic models for the LightChat API.
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

from .schemas import SSEEvent


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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "Hello, how are you?",
                "provider_id": "openai",
                "model_id": "gpt-4",
                "settings": {"temperature": 0.7}
            }
        }
    )


__all__ = ['ChatRequest', 'SSEEvent']
