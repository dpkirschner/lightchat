"""
Base schemas used across the application.
"""
from typing import Optional
from pydantic import BaseModel, Field


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

    model_config = {
        "json_schema_extra": {
            "example": {
                "token": "Hello, how can I help you today?",
                "error": None
            },
            "example_error": {
                "token": None,
                "error": "Failed to connect to provider"
            }
        }
    }
