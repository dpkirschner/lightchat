"""
Pydantic models for the LightChat API.
"""
from typing import Literal

# Re-export types and models
from .schemas import SSEEvent  # noqa: F401
from .providers import (
    ModelInfo,
    ProviderMetadata,
    ProviderType,
    ProviderStatus,
    ChatRequest
)

# Re-export types for backward compatibility
ProviderType = Literal["local", "cloud"]
ProviderStatus = Literal["configured", "needs_api_key", "unavailable"]

__all__ = [
    'SSEEvent',
    'ModelInfo',
    'ProviderMetadata',
    'ProviderType',
    'ProviderStatus',
    'ChatRequest',
]
