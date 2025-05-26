"""Ollama provider implementation for interacting with local Ollama instances."""
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from .base import LLMProvider

logger = logging.getLogger(__name__)

class OllamaProvider(LLMProvider):
    """Provider for interacting with a local Ollama instance."""

    def __init__(self, provider_id: str, display_name: str, ollama_base_url: str = "http://localhost:11434"):
        """Initialize the Ollama provider.
        
        Args:
            provider_id: Unique identifier for this provider instance
            display_name: User-friendly display name
            ollama_base_url: Base URL for the Ollama API (default: http://localhost:11434)
        """
        self.provider_id = provider_id
        self.display_name = display_name
        self.ollama_base_url = ollama_base_url.rstrip('/')

    def get_id(self) -> str:
        """Return the unique identifier for this provider."""
        return self.provider_id

    def get_name(self) -> str:
        """Return the user-friendly display name for this provider."""
        return self.display_name

    async def list_models(self) -> List[Dict[str, Any]]:
        """List all available models from the Ollama instance.
        
        Returns:
            List[Dict[str, Any]]: A list of model dictionaries with id, name, modified_at, size,
                                 parameter_size, and quantization_level.
        """
        url = f"{self.ollama_base_url}/api/tags"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                
                models = []
                for model in data.get("models", []):
                    details = model.get("details", {})
                    models.append({
                        "id": model.get("model", ""),
                        "name": model.get("model", ""),
                        "modified_at": model.get("modified_at", ""),
                        "size": model.get("size", 0),
                        "parameter_size": details.get("parameter_size"),
                        "quantization_level": details.get("quantization_level")
                    })
                return models
                
        except httpx.RequestError as e:
            logger.error(f"Failed to connect to Ollama API at {url}: {str(e)}")
            return []
        except (httpx.HTTPStatusError, KeyError) as e:
            logger.error(f"Error processing Ollama API response from {url}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching models from Ollama: {str(e)}")
            return []

    async def chat_stream(
        self, 
        prompt: str, 
        model_id: Optional[str] = None, 
        settings: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, str], None]:
        """Stream a chat completion from the Ollama API.
        
        Args:
            prompt: The user's input prompt
            model_id: Optional model ID to use for this request
            settings: Optional settings for the chat request
            
        Yields:
            Dict[str, str]: Chunks of the chat response
            
        Raises:
            NotImplementedError: This method is not implemented in this story
        """
        raise NotImplementedError("Chat functionality will be implemented in a subsequent story.")
        
        if False: # pragma: no cover
            yield {} # The type of yield should match AsyncGenerator's YieldType
