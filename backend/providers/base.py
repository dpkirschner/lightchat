"""Base module for LLM provider abstractions."""
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional


class LLMProvider(ABC):
    """Abstract base class for LLM providers.
    
    All concrete LLM provider implementations must inherit from this class
    and implement all abstract methods.
    """
    
    @abstractmethod
    def get_id(self) -> str:
        """Return a unique identifier for this provider instance.
        
        Returns:
            str: A unique string identifier (e.g., "ollama_local_1")
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return a user-friendly display name for this provider.
        
        Returns:
            str: A human-readable name (e.g., "Ollama (Local)")
        """
        pass
    
    @abstractmethod
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all available models from this provider.
        
        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing model details.
        """
        pass
    
    @abstractmethod
    async def chat_stream(
        self, 
        prompt: str, 
        model_id: Optional[str] = None, 
        settings: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, str], None]:
        """Stream a chat completion from the provider.
        
        Args:
            prompt: The user's input prompt
            model_id: Optional model ID to use for this request
            settings: Optional provider-specific settings for this request
            
        Yields:
            Dict[str, str]: Chunks of the response in the format {"token": "..."}
        """
        pass
