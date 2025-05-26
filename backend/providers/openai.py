import os
import openai
from typing import List, Dict, Any, Optional, AsyncGenerator

from backend.providers.base import LLMProvider
# TODO: Define a Pydantic model for ModelInfo if not already defined elsewhere (e.g., in LC-005)
# from backend.schemas import ModelInfo # Assuming ModelInfo is defined here

class OpenAIProvider(LLMProvider):
    def __init__(self, provider_id: str, display_name: str, api_key: Optional[str] = None):
        self.provider_id = provider_id
        self.display_name = display_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client: Optional[openai.AsyncOpenAI] = None

        if self.api_key:
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
        else:
            # For now, we'll print a warning. In a real app, proper logging should be used.
            print(f"Warning: OpenAI API key not found for provider '{self.provider_id}'. Some functionalities will be limited.")

    def get_id(self) -> str:
        return self.provider_id

    def get_name(self) -> str:
        return self.display_name

    async def list_models(self) -> List[Dict[str, Any]]:
        # Implementation for list_models will go here
        # Static list as per requirements
        static_models = [
            {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "owned_by": "openai", "created_at": 0},
            {"id": "gpt-4", "name": "GPT-4", "owned_by": "openai", "created_at": 0},
            {"id": "gpt-4-turbo-preview", "name": "GPT-4 Turbo Preview", "owned_by": "openai", "created_at": 0}
        ]

        if not self.client:
            print("Warning: OpenAI client not initialized. Returning static list of models.")
            return static_models

        try:
            models_page = await self.client.models.list()
            transformed_models = []
            for model in models_page.data:
                transformed_models.append({
                    "id": model.id,
                    "name": model.id, # Or a more descriptive name if available and desired
                    "owned_by": model.owned_by,
                    "created_at": model.created # Ensure this is an int timestamp
                })
            return transformed_models
        except openai.APIConnectionError as e:
            print(f"OpenAI API Connection Error: {e}. Returning static list of models.")
            return static_models
        except openai.AuthenticationError as e:
            print(f"OpenAI API Authentication Error: {e}. Returning static list of models.")
            return static_models
        except openai.RateLimitError as e:
            print(f"OpenAI API Rate Limit Error: {e}. Returning static list of models.")
            return static_models
        except openai.APIError as e:
            print(f"OpenAI API Error: {e}. Returning static list of models.")
            return static_models

    async def chat_stream(self, prompt: str, model_id: Optional[str] = None, settings: Optional[Dict[str, Any]] = None) -> AsyncGenerator[Dict[str, str], None]:
        if not self.client or not self.api_key:
            yield {"error": "OpenAI API key not configured."}
            return

        if not model_id:
            yield {"error": "Model ID is required for OpenAI chat stream."}
            return

        # Ensure settings is a dict if None
        if settings is None:
            settings = {}

        try:
            stream = await self.client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}], # Basic message structure
                stream=True,
                temperature=settings.get("temperature"), # Pass through if available
                max_tokens=settings.get("max_tokens")    # Pass through if available
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        yield {"token": delta_content}
        except openai.APIConnectionError as e:
            print(f"OpenAI API Connection Error during chat stream: {e}")
            yield {"error": f"OpenAI API Error: APIConnectionError - {str(e)}"}
        except openai.AuthenticationError as e:
            print(f"OpenAI API Authentication Error during chat stream: {e}")
            yield {"error": f"OpenAI API Error: AuthenticationError - {str(e)}"}
        except openai.RateLimitError as e:
            print(f"OpenAI API Rate Limit Error during chat stream: {e}")
            yield {"error": f"OpenAI API Error: RateLimitError - {str(e)}"}
        except openai.BadRequestError as e: # e.g. invalid model_id
            print(f"OpenAI API Bad Request Error during chat stream: {e}")
            yield {"error": f"OpenAI API Error: BadRequestError - {str(e)}"}
        except openai.APIError as e:
            print(f"OpenAI API Error during chat stream: {e}")
            yield {"error": f"OpenAI API Error: {type(e).__name__} - {str(e)}"}
        except Exception as e:
            # Catch any other unexpected errors
            print(f"Unexpected error during OpenAI chat stream: {e}")
            yield {"error": f"Unexpected error: {type(e).__name__} - {str(e)}"}
