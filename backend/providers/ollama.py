"""Ollama provider implementation for interacting with local Ollama instances."""
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from .base import LLMProvider

logger = logging.getLogger(__name__)

class OllamaProvider(LLMProvider):
    """Provider for interacting with a local Ollama instance."""

    def __init__(self, provider_id: str, display_name: str, ollama_base_url: str = "http://localhost:11434"):
        self.provider_id = provider_id
        self.display_name = display_name
        self.ollama_base_url = ollama_base_url.rstrip('/')

    def get_id(self) -> str:
        return self.provider_id

    def get_name(self) -> str:
        return self.display_name

    async def list_models(self) -> List[Dict[str, Any]]:
        # This method remains the same as your last correct version
        url = f"{self.ollama_base_url}/api/tags"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                models = []
                for model_data in data.get("models", []):
                    details = model_data.get("details", {})
                    models.append({
                        "id": model_data.get("model", ""),
                        "name": model_data.get("model", ""),
                        "modified_at": model_data.get("modified_at", ""),
                        "size": model_data.get("size", 0),
                        "parameter_size": details.get("parameter_size"),
                        "quantization_level": details.get("quantization_level")
                    })
                return models
        except httpx.RequestError as e:
            logger.error(f"Failed to connect to Ollama API at {url} for list_models: {str(e)}")
            return []
        except httpx.HTTPStatusError as e:
            logger.error(f"Error processing Ollama API response from {url} for list_models: {str(e)}. Response: {e.response.text}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from {url} for list_models: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching models from Ollama at {url}: {str(e)}", exc_info=True)
            return []

    async def chat_stream(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, str], None]:
        if not model_id:
            error_msg = "model_id is required for Ollama chat requests"
            logger.error(error_msg)
            yield {"error": error_msg}
            return

        url = f"{self.ollama_base_url}/api/chat"
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }
        if settings:
            payload.update(settings)

        response_obj: Optional[httpx.Response] = None # Define for finally block
        try:
            async with httpx.AsyncClient() as client:
                response_obj = await client.post(url, json=payload, timeout=None)
                # Manually manage the response lifecycle instead of 'async with response_obj'
                try:
                    response_obj.raise_for_status()
                    buffer = ""
                    async for raw_chunk in response_obj.aiter_text():
                        buffer += raw_chunk
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()

                            if not line:
                                continue

                            if line.startswith('data: '):
                                json_str = line[len('data: '):].strip()
                                if not json_str:
                                    continue
                                if json_str == "[DONE]":
                                    logger.debug("Ollama stream: [DONE] signal received.")
                                    return 
                            elif line.startswith(':'):
                                logger.debug(f"Ollama stream: Comment: {line}")
                                continue
                            else:
                                logger.debug(f"Ollama stream: Skipping non-data line: {line}")
                                continue

                            try:
                                data = json.loads(json_str)
                                
                                # 1. Check for and yield content first
                                if "message" in data and "content" in data["message"]:
                                    content = data["message"]["content"]
                                    if content:
                                        yield {"token": content}
                                
                                # 2. Then, check for an error from Ollama in this message
                                if "error" in data:
                                    ollama_error_msg = data['error']
                                    logger.error(f"Ollama API error in stream: {ollama_error_msg}")
                                    yield {"error": f"Ollama API error: {ollama_error_msg}"}
                                    return # Stop generation on error from Ollama

                                # 3. Finally, check if this message indicates the stream is done
                                if data.get("done"): 
                                    logger.debug("Ollama stream: 'done: true' signal received.")
                                    return # Now, this return happens AFTER content (if any) is yielded

                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse Ollama JSON response line: '{json_str}'. Error: {str(e)}")
                                yield {"error": "Failed to parse Ollama stream response"}
                                return
                            except Exception as e:
                                logger.error(f"Error processing Ollama stream data: '{json_str}'. Error: {str(e)}", exc_info=True)
                                yield {"error": "Error processing Ollama stream data"}
                                return
                    if buffer.strip():
                        logger.debug(f"Ollama stream: Trailing buffer content: {buffer.strip()}")
                finally:
                    if response_obj is not None:
                        await response_obj.aclose() # Ensure the response is closed

        except httpx.RequestError as e:
            error_msg = f"Failed to connect to Ollama API at {url} for chat_stream: {str(e)}"
            logger.error(error_msg)
            yield {"error": "Failed to connect to Ollama service"}
        except httpx.HTTPStatusError as e:
            # This block might be hit if response_obj.raise_for_status() is outside the new try/finally
            # or if client.post itself fails with HTTPStatusError before streaming.
            # The current structure has raise_for_status inside the try/finally that calls aclose.
            error_msg = f"Ollama API returned HTTP error for chat_stream: {e.response.status_code} - {e.response.text}"
            logger.error(error_msg)
            yield {"error": f"Ollama API error: {e.response.status_code}"}
        except Exception as e:
            error_msg = f"Unexpected error in Ollama chat stream: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield {"error": "An unexpected error occurred"}