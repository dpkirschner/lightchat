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

        max_attempts = 2  # 1 initial attempt + 1 retry
        retry_delay = 2  # seconds

        for attempt in range(1, max_attempts + 1):
            client = None
            response = None
            try:
                if attempt > 1:
                    logger.info(f"Retrying Ollama request (attempt {attempt}/{max_attempts})...")
                    import asyncio
                    await asyncio.sleep(retry_delay)

                client = httpx.AsyncClient()
                response = await client.post(url, json=payload, timeout=None)
                
                # Check for retryable status codes
                if response.status_code in {502, 503, 504}:
                    if attempt < max_attempts:
                        logger.warning(f"Received retryable status {response.status_code}, will retry...")
                        continue
                    else:
                        error_msg = f"Ollama API returned retryable status {response.status_code} after {max_attempts} attempts"
                        logger.error(error_msg)
                        yield {"error": f"Ollama API error: {response.status_code}"}
                        return

                # If we get here, either the request was successful or it's a non-retryable error
                response.raise_for_status()
                
                # If we reach here, the request was successful - process the stream
                buffer = ""
                async for raw_chunk in response.aiter_text():
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
                                return  # Stop generation on error from Ollama

                            # 3. Finally, check if this message indicates the stream is done
                            if data.get("done"): 
                                logger.debug("Ollama stream: 'done: true' signal received.")
                                return  # Now, this return happens AFTER content (if any) is yielded

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
                
                # If we've successfully processed the entire stream, break out of the retry loop
                break

            except httpx.RequestError as e:
                if attempt < max_attempts:
                    logger.warning(f"Request error (attempt {attempt}/{max_attempts}): {str(e)}")
                    continue
                else:
                    error_msg = f"Failed to connect to Ollama API after {max_attempts} attempts: {str(e)}"
                    logger.error(error_msg)
                    yield {"error": "Failed to connect to Ollama service after multiple retries"}
                    return

            except httpx.HTTPStatusError as e:
                if e.response.status_code in {502, 503, 504} and attempt < max_attempts:
                    logger.warning(f"Received retryable status {e.response.status_code}, will retry...")
                    continue
                else:
                    error_msg = f"Ollama API returned HTTP error for chat_stream: {e.response.status_code} - {e.response.text}"
                    logger.error(error_msg)
                    yield {"error": f"Ollama API error: {e.response.status_code}"}
                    return

            except Exception as e:
                error_msg = f"Unexpected error in Ollama chat stream: {str(e)}"
                logger.error(error_msg, exc_info=True)
                yield {"error": "An unexpected error occurred"}
                return

            finally:
                # Ensure resources are properly cleaned up
                try:
                    if response is not None:
                        await response.aclose()
                    if client is not None:
                        await client.aclose()
                except Exception as e:
                    logger.error(f"Error cleaning up resources: {str(e)}", exc_info=True)