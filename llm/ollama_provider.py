import ollama
import time
import os
from typing import List, Dict, Any, Optional
from llm.base import BaseLLMProvider
from config.settings import settings
from loguru import logger


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider implementation."""

    def __init__(self):
        super().__init__()
        import os

        # Respect OLLAMA_HOST environment variable (same as Ollama CLI)
        ollama_host = os.environ.get("OLLAMA_HOST")
        if ollama_host:
            # Convert OLLAMA_HOST format to URL for ollama.Client
            # OLLAMA_HOST format: host:port or :port or host: (empty parts default to localhost or port 11434)
            if ollama_host.startswith("://"):
                # Already a full URL
                self.base_url = ollama_host
            elif ollama_host.startswith(":"):
                # :port -> http://localhost:port
                self.base_url = f"http://localhost{ollama_host}"
            elif ollama_host.endswith(":"):
                # host: -> http://host:11434
                host_part = ollama_host[:-1] or "localhost"
                self.base_url = f"http://{host_part}:11434"
            elif ":" in ollama_host:
                # host:port -> http://host:port
                host_part, port_part = ollama_host.split(":", 1)
                host = host_part or "localhost"
                port = port_part or "11434"
                self.base_url = f"http://{host}:{port}"
            else:
                # Just hostname or IP -> http://hostname:11434
                self.base_url = f"http://{ollama_host}:11434"
        else:
            # Use configured base_url from settings
            self.base_url = settings.OLLAMA_BASE_URL

        logger.debug(f"Ollama base URL set to: {self.base_url}")

        self.model = settings.OLLAMA_MODEL
        # Configure ollama client
        self.client = ollama.Client(host=self.base_url)

    def is_available(self) -> bool:
        """Check if Ollama is available and the model is accessible."""
        # Try multiple times with delay to handle Ollama startup time and model download
        max_retries = 3
        retry_delay = 2  # seconds

        logger.info(f"Checking Ollama availability (will retry for up to {max_retries * retry_delay // 60} minutes)...")

        for attempt in range(max_retries):
            try:
                # Try to list models to see if service is running
                models_response = self.client.list()
                logger.debug(f"Ollama list response: {models_response}")

                # Handle different possible response structures from ollama library
                logger.debug(f"Processing Ollama response of type: {type(models_response)}")
                models_list = []

                # Try to get models from the response, regardless of type
                if hasattr(models_response, 'models'):
                    # It might be a ListResponse or similar
                    models_attr = getattr(models_response, 'models')
                    if isinstance(models_attr, list):
                        models_list = models_attr
                        logger.debug(f"Extracted models list via .models attribute: {len(models_list)} models")
                    else:
                        logger.warning(f".models attribute is not a list: {type(models_attr)}")
                # Standard format: {'models': [...]}
                elif isinstance(models_response, dict):
                    if 'models' in models_response and isinstance(models_response['models'], list):
                        models_list = models_response['models']
                        logger.debug(f"Found models list in 'models' key: {len(models_list)} models")
                    # Handle direct list format
                elif isinstance(models_response, list):
                    models_list = models_response
                    logger.debug(f"Got direct list of models: {len(models_list)} models")
                # Handle single model response (dict with name field)
                elif isinstance(models_response, dict) and 'name' in models_response:
                    models_list = [models_response]
                    logger.debug(f"Got single model response: 1 model")
                else:
                    logger.warning(f"Unexpected Ollama response format: {type(models_response)} - {models_response}")
                    models_list = []

                # Extract model names - simplified and robust version
                model_names = []
                logger.debug(f"Starting model name extraction from {len(models_list)} items")

                for i, model in enumerate(models_list):
                    logger.debug(f"Processing model {i}: type={type(model)}, value={model}")

                    # Primary method: Check for .model attribute (what we see in Ollama logs)
                    if hasattr(model, 'model'):
                        model_name = getattr(model, 'model')
                        logger.debug(f"Model {i}: found .model attribute = {model_name!r}")
                        if model_name is not None:
                            model_name_str = str(model_name).strip()
                            if model_name_str:
                                model_names.append(model_name_str)
                                logger.debug(f"Model {i}: extracted name '{model_name_str}'")
                            else:
                                logger.warning(f"Model {i}: .model attribute is empty/whitespace: {model_name!r}")
                        else:
                            logger.warning(f"Model {i}: .model attribute is None")
                    # Fallback 1: Dictionary-style access
                    elif isinstance(model, dict):
                        model_name = model.get('model') or model.get('name')
                        logger.debug(f"Model {i}: dict lookup gave {model_name!r}")
                        if model_name is not None:
                            model_name_str = str(model_name).strip()
                            if model_name_str:
                                model_names.append(model_name_str)
                                logger.debug(f"Model {i}: extracted name '{model_name_str}' from dict")
                            else:
                                logger.warning(f"Model {i}: dict model name is empty/whitespace: {model_name!r}")
                        else:
                            logger.warning(f"Model {i}: no 'model' or 'name' key in dict")
                    # Fallback 2: String conversion
                    elif isinstance(model, str):
                        model_name_str = model.strip()
                        logger.debug(f"Model {i}: string value = {model_name_str!r}")
                        if model_name_str:
                            model_names.append(model_name_str)
                            logger.debug(f"Model {i}: extracted name '{model_name_str}' from string")
                        else:
                            logger.warning(f"Model {i}: string value is empty/whitespace: {model!r}")
                    # Fallback 3: Try to get any string-like attribute
                    else:
                        logger.debug(f"Model {i}: trying fallback attribute extraction")
                        # Try common attribute names
                        for attr_name in ['model', 'name', 'id']:
                            if hasattr(model, attr_name):
                                attr_value = getattr(model, attr_name)
                                if attr_value is not None:
                                    attr_str = str(attr_value).strip()
                                    if attr_str:
                                        model_names.append(attr_str)
                                        logger.debug(f"Model {i}: extracted name '{attr_str}' from .{attr_name}")
                                        break
                        else:
                            logger.warning(f"Model {i}: could not extract name from unknown type {type(model)}")

                logger.debug(f"Model name extraction complete. Found {len(model_names)} model names: {model_names}")

                logger.debug(f"Final model names list: {model_names}")

                # Check if our model is in the list
                model_to_check = self.model
                # For latest tag check, use base model name (without specific version tag)
                if ':' in self.model:
                    base_model = self.model.rsplit(':', 1)[0]
                    model_to_check_latest = f"{base_model}:latest"
                else:
                    model_to_check_latest = f"{self.model}:latest"

                logger.debug(f"Checking for model: '{model_to_check}' or latest: '{model_to_check_latest}'")

                is_available = model_to_check in model_names or model_to_check_latest in model_names
                if is_available:
                    logger.info(f"Ollama provider is available (model: {self.model})")
                    return is_available
                else:
                    # Model not found yet, but service is responding - might be downloading
                    if attempt < max_retries - 1:
                        logger.info(f"Ollama service responding but model {self.model} not found yet (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s")
                        time.sleep(retry_delay)
                    else:
                        # Last attempt and model still not found
                        logger.warning(f"Ollama service responding but model {self.model} not found after {max_retries} attempts")
                        logger.warning(f"Available models: {model_names}")
                        return False

            except Exception as e:
                # If this isn't the last attempt, wait and try again
                if attempt < max_retries - 1:
                    logger.debug(f"Ollama availability check failed (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s: {e}")
                    time.sleep(retry_delay)
                else:
                    # Last attempt failed
                    logger.warning(f"Ollama provider not available after {max_retries} attempts: {e}")
                    return False

        return False  # This line is actually unreachable but kept for clarity

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate a response using Ollama."""
        try:
            # Prepare options for ollama
            options = {
                "temperature": temperature,
            }
            if max_tokens:
                options["num_predict"] = max_tokens

            # Add any additional kwargs to options
            options.update(kwargs)

            # Generate response
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options=options
            )

            return response['message']['content']

        except Exception as e:
            logger.error(f"Error generating response with Ollama: {e}")
            raise RuntimeError(f"Ollama generation failed: {e}")