from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from llm.base import BaseLLMProvider
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class NVIDIAProvider(BaseLLMProvider):
    """NVIDIA API (NIMs) LLM provider implementation."""

    def __init__(self):
        super().__init__()
        self.api_key = settings.NVIDIA_API_KEY
        self.model = settings.NVIDIA_MODEL
        self.client = None

        if self.api_key:
            self.client = AsyncOpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=self.api_key
            )

    def is_available(self) -> bool:
        """Check if NVIDIA API is available and configured."""
        return self.api_key is not None and self.client is not None

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate a response using NVIDIA API."""
        if not self.is_available():
            raise RuntimeError("NVIDIA API key not configured")

        try:
            # Prepare parameters
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens:
                params["max_tokens"] = max_tokens

            # Add any additional kwargs
            params.update(kwargs)

            # Generate response
            response = await self.client.chat.completions.create(**params)

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating response with NVIDIA API: {e}")
            raise RuntimeError(f"NVIDIA API generation failed: {e}")