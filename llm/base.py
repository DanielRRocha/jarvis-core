from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from config.settings import settings


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self):
        self.settings = settings

    @abstractmethod
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate a response from the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated response text
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM provider is available and configured correctly."""
        pass

    def _prepare_messages(
        self,
        user_input: str,
        system_prompt: str,
        conversation_history: List[Dict[str, str]] = None,
        facts_context: str = None
    ) -> List[Dict[str, str]]:
        """Prepare messages for the LLM with system prompt, history, and facts."""
        messages = [{"role": "system", "content": system_prompt}]

        # Add facts context if available
        if facts_context:
            messages.append({
                "role": "system",
                "content": f"Informações relevantes sobre o usuário:\n{facts_context}"
            })

        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)

        # Add current user input
        messages.append({"role": "user", "content": user_input})

        return messages