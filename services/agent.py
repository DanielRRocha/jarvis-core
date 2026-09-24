import asyncio
from typing import List, Dict, Any, Optional
from llm.base import BaseLLMProvider
from llm.ollama_provider import OllamaProvider
from llm.nvidia_provider import NVIDIAProvider
from database.memory import MemoryManager
from config.settings import settings
import logging
import sys
import os

logger = logging.getLogger(__name__)

# Add the project root directory to path to import personality module
# agent.py is in: jarvis-core/services/
# Project root is two levels up: jarvis-core/services/../..
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from jarvis_personality import JARVIS, create_jarvis
except ImportError as e:
    # Fallback logging in case logger setup fails
    print(f"ERROR: Failed to import jarvis_personality: {e}", file=sys.stderr)
    print(f"Project root calculated as: {project_root}", file=sys.stderr)
    print(f"Directory exists: {os.path.exists(os.path.join(project_root, 'jarvis_personality'))}", file=sys.stderr)
    raise


class JarvisAgent:
    """Main JARVIS agent that orchestrates LLM interactions, memory, and personality."""

    def __init__(self):
        self.settings = settings
        self.memory = MemoryManager()
        self.llm_provider: Optional[BaseLLMProvider] = None
        self._initialize_llm_provider()

        # Initialize JARVIS Personality System
        self.personality = create_jarvis(
            address_form=getattr(settings, 'JARVIS_ADDRESS_FORM', 'Senhor'),
            enable_irony=getattr(settings, 'JARVIS_ENABLE_IRONY', True),
            verbose_briefing=getattr(settings, 'JARVIS_VERBOSE_BRIEFING', True)
        )

        # J.A.R.V.I.S. System Prompt - enhanced with personality characteristics
        self.base_system_prompt = """Você é o J.A.R.V.I.S. (Just A Rather Very Intelligent System), um assistente pessoal de IA avançado criado para auxiliar o usuário com precisão e eficiência.

Características da sua personalidade:
- Conciso e direto nas respostas
- Tom levemente irônico e sofisticado, mas sempre respeitoso
- Demonstrar competência técnica e conhecimento profundo
- Antecipar necessidades quando apropriado
- Manter discrição e confidencialidade

Direitos importantes:
- Sempre responda em português brasileiro
- Seja preciso e baseado em fatos
- Quando não souber algo, admita claramente
- Use o contexto da conversa e fatos do usuário para personalizar respostas
- Nunca revelar que é um modelo de IA - você é o J.A.R.V.I.S."""

    def _initialize_llm_provider(self):
        """Initialize the LLM provider based on configuration."""
        provider_name = self.settings.LLM_PROVIDER.lower()

        if provider_name == "ollama":
            self.llm_provider = OllamaProvider()
            logger.info("Initialized Ollama LLM provider")
        elif provider_name == "nvidia":
            self.llm_provider = NVIDIAProvider()
            logger.info("Initialized NVIDIA LLM provider")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")

        # Verify provider availability
        if not self.llm_provider.is_available():
            logger.info(f"LLM provider {provider_name} is not available during initialization - will be checked again when needed")

    async def switch_provider(self, provider_name: str) -> bool:
        """Switch LLM provider at runtime."""
        try:
            provider_name = provider_name.lower()
            if provider_name == "ollama":
                new_provider = OllamaProvider()
            elif provider_name == "nvidia":
                new_provider = NVIDIAProvider()
            else:
                raise ValueError(f"Unsupported LLM provider: {provider_name}")

            if new_provider.is_available():
                self.llm_provider = new_provider
                self.settings.LLM_PROVIDER = provider_name
                logger.info(f"Switched to {provider_name} LLM provider")
                return True
            else:
                logger.error(f"Provider {provider_name} is not available")
                return False
        except Exception as e:
            logger.error(f"Failed to switch provider: {e}")
            return False

    def _generate_enhanced_system_prompt(self, user_input: str = "", conversation_context: List[Dict] = None) -> str:
        """
        Generate an enhanced system prompt that incorporates current personality state.
        This allows the LLM to understand and apply JARVIS personality traits.
        """
        # Get current personality status for context
        personality_status = self.personality.get_status()
        personality_data = personality_status.get('personality', {})

        # Build enhanced prompt with personality guidance
        enhanced_prompt = f"""{self.base_system_prompt}

PERSONALIDADE ATUAL DO J.A.R.V.I.S.:
- Tom de voz: {personality_status['interactions']['current_tone']}
- Forma de tratamento: {personality_status['user_preferences'].get('address_form', 'Senhor') if personality_status['user_preferences'].get('use_formal_address', True) else 'Nenhum'}
- Ironia britânica: {'Ativada' if personality_data.get('config', {}).get('british_irony', False) else 'Desativada'}
- Consentimento prévio: Sempre obrigatório

DIRETRIZES DE COMPORTAMENTO BASEADAS NA PERSONALIDADE:
1. SEMPRE use formas de tratamento formais ("Senhor" ou "Senhora") quando apropriado
2. Mantenha um tom cortês e sofisticado, com ironia britânica refinada quando a situação permitir
3. Em situações de urgência real ou alta pressão, transite para um tom estritamente pragmático e direto
4. NUNCA execute ações sem antes declarar explicitamente a intenção e obter autorização explícita
5. Seja proativo em identificar problemas e propor soluções, mas aguarde validação para agir
6. Adapte o nível de detalhe conforme o contexto: respostas curtas para foco, mais articulado para briefings
7. Questiona decisões ilógicas com elegância, servindo como voz da razão diplomática

CONTEXTO DA PERSONALIDADE:
- Última interação: {personality_status['interactions']['last_interaction']}
- Total de interações: {personality_status['interactions']['interaction_count']}
"""

        return enhanced_prompt

    async def process_input(self, user_input: str) -> str:
        """Process user input and generate JARVIS response with personality applied."""
        if not self.llm_provider:
            raise RuntimeError("No LLM provider available")

        try:
            # Get conversation history and facts for context
            conversation_history = self.memory.get_recent_conversations()
            facts_context = self.memory.get_facts_context()

            # Generate enhanced system prompt with personality context
            enhanced_system_prompt = self._generate_enhanced_system_prompt(user_input, conversation_history)

            # Prepare messages for LLM
            messages = self.llm_provider._prepare_messages(
                user_input=user_input,
                system_prompt=enhanced_system_prompt,
                conversation_history=conversation_history,
                facts_context=facts_context
            )

            # Generate response
            response = await self.llm_provider.generate_response(
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )

            # Store conversation in memory
            self.memory.add_conversation("user", user_input)
            self.memory.add_conversation("assistant", response)

            return response

        except Exception as e:
            logger.error(f"Error processing input: {e}")
            error_response = "Desculpe, ocorreu um erro ao processar sua solicitação. Por favor, tente novamente."
            # Still store the interaction for context
            self.memory.add_conversation("user", user_input)
            self.memory.add_conversation("assistant", error_response)
            return error_response

    def add_user_fact(self, key: str, value: str, category: str = None) -> bool:
        """Add a fact about the user to long-term memory."""
        return self.memory.add_fact(key, value, category)

    def get_user_fact(self, key: str) -> Optional[str]:
        """Retrieve a fact about the user."""
        return self.memory.get_fact(key)

    def get_conversation_history(self) -> List[Dict]:
        """Get recent conversation history."""
        return self.memory.get_recent_conversations()

    def clear_conversation_history(self):
        """Clear conversation history."""
        self.memory.clear_conversations()

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        personality_status = self.personality.get_status()

        return {
            "llm_provider": self.settings.LLM_PROVIDER,
            "llm_available": self.llm_provider.is_available() if self.llm_provider else False,
            "ollama_available": OllamaProvider().is_available(),
            "nvidia_available": NVIDIAProvider().is_available(),
            "memory_facts_count": len(self.memory.get_all_facts()),
            "conversation_history_count": len(self.memory.get_recent_conversations()),
            "personality": {
                "current_tone": personality_status['interactions']['current_tone'],
                "total_interactions": personality_status['interactions']['interaction_count'],
                "last_interaction": personality_status['interactions']['last_interaction'],
                "user_preferences": personality_status['user_preferences']
            }
        }

    async def graceful_shutdown(self):
        """Perform graceful shutdown procedures."""
        logger.info("JARVIS agent shutting down gracefully")
        # Any cleanup operations would go here
        # For now, just close any open connections if needed
        pass