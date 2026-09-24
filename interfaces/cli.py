import asyncio
import sys
import signal
from typing import Optional
from services.agent import JarvisAgent
from config.settings import settings
from loguru import logger


class JarvisCLI:
    """Command-line interface for JARVIS."""

    def __init__(self):
        self.agent = JarvisAgent()
        self.running = False

        # Configure logger
        logger.remove()  # Remove default handler
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=settings.LOG_LEVEL
        )

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False

    def _print_banner(self):
        """Print the JARVIS banner."""
        banner = """
╔═════════════════════════════════════════════════════════════════════════════╗
║                                                                             ║
║         ██╗     █████╗     ██████╗     ██╗   ██╗    ██╗    ███████╗         ║
║         ██║    ██╔══██╗    ██╔══██╗    ██║   ██║    ██║    ██╔════╝         ║
║         ██║    ███████║    ██████╔╝    ██║   ██║    ██║    ███████╗         ║
║    ██╗  ██║    ██╔══██║    ██╔══██╗    ╚██╗ ██╔╝    ██║    ╚════██║         ║
║    ╚█████╔╝    ██║  ██║    ██║  ██║     ╚████╔╝     ██║    ███████║         ║
║     ╚════╝     ╚═╝  ╚═╝    ╚═╝  ╚═╝      ╚═══╝      ╚═╝    ╚══════╝         ║
║                                                                             ║
║                      Just A Rather Very Intelligent System                  ║
║                                Versão 0.1.0                                 ║
║                                                                             ║
╚═════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        print("Digite 'sair' ou 'exit' para encerrar.")
        print("Digite 'ajuda' ou 'help' para ver comandos disponíveis.")
        print("-" * 60)

    def _print_help(self):
        """Print help information."""
        help_text = """
Comandos disponíveis:
  ajuda, help     - Mostra esta ajuda
  sair, exit      - Encerra o JARVIS
  limpar          - Limpa o histórico da conversa
  status          - Mostra o status do sistema
  fato <chave> <valor> [categoria] - Adiciona um fato sobre o usuário
  lembrar <chave> - Recupera um fato sobre o usuário
  provedor <nome> - Troca o provedor LLM (ollama, nvidia)
  modelos         - Lista os modelos disponíveis do provedor atual
        """
        print(help_text)

    async def _handle_command(self, user_input: str) -> bool:
        """Handle special commands. Returns True if should continue, False to exit."""
        input_lower = user_input.strip().lower()

        if input_lower in ['sair', 'exit', 'quit']:
            return False

        elif input_lower in ['ajuda', 'help']:
            self._print_help()
            return True

        elif input_lower == 'limpar':
            self.agent.clear_conversation_history()
            print("Histórico da conversa limpo.")
            return True

        elif input_lower == 'status':
            status = self.agent.get_system_status()
            print("\n=== Status do Sistema ===")
            print(f"Provedor LLM: {status['llm_provider']}")
            print(f"LLM Disponível: {'Sim' if status['llm_available'] else 'Não'}")
            print(f"Ollama Disponível: {'Sim' if status['ollama_available'] else 'Não'}")
            print(f"NVIDIA Disponível: {'Sim' if status['nvidia_available'] else 'Não'}")
            print(f"Fatos na memória: {status['memory_facts_count']}")
            print(f"Mensagens na conversa: {status['conversation_history_count']}")
            print("=" * 24)
            return True

        elif input_lower.startswith('fato '):
            parts = user_input[5:].strip().split(' ', 2)
            if len(parts) >= 2:
                key, value = parts[0], parts[1]
                category = parts[2] if len(parts) > 2 else None
                if self.agent.add_user_fact(key, value, category):
                    print(f"Fato adicionado: {key} = {value}" + (f" [{category}]" if category else ""))
                else:
                    print("Erro ao adicionar fato")
            else:
                print("Uso: fato <chave> <valor> [categoria]")
            return True

        elif input_lower.startswith('lembrar '):
            key = user_input[8:].strip()
            if key:
                value = self.agent.get_user_fact(key)
                if value is not None:
                    print(f"{key}: {value}")
                else:
                    print(f"Fato '{key}' não encontrado")
            else:
                print("Uso: lembrar <chave>")
            return True

        elif input_lower.startswith('provedor '):
            provider = user_input[9:].strip()
            if provider:
                print(f"Trocando para provedor {provider}...")
                success = await self.agent.switch_provider(provider)
                if success:
                    print(f"Provedor alterado para {provider}")
                else:
                    print(f"Falha ao alterar para provedor {provider}")
            else:
                print("Uso: provedor <nome> (ollama ou nvidia)")
            return True

        elif input_lower == 'modelos':
            # Try to get available models from current provider
            try:
                # Check if we have an Ollama provider with a client
                if hasattr(self.agent.llm_provider, 'client') and hasattr(self.agent.llm_provider, 'base_url'):
                    models_response = self.agent.llm_provider.client.list()

                    # Handle different possible response structures from ollama library
                    models_list = []
                    if isinstance(models_response, dict):
                        # Standard format: {'models': [...]}
                        if 'models' in models_response and isinstance(models_response['models'], list):
                            models_list = models_response['models']
                        else:
                            # Maybe the response itself is a list of models
                            # Or try to find any list in the response
                            for key, value in models_response.items():
                                if isinstance(value, list):
                                    models_list = value
                                    break
                    elif isinstance(models_response, list):
                        # Direct list format
                        models_list = models_response

                    # If we still don't have a list, try to see if the response is the model list itself
                    if not models_list and isinstance(models_response, dict) and 'name' in models_response:
                        # Single model response?
                        models_list = [models_response]

                    # Extract model names safely (handles different possible keys)
                    model_names = []
                    for model in models_list:
                        if isinstance(model, dict):
                            # Try different possible keys for model name
                            model_name = model.get('name') or model.get('model')
                            if model_name:
                                model_names.append(str(model_name))
                        elif isinstance(model, str):
                            # If it's just a string
                            model_names.append(model)

                    if model_names:
                        print("\nModelos disponíveis:")
                        for model_name in model_names:
                            print(f"  - {model_name}")
                    else:
                        print("Nenhum modelo encontrado ou formato de resposta inesperado")
                else:
                    print("Função não disponível para este provedor")
            except Exception as e:
                print(f"Erro ao listar modelos: {e}")
            return True

        else:
            # Not a special command, process as normal input
            return None

    async def run(self):
        """Main CLI loop."""
        self._setup_signal_handlers()
        self._print_banner()

        # Check LLM availability
        if not self.agent.llm_provider.is_available():
            print("⚠️  Aviso: Provedor LLM não está disponível. Verifique sua configuração.")
            print("   O JARVIS tentará funcionar, mas pode falhar ao gerar respostas.")
            print()

        self.running = True
        while self.running:
            try:
                # Get user input
                user_input = input("\nVocê: ").strip()

                if not user_input:
                    continue

                # Handle special commands
                command_result = await self._handle_command(user_input)
                if command_result is False:  # Exit command
                    break
                elif command_result is True:  # Handled command
                    continue

                # Process normal input
                print("\nJarvis: ", end="", flush=True)
                response = await self.agent.process_input(user_input)
                print(response)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Unexpected error in CLI loop: {e}")
                print(f"\nErro: {e}")
                print("Tente novamente ou digite 'sair' para encerrar.")

        # Graceful shutdown
        await self.agent.graceful_shutdown()
        print("\nJARVIS encerrado. Até logo!")


async def main():
    """Entry point for the CLI."""
    cli = JarvisCLI()
    await cli.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nJARVIS encerrado pelo usuário.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Erro fatal: {e}")
        sys.exit(1)