# J.A.R.V.I.S. Core

**Just A Rather Very Intelligent System** — Núcleo do assistente pessoal inteligente, pronto para ser empacotado como container Docker.

Este é o **core** do JARVIS, projetado para rodar como um microsserviço isolado. Fora deste repositório ficam: interface de voz (wake word, STT/TTS), integração com Claude Code, automações residenciais e outros serviços que se comunicam via API/message queue.

---

## Arquitetura

```
jarvis-core/
├── main.py                 # Entry point (CLI)
├── pyproject.toml          # Dependências e metadados do pacote
├── config/
│   └── settings.py         # Configuração via pydantic-settings (.env)
├── interfaces/
│   └── cli.py              # Interface de linha de comando interativa
├── services/
│   └── agent.py            # Orquestrador principal (LLM + Memória + Personalidade)
├── llm/
│   ├── base.py             # Interface abstrata BaseLLMProvider
│   ├── ollama_provider.py  # Provedor local via Ollama
│   └── nvidia_provider.py  # Provedor cloud via NVIDIA NIMs
└── database/
    ├── connection.py       # SQLite + context manager
    └── memory.py           # Memória de curto e longo prazo
```

---

## Funcionalidades

| Área | Descrição |
|------|-----------|
| **Multi-LLM** | Suporte a Ollama (local) e NVIDIA NIMs (cloud) com troca em runtime |
| **Memória Híbrida** | Curto prazo (histórico de conversa SQLite) + Longo prazo (fatos persistentes por usuário) |
| **Personalidade JARVIS** | Sistema de personalidade configurável (tom, formalidade, ironia britânica, consentimento prévio) |
| **CLI Interativa** | Comandos: `status`, `fato`, `lembrar`, `provedor`, `modelos`, `limpar`, `ajuda` |
| **Configuração** | 100% via variáveis de ambiente / `.env` (pydantic-settings) |
| **Docker-Ready** | Dependências mínimas, entrypoint definido, healthcheck via `is_available()` |

---

## Requisitos

- **Python ≥ 3.12**
- **Ollama** (opcional, para rodar modelos locais) — `ollama serve`
- **Chave NVIDIA API** (opcional, para NIMs) — configurar `NVIDIA_API_KEY`

---

## Instalação Local (Desenvolvimento)

```bash
# Clone e entre no diretório
git clone <repo>
cd jarvis-core

# Crie venv e instale dependências
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Configure variáveis de ambiente
cp .env.example .env
# Edite .env com suas chaves/modelos

# Rode
jarvis          # ou: python main.py
```

---

## Configuração (`.env`)

```env
# LLM Provider
LLM_PROVIDER=ollama              # ollama | nvidia

# Ollama (local)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# NVIDIA NIMs (cloud)
NVIDIA_API_KEY=sk-xxxxx
NVIDIA_MODEL=nemotron-3-super-120b-a12b

# Database
DATABASE_URL=sqlite:///./jarvis.db

# Sistema
LOG_LEVEL=INFO
MAX_CONVERSATION_HISTORY=10
MAX_FACTS_PER_USER=100

# Personalidade JARVIS
JARVIS_ADDRESS_FORM=Senhor       # Senhor | Senhora | nenhum
JARVIS_ENABLE_IRONY=true
JARVIS_VERBOSE_BRIEFING=true
```

> **Nota**: O arquivo `.env.example` contém todas as opções documentadas.

---

## Uso via CLI

```bash
$ jarvis
```

Comandos disponíveis dentro da CLI:

| Comando | Descrição |
|---------|-----------|
| `ajuda` / `help` | Mostra ajuda |
| `sair` / `exit` | Encerra |
| `limpar` | Limpa histórico de conversa |
| `status` | Mostra status do sistema (provedor, modelos, memória) |
| `fato <chave> <valor> [categoria]` | Armazena fato na memória longa |
| `lembrar <chave>` | Recupera fato armazenado |
| `provedor <ollama\|nvidia>` | Troca provedor LLM em runtime |
| `modelos` | Lista modelos disponíveis no provedor atual |

---

## Docker

### Build

```bash
docker build -t jarvis-core:latest .
```

### Run

```bash
docker run -d \
  --name jarvis-core \
  --restart unless-stopped \
  -p 8000:8000 \                    # Porta da API (futuro)
  -v $(pwd)/data:/app/data \        # Persistência do SQLite
  -v $(pwd)/.env:/app/.env:ro \     # Configuração
  -e OLLAMA_HOST=host.docker.internal:11434 \  # Acesso ao Ollama no host
  jarvis-core:latest
```

> **Importante**: Para usar Ollama local rodando no host, passe `OLLAMA_HOST=host.docker.internal:11434` (Linux/Mac) ou configure a rede `host` no Docker.

### Docker Compose (Recomendado)

```yaml
# docker-compose.yml
services:
  jarvis-core:
    build: .
    container_name: jarvis-core
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./.env:/app/.env:ro
    environment:
      - OLLAMA_HOST=host.docker.internal:11434
    depends_on:
      - ollama  # Se rodar Ollama no compose

  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    # Para GPU NVIDIA:
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]

volumes:
  ollama_data:
```

```bash
docker compose up -d
# Baixe o modelo no container Ollama:
docker exec -it ollama ollama pull llama3.2:3b
```

---

## Estrutura de Dados (SQLite)

### `conversas` — Memória de Curto Prazo
```sql
id, papel (user/assistant), conteudo, timestamp
```
- Limitado por `MAX_CONVERSATION_HISTORY` (padrão: 10)
- Usado como contexto imediato para o LLM

### `fatos` — Memória de Longo Prazo
```sql
id, chave (UNIQUE), valor, categoria, timestamp
```
- Persistente entre sessões
- Categorias livres (ex: `preferencias`, `projetos`, `contatos`)
- Injetado no system prompt a cada interação

---

## Extensão: Adicionando Novos Provedores LLM

1. Crie `llm/novo_provider.py` herdando `BaseLLMProvider`
2. Implemente `generate_response()` e `is_available()`
3. Registre em `services/agent.py` no método `_initialize_llm_provider()`
4. Adicione configurações em `config/settings.py`

---

## Roadmap (Fora do Core)

| Módulo | Descrição | Comunicação |
|--------|-----------|-------------|
| `jarvis-voice` | Wake word (Porcupine), STT (Whisper), TTS (Piper/Say) | gRPC / REST |
| `jarvis-automation` | Home Assistant, MQTT, dispositivos IoT | MQTT / Webhooks |
| `jarvis-code` | Integração Claude Code (execução de tarefas de dev) | STDIN/STDOUT / API |
| `jarvis-api` | FastAPI wrapper para expor o core via HTTP | REST / WebSocket |

---

## Licença

MIT — Daniel Rocha

---

## Referências

- [Ollama](https://ollama.com/)
- [NVIDIA NIMs](https://www.nvidia.com/en-us/nim/)
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [Loguru](https://loguru.readthedocs.io/)