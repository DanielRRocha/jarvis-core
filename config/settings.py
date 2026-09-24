from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # LLM Provider Settings
    LLM_PROVIDER: str = "ollama"  # Options: ollama, nvidia
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"
    NVIDIA_API_KEY: Optional[str] = None
    NVIDIA_MODEL: str = "nemotron-3-super-120b-a12b"

    # Database Settings
    DATABASE_URL: str = "sqlite:///./jarvis.db"

    # System Settings
    LOG_LEVEL: str = "INFO"
    MAX_CONVERSATION_HISTORY: int = 10
    MAX_FACTS_PER_USER: int = 100

    # Wake Word Settings (for V1)
    WAKE_WORD: str = "hey_jarvis"
    WAKE_WORD_SENSITIVITY: float = 0.5

    # TTS Settings (for V1)
    TTS_ENGINE: str = "say"  # Options: say, piper
    PIPER_MODEL_PATH: str = "./models/piper/en_US-lessac-medium.onnx"

    # Claude Code Integration Settings (for V2)
    CLAUDE_CODE_PATH: str = "/usr/local/bin/claude"
    AUTO_APPROVE_TOOL_USE: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()