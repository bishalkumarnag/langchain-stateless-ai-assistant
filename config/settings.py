"""
settings.py - Environment & Application Configuration

Loads settings from .env and provides API keys and model defaults
for LangChain chat models (Google Gemini, OpenAI, etc.).
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# 1. Locate the project root and load the .env file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()


class Settings:
    """
    Central settings class holding credentials and default parameters.
    """

    def __init__(self):
        # 1. Active provider: "google" or "openai"
        self.provider: str = os.getenv("LLM_PROVIDER", "google").lower()

        # 2. Google Gemini configuration
        self.gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
        self.gemini_model: str = os.getenv("GEMINI_DEFAULT_MODEL", "gemini-3.5-flash-lite")

        # 3. OpenAI configuration (for seamless multi-provider switching)
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.openai_model: str = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")

    def validate_provider_keys(self) -> None:
        """
        Validates that the necessary API key for the active provider is configured.
        """
        if self.provider == "google":
            if not self.gemini_api_key or self.gemini_api_key.strip() == "" or "your_gemini" in self.gemini_api_key:
                raise ValueError(
                    "GEMINI_API_KEY is missing in your .env file!\n"
                    "Please set your key in .env or switch LLM_PROVIDER in .env."
                )
        elif self.provider == "openai":
            if not self.openai_api_key or self.openai_api_key.strip() == "" or "your_openai" in self.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is missing in your .env file!\n"
                    "Please set your OpenAI key in .env."
                )
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER '{self.provider}'. Supported: 'google', 'openai'")


# Singleton instance
_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
