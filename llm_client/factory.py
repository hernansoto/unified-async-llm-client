from __future__ import annotations

from .anthropic_client import AnthropicClient
from .base import BaseLLMClient
from .gemini_client import GeminiClient
from .openai_client import OpenAIClient
from .schemas import LLMSettings, Provider


class LLMFactory:

    @staticmethod
    def create_client(
        settings: LLMSettings,
    ) -> BaseLLMClient:

        if settings.provider == Provider.OPENAI:
            return OpenAIClient(settings)

        if settings.provider == Provider.ANTHROPIC:
            return AnthropicClient(settings)

        if settings.provider == Provider.GEMINI:
            return GeminiClient(settings)

        raise ValueError(
            f"Proveedor no soportado: {settings.provider}"
        )