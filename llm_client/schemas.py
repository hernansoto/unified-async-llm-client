from __future__ import annotations

import os
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class ChatMessage(BaseModel):
    """Mensaje normalizado para cualquier proveedor."""

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ModelParameters(BaseModel):
    """Parámetros comunes de generación validados con Pydantic."""

    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=300, ge=1, le=128_000)
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)


class ModelResponse(BaseModel):
    """Respuesta normalizada del cliente unificado."""

    success: bool
    provider: Provider
    model: str
    content: str = ""
    error: str | None = None


class LLMSettings(BaseModel):
    """Configuración general y secretos de los proveedores."""

    provider: Provider = Provider.OPENAI

    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None

    openai_model: str = "gpt-4.1-mini"
    anthropic_model: str = "claude-sonnet-4-6"
    gemini_model: str = "gemini-3.6-flash"

    max_retries: int = Field(default=2, ge=0, le=10)

    @model_validator(mode="after")
    def validate_selected_provider_key(self) -> "LLMSettings":

        if self.provider == Provider.OPENAI and self.openai_api_key is None:
            raise ValueError(
                "OPENAI_API_KEY es obligatoria cuando LLM_PROVIDER=openai."
            )

        if (
            self.provider == Provider.ANTHROPIC
            and self.anthropic_api_key is None
        ):
            raise ValueError(
                "ANTHROPIC_API_KEY es obligatoria "
                "cuando LLM_PROVIDER=anthropic."
            )

        if self.provider == Provider.GEMINI and self.gemini_api_key is None:
            raise ValueError(
                "GEMINI_API_KEY es obligatoria cuando LLM_PROVIDER=gemini."
            )

        return self

    @classmethod
    def from_env(cls) -> "LLMSettings":

        provider_value = os.getenv(
            "LLM_PROVIDER",
            Provider.OPENAI.value,
        ).lower()

        return cls(
            provider=Provider(provider_value),

            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            gemini_api_key=os.getenv("GEMINI_API_KEY") or None,

            openai_model=os.getenv(
                "OPENAI_MODEL",
                "gpt-4.1-mini",
            ),

            anthropic_model=os.getenv(
                "ANTHROPIC_MODEL",
                "claude-sonnet-4-6",
            ),

            gemini_model=os.getenv(
                "GEMINI_MODEL",
                "gemini-2.5-flash",
            ),

            max_retries=int(
                os.getenv("MAX_RETRIES", "2")
            ),
        )

    @staticmethod
    def model_parameters_from_env() -> ModelParameters:

        return ModelParameters(
            temperature=float(
                os.getenv("TEMPERATURE", "0.2")
            ),
            max_tokens=int(
                os.getenv("MAX_TOKENS", "300")
            ),
            timeout_seconds=float(
                os.getenv("REQUEST_TIMEOUT", "30")
            ),
        )