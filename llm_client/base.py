from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence

from .schemas import ChatMessage, ModelParameters, ModelResponse, Provider


class BaseLLMClient(ABC):
    """Interfaz común para clientes LLM asíncronos."""

    @property
    @abstractmethod
    def provider(self) -> Provider:
        raise NotImplementedError

    @property
    @abstractmethod
    def model(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def generate(
        self,
        messages: Sequence[ChatMessage],
        params: ModelParameters,
    ) -> ModelResponse:
        """Genera una respuesta completa sin bloquear el event loop."""
        raise NotImplementedError

    @abstractmethod
    async def stream(
        self,
        messages: Sequence[ChatMessage],
        params: ModelParameters,
    ) -> AsyncIterator[str]:
        """Entrega fragmentos de texto a medida que llegan del proveedor."""
        if False:
            yield ""
        raise NotImplementedError
