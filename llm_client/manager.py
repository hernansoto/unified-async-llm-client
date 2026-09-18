from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from .factory import LLMFactory
from .schemas import (
    ChatMessage,
    LLMSettings,
    ModelParameters,
    ModelResponse,
)


class AsyncLLMManager:
    """Fachada unificada para generar respuestas con distintos proveedores."""

    def __init__(
        self,
        settings: LLMSettings,
        default_params: ModelParameters | None = None,
    ) -> None:
        self._client = LLMFactory.create_client(settings)
        self._default_params = default_params or ModelParameters()

    @property
    def provider_name(self) -> str:
        return self._client.provider.value

    @property
    def model_name(self) -> str:
        return self._client.model

    async def generate(
        self,
        messages: Sequence[ChatMessage],
        params: ModelParameters | None = None,
    ) -> ModelResponse:
        return await self._client.generate(
            messages,
            params or self._default_params,
        )

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        params: ModelParameters | None = None,
    ) -> AsyncIterator[str]:
        async for chunk in self._client.stream(
            messages,
            params or self._default_params,
        ):
            yield chunk
