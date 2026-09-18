from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import anthropic
from anthropic import AsyncAnthropic

from .base import BaseLLMClient
from .schemas import (
    ChatMessage,
    LLMSettings,
    ModelParameters,
    ModelResponse,
    Provider,
)


class AnthropicClient(BaseLLMClient):
    def __init__(self, settings: LLMSettings) -> None:
        if settings.anthropic_api_key is None:
            raise ValueError("Falta ANTHROPIC_API_KEY.")

        self._model = settings.anthropic_model
        self._client = AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            max_retries=settings.max_retries,
        )

    @property
    def provider(self) -> Provider:
        return Provider.ANTHROPIC

    @property
    def model(self) -> str:
        return self._model

    @staticmethod
    def _prepare_messages(
        messages: Sequence[ChatMessage],
    ) -> tuple[str | None, list[dict[str, str]]]:
        system_messages = [
            message.content
            for message in messages
            if message.role == "system"
        ]

        provider_messages = [
            {"role": message.role, "content": message.content}
            for message in messages
            if message.role != "system"
        ]

        system = "\n\n".join(system_messages) if system_messages else None
        return system, provider_messages

    def _request_kwargs(
        self,
        messages: Sequence[ChatMessage],
        params: ModelParameters,
    ) -> dict:
        system, provider_messages = self._prepare_messages(messages)

        kwargs = {
            "model": self.model,
            "max_tokens": params.max_tokens,
            "messages": provider_messages,
            "extra_body": {"temperature": params.temperature},
        }

        if system:
            kwargs["system"] = system

        return kwargs

    async def generate(
        self,
        messages: Sequence[ChatMessage],
        params: ModelParameters,
    ) -> ModelResponse:
        try:
            response = await self._client.with_options(
                timeout=params.timeout_seconds
            ).messages.create(
                **self._request_kwargs(messages, params)
            )

            content = "".join(
                block.text
                for block in response.content
                if getattr(block, "type", None) == "text"
            )

            return ModelResponse(
                success=True,
                provider=self.provider,
                model=self.model,
                content=content,
            )

        except anthropic.RateLimitError:
            return self._error(
                "Rate limit de Anthropic alcanzado. Intenta nuevamente más tarde."
            )
        except anthropic.APITimeoutError:
            return self._error(
                "La solicitud a Anthropic superó el tiempo de espera."
            )
        except anthropic.APIConnectionError:
            return self._error("No fue posible conectarse con Anthropic.")
        except anthropic.APIStatusError as exc:
            return self._error(
                f"Anthropic respondió con HTTP {exc.status_code}."
            )
        except Exception as exc:
            return self._error(
                f"Error inesperado de Anthropic: {type(exc).__name__}."
            )

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        params: ModelParameters,
    ) -> AsyncIterator[str]:
        try:
            raw_stream = await self._client.with_options(
                timeout=params.timeout_seconds
            ).messages.create(
                **self._request_kwargs(messages, params),
                stream=True,
            )

            async for event in raw_stream:
                if getattr(event, "type", None) != "content_block_delta":
                    continue

                delta = getattr(event, "delta", None)
                if getattr(delta, "type", None) == "text_delta":
                    text = getattr(delta, "text", "")
                    if text:
                        yield text

        except anthropic.RateLimitError:
            yield "[ERROR] Rate limit de Anthropic alcanzado."
        except anthropic.APITimeoutError:
            yield "[ERROR] La solicitud a Anthropic superó el tiempo de espera."
        except anthropic.APIConnectionError:
            yield "[ERROR] No fue posible conectarse con Anthropic."
        except anthropic.APIStatusError as exc:
            yield f"[ERROR] Anthropic respondió con HTTP {exc.status_code}."
        except Exception as exc:
            yield f"[ERROR] Error inesperado de Anthropic: {type(exc).__name__}."

    def _error(self, message: str) -> ModelResponse:
        return ModelResponse(
            success=False,
            provider=self.provider,
            model=self.model,
            error=message,
        )
