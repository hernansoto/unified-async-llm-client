from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import openai
from openai import AsyncOpenAI

from .base import BaseLLMClient
from .schemas import (
    ChatMessage,
    LLMSettings,
    ModelParameters,
    ModelResponse,
    Provider,
)


class OpenAIClient(BaseLLMClient):
    def __init__(self, settings: LLMSettings) -> None:
        if settings.openai_api_key is None:
            raise ValueError("Falta OPENAI_API_KEY.")

        self._model = settings.openai_model
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            max_retries=settings.max_retries,
        )

    @property
    def provider(self) -> Provider:
        return Provider.OPENAI

    @property
    def model(self) -> str:
        return self._model

    @staticmethod
    def _messages(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
        return [
            {"role": message.role, "content": message.content}
            for message in messages
        ]

    async def generate(
        self,
        messages: Sequence[ChatMessage],
        params: ModelParameters,
    ) -> ModelResponse:
        try:
            response = await self._client.with_options(
                timeout=params.timeout_seconds
            ).chat.completions.create(
                model=self.model,
                messages=self._messages(messages),
                temperature=params.temperature,
                max_completion_tokens=params.max_tokens,
            )

            content = response.choices[0].message.content or ""

            return ModelResponse(
                success=True,
                provider=self.provider,
                model=self.model,
                content=content,
            )

        except openai.RateLimitError:
            return self._error(
                "Rate limit alcanzado. Intenta nuevamente más tarde."
            )
        except openai.APITimeoutError:
            return self._error(
                "La solicitud a OpenAI superó el tiempo de espera."
            )
        except openai.APIConnectionError:
            return self._error("No fue posible conectarse con OpenAI.")
        except openai.APIStatusError as exc:
            return self._error(
                f"OpenAI respondió con HTTP {exc.status_code}."
            )
        except Exception as exc:
            return self._error(
                f"Error inesperado de OpenAI: {type(exc).__name__}."
            )

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        params: ModelParameters,
    ) -> AsyncIterator[str]:
        try:
            stream = await self._client.with_options(
                timeout=params.timeout_seconds
            ).chat.completions.create(
                model=self.model,
                messages=self._messages(messages),
                temperature=params.temperature,
                max_completion_tokens=params.max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if not chunk.choices:
                    continue

                text = chunk.choices[0].delta.content
                if text:
                    yield text

        except openai.RateLimitError:
            yield "[ERROR] Rate limit de OpenAI alcanzado."
        except openai.APITimeoutError:
            yield "[ERROR] La solicitud a OpenAI superó el tiempo de espera."
        except openai.APIConnectionError:
            yield "[ERROR] No fue posible conectarse con OpenAI."
        except openai.APIStatusError as exc:
            yield f"[ERROR] OpenAI respondió con HTTP {exc.status_code}."
        except Exception as exc:
            yield f"[ERROR] Error inesperado de OpenAI: {type(exc).__name__}."

    def _error(self, message: str) -> ModelResponse:
        return ModelResponse(
            success=False,
            provider=self.provider,
            model=self.model,
            error=message,
        )
