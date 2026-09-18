from __future__ import annotations

import asyncio

from collections.abc import AsyncIterator, Sequence

from google import genai
from google.genai import errors, types

from .base import BaseLLMClient
from .schemas import (
    ChatMessage,
    LLMSettings,
    ModelParameters,
    ModelResponse,
    Provider,
)


class GeminiClient(BaseLLMClient):

    def __init__(self, settings: LLMSettings) -> None:

        if settings.gemini_api_key is None:
            raise ValueError("Falta GEMINI_API_KEY.")

        self._model = settings.gemini_model

        client = genai.Client(
            api_key=settings.gemini_api_key.get_secret_value()
        )

        # Cliente asíncrono oficial
        self._client = client.aio

    @property
    def provider(self) -> Provider:
        return Provider.GEMINI

    @property
    def model(self) -> str:
        return self._model

    @staticmethod
    def _prepare_messages(
        messages: Sequence[ChatMessage],
    ) -> tuple[str | None, list[types.Content]]:

        system_messages = [
            message.content
            for message in messages
            if message.role == "system"
        ]

        system_instruction = (
            "\n\n".join(system_messages)
            if system_messages
            else None
        )

        contents: list[types.Content] = []

        for message in messages:

            if message.role == "system":
                continue

            role = (
                "model"
                if message.role == "assistant"
                else "user"
            )

            contents.append(
                types.Content(
                    role=role,
                    parts=[
                        types.Part.from_text(
                            text=message.content
                        )
                    ],
                )
            )

        return system_instruction, contents

    @staticmethod
    def _generation_config(
        system_instruction: str | None,
        params: ModelParameters,
    ) -> types.GenerateContentConfig:

        return types.GenerateContentConfig(
    system_instruction=system_instruction,
    max_output_tokens=params.max_tokens,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(
        disable=True
    ),
)

    async def generate(
        self,
        messages: Sequence[ChatMessage],
        params: ModelParameters,
    ) -> ModelResponse:

        system_instruction, contents = self._prepare_messages(
            messages
        )

        try:

            async with asyncio.timeout(
                params.timeout_seconds
            ):

                response = await self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=self._generation_config(
                        system_instruction,
                        params,
                    ),
                )

            return ModelResponse(
                success=True,
                provider=self.provider,
                model=self.model,
                content=response.text or "",
            )

        except TimeoutError:

            return self._error(
                "La solicitud a Gemini superó "
                "el tiempo de espera."
            )

        except errors.APIError as exc:

            if exc.code == 429:
                return self._error(
                    "Rate limit de Gemini alcanzado. "
                    "Intenta nuevamente más tarde."
                )

            return self._error(
                f"Gemini respondió con error "
                f"HTTP {exc.code}: {exc.message}"
            )

        except Exception as exc:

            return self._error(
                f"Error inesperado de Gemini: "
                f"{type(exc).__name__}."
            )

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        params: ModelParameters,
    ) -> AsyncIterator[str]:

        system_instruction, contents = self._prepare_messages(
            messages
        )

        try:

            async with asyncio.timeout(
                params.timeout_seconds
            ):

                stream = (
                    await self._client.models.generate_content_stream(
                        model=self.model,
                        contents=contents,
                        config=self._generation_config(
                            system_instruction,
                            params,
                        ),
                    )
                )

                async for chunk in stream:

                    if chunk.text:
                        yield chunk.text

        except TimeoutError:

            yield (
                "[ERROR] La solicitud a Gemini "
                "superó el tiempo de espera."
            )

        except errors.APIError as exc:

            if exc.code == 429:
                yield (
                    "[ERROR] Rate limit de Gemini alcanzado."
                )
            else:
                yield (
                    f"[ERROR] Gemini respondió con "
                    f"HTTP {exc.code}: {exc.message}"
                )

        except Exception as exc:

            yield (
                f"[ERROR] Error inesperado de Gemini: "
                f"{type(exc).__name__}."
            )

    def _error(
        self,
        message: str,
    ) -> ModelResponse:

        return ModelResponse(
            success=False,
            provider=self.provider,
            model=self.model,
            error=message,
        )