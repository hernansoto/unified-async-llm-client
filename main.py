import asyncio

from dotenv import load_dotenv

from llm_client import AsyncLLMManager, ChatMessage, LLMSettings


async def main() -> None:
    load_dotenv()

    settings = LLMSettings.from_env()
    params = LLMSettings.model_parameters_from_env()

    manager = AsyncLLMManager(
        settings=settings,
        default_params=params,
    )

    messages = [
        ChatMessage(
            role="system",
            content="Responde de forma clara, breve y didáctica.",
        ),
        ChatMessage(
            role="user",
            content="¿Qué es la entropía?",
        ),
    ]

    print(
        f"\nProveedor: {manager.provider_name} | "
        f"Modelo: {manager.model_name}"
    )

    print("\n=== MODO NORMAL ===\n")

    response = await manager.generate(messages)

    if response.success:
        print(response.content)
    else:
        print(f"[ERROR] {response.error}")

    print("\n=== MODO STREAMING ===\n")

    async for chunk in manager.stream(messages):
        print(chunk, end="", flush=True)

    print()


if __name__ == "__main__":
    asyncio.run(main())
