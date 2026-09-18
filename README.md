# Unified Async LLM Client

Pre-entrega 1 del curso de AI Engineering.

Este proyecto implementa un cliente unificado y asíncrono para trabajar con
OpenAI y Anthropic bajo una misma interfaz.

## Objetivos cumplidos

- Interfaz común para OpenAI y Anthropic.
- Uso exclusivo de clientes asíncronos (`AsyncOpenAI` y `AsyncAnthropic`).
- Generación normal mediante `async/await`.
- Streaming mediante generadores asíncronos y `yield`.
- Validación con Pydantic.
- API keys almacenadas como `SecretStr`.
- Validación de `temperature` entre `0` y `2`.
- Validación de `max_tokens`.
- Selección del proveedor mediante `LLM_PROVIDER`.
- Manejo controlado de errores de conexión, timeout, rate limit y errores HTTP.
- Retries configurables mediante los SDK oficiales.
- Variables de entorno mediante `python-dotenv`.

## Estructura

```text
unified-async-llm-client/
├── llm_client/
│   ├── __init__.py
│   ├── schemas.py
│   ├── base.py
│   ├── factory.py
│   ├── manager.py
│   ├── openai_client.py
│   └── anthropic_client.py
├── .env.example
├── .gitignore
├── main.py
├── requirements.txt
└── README.md
```

## Requisitos

- Python 3.12+
- Una API key de OpenAI y/o Anthropic.

## 1. Crear el entorno virtual

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

## 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

Dependencias:

- `openai`
- `anthropic`
- `pydantic`
- `python-dotenv`

## 3. Configurar variables de entorno

Copia `.env.example` como `.env`.

### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

### Linux / macOS

```bash
cp .env.example .env
```

Ejemplo con OpenAI:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=tu_api_key
ANTHROPIC_API_KEY=

OPENAI_MODEL=gpt-4.1-mini
ANTHROPIC_MODEL=claude-sonnet-4-6

TEMPERATURE=0.2
MAX_TOKENS=300
REQUEST_TIMEOUT=30
MAX_RETRIES=2
```

Para Anthropic:

```env
LLM_PROVIDER=anthropic
OPENAI_API_KEY=
ANTHROPIC_API_KEY=tu_api_key
```

No subas `.env` al repositorio. Está excluido en `.gitignore`.

## 4. Ejecutar

```bash
python main.py
```

El script realiza la pregunta:

> ¿Qué es la entropía?

Primero obtiene la respuesta completa y después ejecuta la misma consulta en
modo streaming.

## Arquitectura

### `schemas.py`

`ChatMessage` normaliza `role` y `content`.

`ModelParameters` valida:

- `temperature`: entre `0` y `2`;
- `max_tokens`: entre `1` y `128000`;
- `timeout_seconds`: mayor a cero.

`LLMSettings` carga proveedor, modelos y API keys. Las claves se almacenan como
`SecretStr`.

`ModelResponse` normaliza respuestas correctas y errores.

### `BaseLLMClient`

Define el contrato:

```python
async def generate(...)
async def stream(...)
```

### `OpenAIClient` y `AnthropicClient`

Implementan el contrato usando los clientes asíncronos oficiales.

No se utiliza `time.sleep()` ni ningún cliente síncrono.

### `LLMFactory`

Selecciona el cliente en función de `LLM_PROVIDER`.

### `AsyncLLMManager`

Es la fachada consumida por la aplicación:

```python
response = await manager.generate(messages)
```

Streaming:

```python
async for chunk in manager.stream(messages):
    print(chunk, end="", flush=True)
```

El código consumidor no cambia al intercambiar proveedor.

## Manejo de errores

Se capturan errores de:

- rate limit;
- timeout;
- conexión;
- estado HTTP de la API.

En modo normal se devuelve un `ModelResponse` controlado en lugar de propagar
el error hasta `main.py`.

En streaming, el generador entrega un fragmento `[ERROR] ...` si ocurre una
falla.

`MAX_RETRIES` se configura en los SDKs oficiales para aprovechar su mecanismo
de reintentos.

## Nota sobre `temperature` y Anthropic

La consigna exige validar una temperatura de `0` a `2`.

Versiones recientes del SDK de Anthropic retiraron `temperature` de la firma
tipada de `messages.create()` para los modelos más nuevos. Por esa razón el
ejemplo usa `claude-sonnet-4-6` y envía el valor mediante `extra_body`.

Si se configura un modelo Anthropic que no acepta sampling personalizado,
se debe omitir ese parámetro para ese modelo.

## Seguridad

Las API keys:

- se cargan desde `.env`;
- no están hardcodeadas;
- usan `SecretStr`;
- `.env` está ignorado por Git.

## Criterios de la entrega

| Criterio | Implementación |
| --- | --- |
| Python 3.12 | Sí |
| OpenAI + Anthropic | Sí |
| Interfaz común | `BaseLLMClient` |
| Manager unificado | `AsyncLLMManager` |
| Async/await | Sí |
| Streaming | `AsyncIterator` + `yield` |
| Pydantic | `schemas.py` |
| Temperature 0-2 | `Field(ge=0, le=2)` |
| max_tokens | Validado |
| `.env.example` | Sí |
| Rate limit controlado | Sí |
| Errores de red controlados | Sí |
| Retry | `MAX_RETRIES` |
| Script de validación | `main.py` |
